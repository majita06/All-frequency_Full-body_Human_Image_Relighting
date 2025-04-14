import os
from argparse import ArgumentParser
from glob import glob
from tqdm import tqdm
import torch
import numpy as np
from models.unet import UNet
from models.unet_multi import UNet_multi
from utils import utils
import cv2
from shaders.lambert import Lambert
from shaders.disney import Disney
from shaders.shadow_mapping import ShadowMapping


class EvalReal:
    def __init__(self, opts):
        self.opts = opts
        os.makedirs(self.opts.out_dir, exist_ok=True)

        self.img_dirs = sorted(glob('%s/*' % self.opts.img_dir))
        self.als_paths = sorted(glob('%s/*/als.npy' % self.opts.als_dir))

        self.net_firststage = UNet_multi(self.opts,in_channels=4).to(self.opts.device)
        self.net_firststage.load_state_dict(torch.load(self.opts.checkpoint_path_firststage)['model_state_dict'])
        self.net_firststage.eval()
        self.net_firststage.half()

        self.net_depth = UNet(self.opts,in_channels=4,out_channels=1).to(self.opts.device)
        self.net_depth.load_state_dict(torch.load(self.opts.checkpoint_path_depth)['model_state_dict'])
        self.net_depth.eval()

        self.net_refineshadow = UNet(self.opts,in_channels=4,out_channels=1,n_layer=3).to(self.opts.device)
        self.net_refineshadow.load_state_dict(torch.load(self.opts.checkpoint_path_refineshadow)['model_state_dict'])
        self.net_refineshadow.eval()
        self.net_refineshadow.half()

        self.lambert = Lambert().lambert
        self.disney = Disney(self.opts).disney
        self.shadow_mapping = ShadowMapping(self.opts,resolution=self.opts.resolution)

        self.ones_tile = torch.ones((1, 1, self.opts.resolution, self.opts.resolution)).to(self.opts.device)



    def recon(self, mask, albedo_pred, normal_pred, specular_pred, roughness_pred, depth_pred, depth_norm_pred, source_als_pred):
        dict_recon = {}

        mask_vis = torch.einsum('bxhw,bdx->bdhw',normal_pred,source_als_pred[:,:,0:3]).clamp(0,1)
        mask_vis[mask_vis>0] = 1
        with torch.no_grad():
            source_hard_shadows_pred, source_soft_shadows_pred = self.shadow_mapping(depth_pred, source_als_pred, mask, train=False)
        source_hard_shadows_pred = mask_vis * source_hard_shadows_pred
        source_soft_shadows_pred = mask_vis * source_soft_shadows_pred

        net_input = torch.cat([source_hard_shadows_pred.permute(1,0,2,3),
                                source_soft_shadows_pred.permute(1,0,2,3),
                                torch.einsum('d,ophw->dohw',source_als_pred[0,:,3]/self.opts.range_sigma,self.ones_tile),
                                mask * depth_norm_pred.expand(self.opts.n_light,-1,-1,-1)],dim=1)
        with torch.no_grad():
            source_shadows_pred = self.net_refineshadow(net_input.half()).float() 
        source_shadows_pred = torch.sigmoid(source_shadows_pred).permute(1,0,2,3)#[n_light, 1, h, w]->[1, n_light, h, w]

        source_diffuse_shading_wo_shadows_pred = self.lambert(normal_pred, source_als_pred, per_light=True)
        source_diffuse_shading_wo_shadows_pred = mask * source_diffuse_shading_wo_shadows_pred
        dict_recon['source_diffuse_shading_wo_shadow_pred'] = torch.sum(source_diffuse_shading_wo_shadows_pred,dim=1)

        source_specular_shading_wo_shadows_pred = self.disney(normal_pred, source_als_pred, specular_pred, roughness_pred, per_light=True)
        source_specular_shading_wo_shadows_pred = mask * source_specular_shading_wo_shadows_pred
        dict_recon['source_specular_shading_wo_shadow_pred'] = torch.sum(source_specular_shading_wo_shadows_pred,dim=1)
        
        dict_recon['source_diffuse_shading_w_shadow_pred'] = torch.einsum('bdhw,bdchw->bchw',
                                                                          source_shadows_pred,
                                                                          source_diffuse_shading_wo_shadows_pred)
        dict_recon['source_specular_shading_w_shadow_pred'] = torch.einsum('bdhw,bdchw->bchw',
                                                                           source_shadows_pred,
                                                                           source_specular_shading_wo_shadows_pred)

        dict_recon['source_rendering_wo_shadow_wo_specular_pred'] = albedo_pred * dict_recon['source_diffuse_shading_wo_shadow_pred']
        dict_recon['source_rendering_w_shadow_wo_specular_pred'] = albedo_pred * dict_recon['source_diffuse_shading_w_shadow_pred']
        dict_recon['source_rendering_wo_shadow_w_specular_pred'] = albedo_pred * (dict_recon['source_diffuse_shading_wo_shadow_pred'] + dict_recon['source_specular_shading_wo_shadow_pred'])
        dict_recon['source_rendering_w_shadow_w_specular_pred'] = albedo_pred * (dict_recon['source_diffuse_shading_w_shadow_pred'] + dict_recon['source_specular_shading_w_shadow_pred'])                    
        return dict_recon
    

    def relit(self, mask, albedo_pred, normal_pred, specular_pred, roughness_pred, depth_pred, depth_norm_pred, target_als):
        dict_relit = {}

        mask_vis = torch.einsum('bxhw,bdx->bdhw',normal_pred,target_als[:,:,0:3]).clamp(0,1)
        mask_vis[mask_vis>0] = 1
        with torch.no_grad():
            target_hard_shadows_pred, target_soft_shadows_pred = self.shadow_mapping(depth_pred, target_als, mask, train=False) #[1, n_light, h, w]
        target_hard_shadows_pred = mask_vis * target_hard_shadows_pred
        target_soft_shadows_pred = mask_vis * target_soft_shadows_pred
        
        net_input = torch.cat([target_hard_shadows_pred.permute(1,0,2,3),
                                target_soft_shadows_pred.permute(1,0,2,3),
                                torch.einsum('d,ophw->dohw',target_als[0,:,3]/self.opts.range_sigma,self.ones_tile),
                                mask * depth_norm_pred.expand(self.opts.n_light,-1,-1,-1)],dim=1)
        with torch.no_grad():
            target_shadows_pred = self.net_refineshadow(net_input.half()).float()
        target_shadows_pred = torch.sigmoid(target_shadows_pred).permute(1,0,2,3)#[n_light, 1, h, w]->[1,n_light, h, w]

        target_diffuse_shading_wo_shadows_pred = self.lambert(normal_pred, target_als, per_light=True)
        target_diffuse_shading_wo_shadows_pred = mask * target_diffuse_shading_wo_shadows_pred
        dict_relit['target_diffuse_shading_wo_shadow_pred'] = torch.sum(target_diffuse_shading_wo_shadows_pred,dim=1)

        target_specular_shading_wo_shadows_pred = self.disney(normal_pred, target_als, specular_pred, roughness_pred, per_light=True)
        target_specular_shading_wo_shadows_pred = mask * target_specular_shading_wo_shadows_pred
        dict_relit['target_specular_shading_wo_shadow_pred'] = torch.sum(target_specular_shading_wo_shadows_pred,dim=1)
        
        dict_relit['target_diffuse_shading_w_shadow_pred'] = torch.einsum('bdhw,bdchw->bchw',
                                                            target_shadows_pred,
                                                            target_diffuse_shading_wo_shadows_pred)
        dict_relit['target_specular_shading_w_shadow_pred'] = torch.einsum('bdhw,bdchw->bchw',
                                                            target_shadows_pred,
                                                            target_specular_shading_wo_shadows_pred)

        dict_relit['target_rendering_wo_shadow_wo_specular_pred'] = albedo_pred * dict_relit['target_diffuse_shading_wo_shadow_pred']
        dict_relit['target_rendering_w_shadow_wo_specular_pred'] = albedo_pred * dict_relit['target_diffuse_shading_w_shadow_pred']
        dict_relit['target_rendering_wo_shadow_w_specular_pred'] = albedo_pred * (dict_relit['target_diffuse_shading_wo_shadow_pred'] + dict_relit['target_specular_shading_wo_shadow_pred'])
        dict_relit['target_rendering_w_shadow_w_specular_pred'] = albedo_pred * (dict_relit['target_diffuse_shading_w_shadow_pred'] + dict_relit['target_specular_shading_w_shadow_pred'])

        dict_relit['target_diffuse_shading_w_hardshadow_pred'] = torch.einsum('bdhw,bdchw->bchw',
                                                                target_hard_shadows_pred,
                                                                target_diffuse_shading_wo_shadows_pred)
        dict_relit['target_diffuse_shading_w_softshadow_pred'] = torch.einsum('bdhw,bdchw->bchw',
                                                                target_soft_shadows_pred,
                                                                target_diffuse_shading_wo_shadows_pred)
        return dict_relit

    def evaluation(self):
        for img_dir in self.img_dirs:
            human_id = os.path.basename(img_dir)
            print('Processing %s' % human_id)
            out_dir = '%s/%s' % (self.opts.out_dir, human_id)
            os.makedirs(out_dir, exist_ok=True)

            img_path = '%s/input_w_bg.jpg' % img_dir
            mask_path = '%s/mask.png' % img_dir

            img_w_bg = utils.np2torch(cv2.imread(img_path, cv2.IMREAD_COLOR).astype(np.float32),self.opts.device).permute(2,0,1)[None,[2,1,0]]/255.
            mask = utils.np2torch(cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE).astype(np.float32),self.opts.device)[None,None]/255.
            img_w_bg, mask = utils.centering_img(img_w_bg, mask)
            mask[mask < 0.5] = 0
            mask[mask >= 0.5] = 1
            cv2.imwrite('%s/input_img.png' % out_dir, utils.torch2np(255 * img_w_bg[0,[2,1,0]].permute(1,2,0)))


            '''
            intrinsics estimation
            '''
            net_input = torch.cat([img_w_bg, mask], dim=1)
            with torch.no_grad():
                albedo_pred, normal_pred, specular_pred, roughness_pred, source_als_pred = self.net_firststage(net_input.half())
            albedo_pred = mask * albedo_pred.float()
            normal_pred = mask * normal_pred.float()
            specular_pred = mask * specular_pred.float()
            roughness_pred = mask * roughness_pred.float()
            source_als_pred = source_als_pred.float()

            with torch.no_grad():
                depth_norm_pred = self.net_depth(net_input)
            depth_norm_pred = torch.sigmoid(depth_norm_pred)
            depth_norm_pred = depth_norm_pred - torch.median(depth_norm_pred[mask==1]) + (self.opts.camera_distance - self.opts.d_all_min) / (self.opts.d_all_max - self.opts.d_all_min)
            depth_pred = (self.opts.d_all_max - self.opts.d_all_min) * depth_norm_pred + self.opts.d_all_min
            
            intrinsics_save_dir = '%s/intrinsics' % out_dir
            os.makedirs(intrinsics_save_dir, exist_ok=True)
            cv2.imwrite('%s/albedo.png' % intrinsics_save_dir, utils.torch2np(255 * utils.lrgb2srgb(utils.clip_img(albedo_pred,mask)[0][0,[2,1,0]].permute(1,2,0))))
            cv2.imwrite('%s/normal.png' % intrinsics_save_dir, utils.torch2np(255 * utils.clip_img(mask*0.5*(normal_pred+1),mask)[0][0,[2,1,0]].permute(1,2,0)))
            cv2.imwrite('%s/specular.png' % intrinsics_save_dir, utils.torch2np(255 * utils.clip_img(specular_pred,mask)[0][0,0]))
            cv2.imwrite('%s/roughness.png' % intrinsics_save_dir, utils.torch2np(255 * utils.clip_img(roughness_pred,mask)[0][0,0]))
            cv2.imwrite('%s/depth.png' % intrinsics_save_dir, cv2.applyColorMap((255 * utils.torch2np(utils.clip_img(mask*depth_norm_pred,mask)[0][0,0])).astype(np.uint8),cv2.COLORMAP_INFERNO))

            '''
            reconstruction
            '''
            dict_recon = self.recon(mask, albedo_pred, normal_pred, specular_pred, roughness_pred, depth_pred, depth_norm_pred, source_als_pred)

            recon_save_dir = '%s/recon' % out_dir
            os.makedirs(recon_save_dir, exist_ok=True)
            cv2.imwrite('%s/source_diffuse_shading_wo_shadow.png' % recon_save_dir,
                        utils.torch2np(255 * utils.lrgb2srgb(utils.clip_img(dict_recon['source_diffuse_shading_wo_shadow_pred'],mask)[0][0,[2,1,0]].permute(1,2,0).clamp(0,1))))
            cv2.imwrite('%s/source_specular_shading_wo_shadow.png' % recon_save_dir,
                        utils.torch2np(255 * utils.lrgb2srgb(utils.clip_img(dict_recon['source_specular_shading_wo_shadow_pred'],mask)[0][0,[2,1,0]].permute(1,2,0).clamp(0,1))))
            cv2.imwrite('%s/source_diffuse_shading_w_shadow.png' % recon_save_dir,
                        utils.torch2np(255 * utils.lrgb2srgb(utils.clip_img(dict_recon['source_diffuse_shading_w_shadow_pred'],mask)[0][0,[2,1,0]].permute(1,2,0).clamp(0,1))))
            cv2.imwrite('%s/source_specular_shading_w_shadow.png' % recon_save_dir,
                        utils.torch2np(255 * utils.lrgb2srgb(utils.clip_img(dict_recon['source_specular_shading_w_shadow_pred'],mask)[0][0,[2,1,0]].permute(1,2,0).clamp(0,1))))

            cv2.imwrite('%s/source_rendering_wo_shadow_wo_specular.png' % recon_save_dir,
                        utils.torch2np(255 * utils.lrgb2srgb(utils.clip_img(dict_recon['source_rendering_wo_shadow_wo_specular_pred'],mask)[0][0,[2,1,0]].permute(1,2,0).clamp(0,1))))
            cv2.imwrite('%s/source_rendering_w_shadow_wo_specular.png' % recon_save_dir,
                        utils.torch2np(255 * utils.lrgb2srgb(utils.clip_img(dict_recon['source_rendering_w_shadow_wo_specular_pred'],mask)[0][0,[2,1,0]].permute(1,2,0).clamp(0,1))))
            cv2.imwrite('%s/source_rendering_wo_shadow_w_specular.png' % recon_save_dir,
                        utils.torch2np(255 * utils.lrgb2srgb(utils.clip_img(dict_recon['source_rendering_wo_shadow_w_specular_pred'],mask)[0][0,[2,1,0]].permute(1,2,0).clamp(0,1))))
            cv2.imwrite('%s/source_rendering_w_shadow_w_specular.png' % recon_save_dir,
                        utils.torch2np(255 * utils.lrgb2srgb(utils.clip_img(dict_recon['source_rendering_w_shadow_w_specular_pred'],mask)[0][0,[2,1,0]].permute(1,2,0).clamp(0,1))))
            

            print('Relighting')
            for als_path in self.als_paths:
                env_name = os.path.basename(als_path[:-len('/als.npy')])
                relit_save_dir = '%s/%s' % (out_dir, env_name)
                os.makedirs(relit_save_dir, exist_ok=True)

                target_als = utils.np2torch(np.load(als_path),self.opts.device)[None]
                for i_frame in tqdm(range(self.opts.n_frame),ncols=50,desc=env_name):
                    theta = 2 * np.pi * i_frame / self.opts.n_frame
                    target_als_rot = utils.rotation_als(target_als, theta)
                    
                    '''
                    relighting
                    '''
                    dict_relit = self.relit(mask, albedo_pred, normal_pred, specular_pred, roughness_pred, depth_pred, depth_norm_pred, target_als_rot)

                    cv2.imwrite('%s/target_diffuse_shading_w_hardshadow__%03d.png' % (relit_save_dir,i_frame),
                                utils.torch2np(255 * utils.lrgb2srgb(utils.clip_img(dict_relit['target_diffuse_shading_w_hardshadow_pred'],mask)[0][0,[2,1,0]].permute(1,2,0).clamp(0,1))))
                    cv2.imwrite('%s/target_diffuse_shading_w_softshadow__%03d.png' % (relit_save_dir,i_frame),
                                utils.torch2np(255 * utils.lrgb2srgb(utils.clip_img(dict_relit['target_diffuse_shading_w_softshadow_pred'],mask)[0][0,[2,1,0]].permute(1,2,0).clamp(0,1))))


                    cv2.imwrite('%s/target_diffuse_shading_wo_shadow__%03d.png' % (relit_save_dir,i_frame),
                                utils.torch2np(255 * utils.lrgb2srgb(utils.clip_img(dict_relit['target_diffuse_shading_wo_shadow_pred'],mask)[0][0,[2,1,0]].permute(1,2,0).clamp(0,1))))
                    cv2.imwrite('%s/target_specular_shading_wo_shadow__%03d.png' % (relit_save_dir,i_frame),
                                utils.torch2np(255 * utils.lrgb2srgb(utils.clip_img(dict_relit['target_specular_shading_wo_shadow_pred'],mask)[0][0,[2,1,0]].permute(1,2,0).clamp(0,1))))
                    cv2.imwrite('%s/target_diffuse_shading_w_shadow__%03d.png' % (relit_save_dir,i_frame),
                                utils.torch2np(255 * utils.lrgb2srgb(utils.clip_img(dict_relit['target_diffuse_shading_w_shadow_pred'],mask)[0][0,[2,1,0]].permute(1,2,0).clamp(0,1))))
                    cv2.imwrite('%s/target_specular_shading_w_shadow__%03d.png' % (relit_save_dir,i_frame),
                                utils.torch2np(255 * utils.lrgb2srgb(utils.clip_img(dict_relit['target_specular_shading_w_shadow_pred'],mask)[0][0,[2,1,0]].permute(1,2,0).clamp(0,1))))

                    cv2.imwrite('%s/target_rendering_wo_shadow_wo_specular__%03d.png' % (relit_save_dir,i_frame),
                                utils.torch2np(255 * utils.lrgb2srgb(utils.clip_img(dict_relit['target_rendering_wo_shadow_wo_specular_pred'],mask)[0][0,[2,1,0]].permute(1,2,0).clamp(0,1))))
                    cv2.imwrite('%s/target_rendering_w_shadow_wo_specular__%03d.png' % (relit_save_dir,i_frame),
                                utils.torch2np(255 * utils.lrgb2srgb(utils.clip_img(dict_relit['target_rendering_w_shadow_wo_specular_pred'],mask)[0][0,[2,1,0]].permute(1,2,0).clamp(0,1))))
                    cv2.imwrite('%s/target_rendering_wo_shadow_w_specular__%03d.png' % (relit_save_dir,i_frame),
                                utils.torch2np(255 * utils.lrgb2srgb(utils.clip_img(dict_relit['target_rendering_wo_shadow_w_specular_pred'],mask)[0][0,[2,1,0]].permute(1,2,0).clamp(0,1))))
                                
                    cv2.imwrite('%s/target_rendering_w_shadow_w_specular__%03d.png' % (relit_save_dir,i_frame),
                                utils.torch2np(255 * utils.lrgb2srgb(utils.clip_img(dict_relit['target_rendering_w_shadow_w_specular_pred'],mask)[0][0,[2,1,0]].permute(1,2,0).clamp(0,1))))
                
                # Generate video from the frames created above using ffmpeg
                frame_paths = '%s/target_rendering_w_shadow_w_specular__%%03d.png' % relit_save_dir
                save_video_path = '%s.mp4' % relit_save_dir
                utils.generate_video(save_video_path, frame_paths, self.opts.fps, disp_log=False)
                

def main(): 
    parser = ArgumentParser()
    parser.add_argument('--device', default='cuda') # cuda or cpu
    parser.add_argument('--out_dir', default='/home/tajima/All-frequency_Full-body_Human_Image_Relighting/outputs/eval_real')
    parser.add_argument('--batch_size', default=1, type=int) #FIX
    parser.add_argument('--img_dir', default='/home/tajima/dataset/realimg/unsplash')
    parser.add_argument('--checkpoint_path_firststage', default='/home/tajima/All-frequency_Full-body_Human_Image_Relighting/outputs/train_firststage/checkpoints/0019epoch.pth')
    parser.add_argument('--checkpoint_path_depth', default='/home/tajima/All-frequency_Full-body_Human_Image_Relighting/outputs/train_depth/checkpoints/0179epoch.pth')
    parser.add_argument('--checkpoint_path_refineshadow', default='/home/tajima/All-frequency_Full-body_Human_Image_Relighting/outputs/train_refineshadow/checkpoints/0019epoch.pth')
    parser.add_argument('--d_all_min', default=2.873195, type=float)
    parser.add_argument('--d_all_max', default=5.990806, type=float)
    parser.add_argument('--camera_distance', default=4.0, type=float)
    parser.add_argument('--focal_length', default=50, type=float)
    parser.add_argument('--depth_threshold', default=100000, type=float)
    parser.add_argument('--shadow_threshold', default=0.005, type=float)
    parser.add_argument('--window', default=2, type=float)
    parser.add_argument('--range_sigma', default=20, type=float)
    parser.add_argument('--resolution_optimize', default=256, type=int)
    parser.add_argument('--resolution', default=1024, type=int)
    parser.add_argument('--eps', default=1e-6, type=float)
    parser.add_argument('--als_dir', default='/home/tajima/dataset/EG25/als')
    parser.add_argument('--n_light', default=16, type=int)
    parser.add_argument('--n_frame', default=120, type=int)
    parser.add_argument('--fps', default=24, type=float)
    opts = parser.parse_args()

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.use_deterministic_algorithms = False
    torch.autograd.set_detect_anomaly(True)

    EvalReal(opts).evaluation()

if __name__ == '__main__':
    main()