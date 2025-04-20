import sys
sys.path.append(".")
sys.path.append("..")
from glob import glob
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="True"
from argparse import ArgumentParser
from tqdm import tqdm
import torch
import torch.nn.functional as F
import numpy as np
from utils import utils
import cv2
import random
from shaders.shadow_mapping import ShadowMapping
from shaders.lambert import Lambert
from shaders.disney import Disney
import kornia
from losses.repulsive_loss import RepulsiveLoss
import json

class OptimizeAls:
    def __init__(self, opts):
        self.opts = opts
        os.makedirs(self.opts.out_dir, exist_ok=True)
        
        # Save options
        with open('%s/opt.json' % self.opts.out_dir, 'w') as f:
            json.dump(vars(self.opts), f, indent=4, sort_keys=True)

        self.env_paths = sorted(glob('%s/*/*.hdr' % self.opts.env_dir))[self.opts.start:self.opts.end]

        self.mask_sphere = utils.np2torch(cv2.imread('%s/mask_sphere.png' % self.opts.sphere_dir, cv2.IMREAD_GRAYSCALE)[None,None].astype(np.float32),self.opts.device)/255.
        self.normal_gt_sphere = utils.np2torch(cv2.imread('%s/normal_sphere.png' % self.opts.sphere_dir,cv2.IMREAD_COLOR).astype(np.float32),self.opts.device).permute(2,0,1)[None,[2,1,0]]/255.
        self.normal_gt_sphere[:,0] = 2.*self.normal_gt_sphere[:,0]-1.
        self.normal_gt_sphere[:,1] = -(2.*self.normal_gt_sphere[:,1]-1.)
        self.normal_gt_sphere = self.normal_gt_sphere / np.linalg.norm(self.normal_gt_sphere, axis=1, keepdims=True).clip(self.opts.eps, None)
        self.normal_gt_sphere_and_plane = utils.np2torch(cv2.imread('%s/normal_sphere_and_plane.png' % self.opts.sphere_dir,cv2.IMREAD_COLOR).astype(np.float32),self.opts.device).permute(2,0,1)[None,[2,1,0]]/255.
        self.normal_gt_sphere_and_plane[:,0] = 2.*self.normal_gt_sphere_and_plane[:,0]-1.
        self.normal_gt_sphere_and_plane[:,1] = -(2.*self.normal_gt_sphere_and_plane[:,1]-1.)
        self.normal_gt_sphere_and_plane = self.normal_gt_sphere_and_plane / np.linalg.norm(self.normal_gt_sphere_and_plane, axis=1, keepdims=True).clip(self.opts.eps, None)
        self.depth_gt_sphere_and_plane = utils.np2torch(cv2.imread('%s/depth_sphere_and_plane.exr' % self.opts.sphere_dir,-1),self.opts.device)[None,None,...,0]

        self.shadow_mapping = ShadowMapping(self.opts, resolution=self.opts.resolution_optimize)
        self.lambert = Lambert().lambert
        self.disney = Disney(self.opts).disney
        self.ones = torch.ones((1,1,self.opts.resolution_optimize,self.opts.resolution_optimize)).to(self.opts.device, non_blocking=True)
        self.mask_edge = torch.zeros_like(self.ones)
        self.mask_edge[:,:,1:self.opts.resolution_optimize-1,1:self.opts.resolution_optimize-1] = 1
        self.repulsive_loss = RepulsiveLoss().repulsive_loss

    def optimize_dir_and_inten(self, env_name, out_dir):
        # Prepare data
        diffuse_shading_wo_shadow_sphere_gt = torch.zeros((self.opts.n_rot,3,self.opts.resolution_optimize,self.opts.resolution_optimize)).to(self.opts.device, non_blocking=True)
        specular_shading_wo_shadow_sphere_gt = torch.zeros((self.opts.n_rot,3,self.opts.resolution_optimize,self.opts.resolution_optimize)).to(self.opts.device, non_blocking=True)
        for i in range(self.opts.n_rot):
            if i < (self.opts.n_rot - 2):
                theta = 2 * np.pi * i / (self.opts.n_rot - 2)
                diffuse_shading_wo_shadow_sphere_gt[i:i+1] = utils.np2torch(cv2.imread('%s/%s/diffuse_shading_wo_shadow_sphere__%.4f.exr' % (self.opts.sphere_dir,env_name,theta),-1),self.opts.device).permute(2,0,1)[None,[2,1,0]]
                specular_shading_wo_shadow_sphere_gt[i:i+1] = utils.np2torch(cv2.imread('%s/%s/specular_shading_wo_shadow_sphere__%.4f.exr' % (self.opts.sphere_dir,env_name,theta),-1),self.opts.device).permute(2,0,1)[None,[2,1,0]]
            elif i == (self.opts.n_rot - 2):
                diffuse_shading_wo_shadow_sphere_gt[i:i+1] = utils.np2torch(cv2.imread('%s/%s/diffuse_shading_wo_shadow_sphere__up.exr' % (self.opts.sphere_dir,env_name),-1),self.opts.device).permute(2,0,1)[None,[2,1,0]]
                specular_shading_wo_shadow_sphere_gt[i:i+1] = utils.np2torch(cv2.imread('%s/%s/specular_shading_wo_shadow_sphere__up.exr' % (self.opts.sphere_dir,env_name),-1),self.opts.device).permute(2,0,1)[None,[2,1,0]]
            elif i == (self.opts.n_rot - 1):
                diffuse_shading_wo_shadow_sphere_gt[i:i+1] = utils.np2torch(cv2.imread('%s/%s/diffuse_shading_wo_shadow_sphere__bottom.exr' % (self.opts.sphere_dir,env_name),-1),self.opts.device).permute(2,0,1)[None,[2,1,0]]
                specular_shading_wo_shadow_sphere_gt[i:i+1] = utils.np2torch(cv2.imread('%s/%s/specular_shading_wo_shadow_sphere__bottom.exr' % (self.opts.sphere_dir,env_name),-1),self.opts.device).permute(2,0,1)[None,[2,1,0]]
        diffuse_shading_wo_shadow_sphere_gt = self.mask_sphere * diffuse_shading_wo_shadow_sphere_gt
        specular_shading_wo_shadow_sphere_gt = self.mask_sphere * specular_shading_wo_shadow_sphere_gt
        

        # Optimization setting
        _intensity = torch.ones((self.opts.n_light,3)).to(self.opts.device, non_blocking=True)
        _direction = utils.np2torch(utils.fibonacci_sphere(self.opts.n_light),self.opts.device) #[n_light,3]
        _intensity.requires_grad = True
        _direction.requires_grad = True
        optimizer = torch.optim.Adam([_intensity,_direction], lr=self.opts.lr_max)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.opts.t_max, eta_min=self.opts.lr_min)

        log_dir_and_itr_path = '%s/log_dir_and_itr.txt' % out_dir
        utils.generate_log_txt(log_dir_and_itr_path,['itr','l1_loss_diffuse','l1_loss_specular','repulsive_loss','loss'])
        min_loss = 2e25
        for itr in tqdm(range(self.opts.n_itr_dir_and_inten),desc='itr,dir',ncols=50):
            optimizer.zero_grad(set_to_none=True)
            
            direction = F.normalize(_direction,dim=1) #[n_light,3]
            intensity = F.softplus(_intensity)

            als = torch.cat([direction,torch.zeros_like(intensity[:,0:1]),intensity],dim=1)[None] #[1,n_light,7]
            als_rot = torch.zeros_like(als).repeat(self.opts.n_rot,1,1)#[n_rot,n_light,7]
            for i in range(self.opts.n_rot):
                if i < (self.opts.n_rot - 2):
                    theta = 2 * np.pi * i / (self.opts.n_rot-2)
                    als_rot[i:i+1] = utils.rotation_als(als,theta)
                elif i == (self.opts.n_rot - 2):
                    theta = np.pi/2
                    als_rot[i:i+1] = utils.rotation_als(als,theta,vertical=True)
                elif i == (self.opts.n_rot - 1):
                    theta = -np.pi/2
                    als_rot[i:i+1] = utils.rotation_als(als,theta,vertical=True)

            diffuse_shading_wo_shadow_sphere_pred = self.lambert(self.normal_gt_sphere.expand(self.opts.n_rot,-1,-1,-1), als_rot)
            diffuse_shading_wo_shadow_sphere_pred = self.mask_sphere * diffuse_shading_wo_shadow_sphere_pred

            specular_shading_wo_shadow_sphere_pred = self.disney(self.normal_gt_sphere.expand(self.opts.n_rot,-1,-1,-1), als_rot, 0.5*self.ones, 0.5*self.ones)
            specular_shading_wo_shadow_sphere_pred = self.mask_sphere * specular_shading_wo_shadow_sphere_pred

            loss = 0
            loss_l1_diffuse = F.l1_loss(utils.lrgb2srgb(diffuse_shading_wo_shadow_sphere_gt.clamp(self.opts.eps,None)), utils.lrgb2srgb(diffuse_shading_wo_shadow_sphere_pred.clamp(self.opts.eps,None)))
            loss_l1_specular = F.l1_loss(utils.lrgb2srgb(specular_shading_wo_shadow_sphere_gt.clamp(self.opts.eps,None)), utils.lrgb2srgb(specular_shading_wo_shadow_sphere_pred.clamp(self.opts.eps,None)))
            loss_repulsive = self.repulsive_loss(als)
            loss = loss + loss_l1_diffuse + loss_l1_specular + 0.01*loss_repulsive

            loss.backward()
            optimizer.step()
            scheduler.step()

            with open(log_dir_and_itr_path,'a') as f:
                f.write('%d,%.9f,%.9f,%.9f,%.9f\n' % (itr,loss_l1_diffuse.item(),loss_l1_specular.item(),loss_repulsive.item(),loss.item()))

            if loss.item() < min_loss:
                min_loss = loss.item()
                directions_best = als_rot[:,:,0:3].detach().clone()
                intensities_best = als_rot[:,:,4:7].detach().clone()
                save_img = torch.cat([diffuse_shading_wo_shadow_sphere_pred,diffuse_shading_wo_shadow_sphere_gt,specular_shading_wo_shadow_sphere_pred,specular_shading_wo_shadow_sphere_gt],dim=2).detach().clone()

        del diffuse_shading_wo_shadow_sphere_gt, diffuse_shading_wo_shadow_sphere_pred, specular_shading_wo_shadow_sphere_gt, specular_shading_wo_shadow_sphere_pred
        torch.cuda.empty_cache()

        cv2.imwrite('%s/best_dir_and_itr.png' % out_dir, 
                    utils.torch2np(255 * utils.lrgb2srgb(save_img.permute(2,0,3,1).reshape(save_img.shape[2],-1,3)[...,[2,1,0]])))

        return directions_best, intensities_best


    def optimize_sigma(self, env_name, directions_best, intensities_best, out_dir):
        
        # Prepare data
        diffuse_shading_w_shadow_sphere_and_plane_gt = torch.zeros((self.opts.n_rot,3,self.opts.resolution_optimize,self.opts.resolution_optimize)).to(self.opts.device, non_blocking=True)
        for i in range(self.opts.n_rot):
            if i < (self.opts.n_rot - 2):
                theta = 2 * np.pi * i / (self.opts.n_rot-2)
                diffuse_shading_w_shadow_sphere_and_plane_gt[i:i+1] = utils.np2torch(cv2.imread('%s/%s/diffuse_shading_w_shadow_sphere_and_plane__%.4f.exr' % (self.opts.sphere_dir,env_name,theta),-1),self.opts.device).permute(2,0,1)[None,[2,1,0]]
            elif i == (self.opts.n_rot - 2):
                diffuse_shading_w_shadow_sphere_and_plane_gt[i:i+1] = utils.np2torch(cv2.imread('%s/%s/diffuse_shading_w_shadow_sphere_and_plane__up.exr' % (self.opts.sphere_dir,env_name),-1),self.opts.device).permute(2,0,1)[None,[2,1,0]]
            elif i == (self.opts.n_rot - 1):
                diffuse_shading_w_shadow_sphere_and_plane_gt[i:i+1] = utils.np2torch(cv2.imread('%s/%s/diffuse_shading_w_shadow_sphere_and_plane__bottom.exr' % (self.opts.sphere_dir,env_name),-1),self.opts.device).permute(2,0,1)[None,[2,1,0]]
        diffuse_shading_w_shadow_sphere_and_plane_gt = self.mask_edge * diffuse_shading_w_shadow_sphere_and_plane_gt

        diffuse_shading_wo_shadows_sphere_and_plane_pred = self.lambert(self.normal_gt_sphere_and_plane.expand(self.opts.n_rot,-1,-1,-1),
                                                                        torch.cat([directions_best,torch.zeros_like(intensities_best[:,:,0:1]),intensities_best],dim=2), per_light=True) #[n_rot,n_light,3]

        # Optimization setting
        _sigma = torch.zeros((self.opts.n_light,1)).to(self.opts.device, non_blocking=True) #[n_light,1]
        _sigma.requires_grad = True
        optimizer = torch.optim.Adam([_sigma], lr=self.opts.lr_max)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.opts.t_max, eta_min=self.opts.lr_min)


        log_sigma_path = '%s/log_sigma.txt' % out_dir
        utils.generate_log_txt(log_sigma_path,['itr','lap_loss','loss'])
        min_loss = 2e25
        for itr in tqdm(range(self.opts.n_itr_sigma),desc='sigma',ncols=50):
            optimizer.zero_grad(set_to_none=True)

            sigma = self.opts.range_sigma * torch.sigmoid(_sigma)[None].expand(self.opts.n_rot,-1,-1)
            als_rot = torch.cat([directions_best,sigma,intensities_best],dim=2) #[n_rot,n_light,7]

            _, soft_shadows = self.shadow_mapping(self.depth_gt_sphere_and_plane.expand(self.opts.n_rot,-1,-1,-1), als_rot, torch.ones_like(self.depth_gt_sphere_and_plane.expand(self.opts.n_rot,-1,-1,-1)))
            diffuse_shading_w_shadow_sphere_and_plane_pred = torch.einsum('bdhw,bdchw->bchw',
                                                                            soft_shadows,
                                                                            diffuse_shading_wo_shadows_sphere_and_plane_pred)
            diffuse_shading_w_shadow_sphere_and_plane_pred = self.mask_edge * diffuse_shading_w_shadow_sphere_and_plane_pred


            loss = 0
            loss_lap = 0
            for lap_size in self.opts.lap_sizes:
                shadow_lap = kornia.filters.laplacian(utils.lrgb2srgb(diffuse_shading_w_shadow_sphere_and_plane_gt.clamp(self.opts.eps,None)),lap_size)
                shadow_lap_hat = kornia.filters.laplacian(utils.lrgb2srgb(diffuse_shading_w_shadow_sphere_and_plane_pred.clamp(self.opts.eps,None)),lap_size)            
                loss_lap = loss_lap + F.l1_loss(shadow_lap, shadow_lap_hat)
            loss = loss + loss_lap

            loss.backward()
            optimizer.step()
            scheduler.step()

            with open(log_sigma_path,'a') as f:
                f.write('%d,%.9f,%.9f\n' % (itr,loss_lap.item(),loss.item())) #TODO

            if loss.item() < min_loss:
                min_loss = loss.item()
                als_best = als_rot[0:1].detach().clone()
                save_img = torch.cat([diffuse_shading_w_shadow_sphere_and_plane_pred,diffuse_shading_w_shadow_sphere_and_plane_gt],dim=2).detach().clone()

        cv2.imwrite('%s/best_sigma.png' % out_dir,
                    utils.torch2np(255 * utils.lrgb2srgb(save_img.permute(2,0,3,1).reshape(save_img.shape[2],-1,3)[...,[2,1,0]])))
        return als_best


    def optimization(self):
        for i_env,env_path in enumerate(self.env_paths):
            env_name = os.path.basename(env_path)[:-len('.hdr')]
            print('(%d/%d):%s' % (i_env+1,len(self.env_paths),env_name))
            out_dir = '%s/%s' % (self.opts.out_dir,env_name)
            os.makedirs(out_dir,exist_ok=True)

            directions_best, intensities_best = self.optimize_dir_and_inten(env_name,out_dir)

            als_best = self.optimize_sigma(env_name, directions_best, intensities_best,out_dir)

            np.save('%s/als.npy' % out_dir, utils.torch2np(als_best[0]))

            env = 255 * utils.lrgb2srgb(utils.np2torch(cv2.imread(env_path,-1),self.opts.device).permute(2,0,1)[None,[2,1,0]]).clamp(0,1)
            mapping = utils.mapping_als(env, als_best)
            cv2.imwrite('%s/mapping.png' % out_dir, mapping[...,[2,1,0]])


def main():
    parser = ArgumentParser()
    parser.add_argument('--device', default='cuda') # 'cuda' or 'cpu'
    parser.add_argument('--out_dir', default=None)
    parser.add_argument('--start', default=0, type=int)
    parser.add_argument('--end', default=99999, type=int)
    parser.add_argument('--t_max', default=20, type=int)
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--n_itr_dir_and_inten', default=1000, type=int)
    parser.add_argument('--n_itr_sigma', default=300, type=int)
    parser.add_argument('--env_dir', default=None)
    parser.add_argument('--sphere_dir', default='./data/sphere')
    parser.add_argument('--lr_max', default=1, type=float)
    parser.add_argument('--lr_min', default=0.00001, type=float)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--camera_distance', default=4.0, type=float)
    parser.add_argument('--focal_length', default=50, type=float)
    parser.add_argument('--resolution_optimize', default=256, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--n_rot', default=5+1+1, type=int)
    parser.add_argument('--n_light', default=16, type=int)
    parser.add_argument('--window', default=2, type=float)
    parser.add_argument('--depth_threshold', default=100000, type=float)
    parser.add_argument('--shadow_threshold', default=0.005, type=float)
    parser.add_argument('--eps', default=1e-6, type=float)
    parser.add_argument('--lap_sizes', nargs='+', type=int, default=[15,21,33])
    parser.add_argument('--range_sigma', default=20, type=float)

    opts = parser.parse_args()

    np.random.seed(opts.seed)
    random.seed(opts.seed)
    torch.manual_seed(opts.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(opts.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.use_deterministic_algorithms = False
    torch.autograd.set_detect_anomaly(True)
    OptimizeAls(opts).optimization()

if __name__ == '__main__':
    main()

