import sys
sys.path.append(".")
sys.path.append("..")
import os
from argparse import ArgumentParser
from tqdm import tqdm
from torch.utils.data import DataLoader
from collections import defaultdict
import torch
import torch.nn.functional as F
import numpy as np
from datasets.images_dataset import ImagesDataset
from models.unet import UNet
from utils import utils, metrics
import cv2
import random
from shaders.shadow_mapping import ShadowMapping
from shaders.lambert import Lambert
import json

class TrainRefineShadow:
    def __init__(self, opts):
        self.opts = opts
        os.makedirs(self.opts.out_dir, exist_ok=True)

        # Save options
        with open('%s/opt.json' % self.opts.out_dir, 'w') as f:
            json.dump(vars(self.opts), f, indent=4, sort_keys=True)

        self.net = UNet(self.opts,in_channel=4,out_channel=1,n_layer=3).to(self.opts.device)
        if self.opts.checkpoint_path is not None:
            self.net.load_state_dict(torch.load(self.opts.checkpoint_path)['model_state_dict'])
        
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.opts.lr_max)
        if self.opts.checkpoint_path is not None:
            self.optimizer.load_state_dict(torch.load(self.opts.checkpoint_path)['optimizer_state_dict'])
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.opts.t_max, eta_min=self.opts.lr_min)
        if self.opts.checkpoint_path is not None:
            self.scheduler.load_state_dict(torch.load(self.opts.checkpoint_path)['scheduler_state_dict'])


        g = torch.Generator()
        g.manual_seed(self.opts.seed)
        self.train_dataset = ImagesDataset(opts=self.opts,train_or_val='train',id='refineshadow')
        self.val_dataset = ImagesDataset(opts=self.opts,train_or_val='val',id='refineshadow')
        self.train_dataloader = DataLoader(self.train_dataset,
                                           batch_size=self.opts.batch_size,
                                           shuffle=True,
                                           num_workers=self.opts.num_workers,
                                           drop_last=True,
                                           pin_memory=True,
                                           worker_init_fn=self.seed_worker,
                                           generator=g)
        self.val_dataloader = DataLoader(self.val_dataset,
                                          batch_size=self.opts.batch_size,
                                          shuffle=True,
                                          num_workers=self.opts.num_workers,
                                          drop_last=True,
                                          pin_memory=True,
                                          worker_init_fn=self.seed_worker,
                                          generator=g)
        self.log_train_path = '%s/log_train.txt' % self.opts.out_dir
        self.log_val_path = '%s/log_val.txt' % self.opts.out_dir

        self.list_loss_name = ['l1_loss']
        self.list_metric_name = ['rmse_w_mask', 'ssim_w_mask']#, 'lpips']
        utils.generate_log_txt(self.log_train_path,['epoch'] + self.list_loss_name)
        utils.generate_log_txt(self.log_val_path,['epoch'] + self.list_metric_name)

        #self.lpips = metrics.LPIPS(self.opts).lpips

        self.shadow_mapping = ShadowMapping(self.opts,resolution=self.opts.resolution)
        self.lambert = Lambert().lambert
        self.ones_tile = torch.ones((1,1,self.opts.resolution,self.opts.resolution)).to(self.opts.device, non_blocking=True)

        self.checkpoint_save_dir = '%s/checkpoints' % self.opts.out_dir
        os.makedirs(self.checkpoint_save_dir, exist_ok=True)

        self.scaler = torch.amp.GradScaler(self.opts.device,enabled=self.opts.amp)


    def seed_worker(self, worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)


    def train_val(self, epoch, is_train):
        self.net.train() if is_train else self.net.eval()
        list_log = defaultdict(list)
        for itr, data_dict in enumerate(tqdm(self.train_dataloader if is_train else self.val_dataloader,desc='%d' % epoch,ncols=50)):
            with torch.set_grad_enabled(is_train):
                with torch.autocast(device_type=self.opts.device,dtype=torch.float16,enabled=self.opts.amp):
                    if self.opts.debug and itr >= 5:
                        break
                    if is_train:
                        self.optimizer.zero_grad(set_to_none=True)

                    human_id = data_dict['human_id']
                    mask = data_dict['mask'].to(self.opts.device, non_blocking=True)
                    normal_gt = data_dict['normal'].to(self.opts.device, non_blocking=True)
                    depth_gt = data_dict['depth'].to(self.opts.device, non_blocking=True)
                    source_als = data_dict['source_als'].to(self.opts.device, non_blocking=True)
                    source_diffuse_shading_w_shadow_gt = data_dict['source_diffuse_shading_w_shadow'].to(self.opts.device, non_blocking=True)
                    if self.opts.use_prepare_shadow:
                        source_hard_shadows = data_dict['source_hard_shadows'].to(self.opts.device, non_blocking=True)
                        source_soft_shadows = data_dict['source_soft_shadows'].to(self.opts.device, non_blocking=True)

                    if np.random.rand() > 0.5:
                        mask = torch.flip(mask, [3])
                        normal_gt = torch.flip(normal_gt, [3])
                        normal_gt[:,0] *= -1
                        depth_gt = torch.flip(depth_gt, [3])
                        source_als[:,:,0] *= -1
                        source_diffuse_shading_w_shadow_gt = torch.flip(source_diffuse_shading_w_shadow_gt, [3])
                        if self.opts.use_prepare_shadow:
                            source_hard_shadows = torch.flip(source_hard_shadows, [3])
                            source_soft_shadows = torch.flip(source_soft_shadows, [3])
                    source_diffuse_shading_w_shadow_gt = mask * source_diffuse_shading_w_shadow_gt

                    
                    if not self.opts.use_prepare_shadow:
                        mask_vis = torch.einsum('bxhw,bdx->bdhw',normal_gt,source_als[:,:,0:3]).clamp(0,1)
                        mask_vis[mask_vis>0] = 1
                        with torch.no_grad():
                            source_hard_shadows, source_soft_shadows = self.shadow_mapping(depth_gt, source_als, mask, train=False)
                        source_hard_shadows = mask_vis * source_hard_shadows
                        source_soft_shadows = mask_vis * source_soft_shadows


                    depth_norm_gt = depth_gt.clone()
                    depth_norm_gt[mask == 0] = float('nan')
                    depth_norm_gt = depth_gt + (self.opts.camera_distance - torch.nanmedian(depth_norm_gt.view(self.opts.batch_size,-1),dim=1)[0][:,None,None,None])
                    depth_norm_gt = ((depth_norm_gt - self.opts.d_all_min) / (self.opts.d_all_max - self.opts.d_all_min))
                
                
                    source_refine_shadows = torch.zeros_like(source_hard_shadows)
                    for i_light in range(self.opts.n_light):
                        net_input = torch.cat([source_hard_shadows[:,i_light:i_light+1],
                                                source_soft_shadows[:,i_light:i_light+1],
                                                torch.einsum('bo,bohw->bohw',source_als[:,i_light:i_light+1,3]/self.opts.range_sigma,self.ones_tile),
                                                mask * depth_norm_gt],dim=1)
                        source_refine_shadows[:,i_light:i_light+1] = self.net(net_input)
                    source_refine_shadows = torch.sigmoid(source_refine_shadows)


                    source_diffuse_shading_wo_shadows = self.lambert(normal_gt, source_als, per_light=True)
                    source_diffuse_shading_w_shadow_pred = torch.einsum('bdhw,bdchw->bchw',
                                                                source_refine_shadows,
                                                                source_diffuse_shading_wo_shadows)
                    source_diffuse_shading_w_shadow_pred = mask * source_diffuse_shading_w_shadow_pred

                    if is_train:
                        loss = 0
                        for loss_name in self.list_loss_name:
                            if loss_name == 'l1_loss':
                                _loss = F.l1_loss(utils.lrgb2srgb(source_diffuse_shading_w_shadow_gt.clamp(self.opts.eps,None)),
                                                  utils.lrgb2srgb(source_diffuse_shading_w_shadow_pred.clamp(self.opts.eps,None)))
                                list_log[loss_name].append(_loss.item())
                                loss = loss + _loss

                        self.scaler.scale(loss).backward()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        for metric_name in self.list_metric_name:
                            if metric_name == 'rmse_w_mask':
                                list_log[metric_name].append(metrics.rmse_w_mask(utils.lrgb2srgb(source_diffuse_shading_w_shadow_gt),
                                                                                 utils.lrgb2srgb(source_diffuse_shading_w_shadow_pred),
                                                                                 mask))
                            elif metric_name == 'ssim_w_mask':
                                list_log[metric_name].append(metrics.ssim_w_mask(utils.lrgb2srgb(source_diffuse_shading_w_shadow_gt),
                                                                                 utils.lrgb2srgb(source_diffuse_shading_w_shadow_pred),
                                                                                 mask))
                            elif metric_name == 'lpips':
                                list_log[metric_name].append(self.lpips(utils.lrgb2srgb(source_diffuse_shading_w_shadow_gt),
                                                                        utils.lrgb2srgb(source_diffuse_shading_w_shadow_pred)))
        if is_train:
            self.scheduler.step()
   
        with open(self.log_train_path if is_train else self.log_val_path,"a") as f:
            f.write('%d' % epoch)
            for key in self.list_loss_name if is_train else self.list_metric_name:
                f.write(',%.7f' % np.mean(np.array(list_log[key])))
            f.write('\n')

        if ((epoch + 1) / self.opts.t_max) % 2 == 1:
            batch_save = 0
            cv2.imwrite('%s/%04depoch_%s_%s.png' % (self.opts.out_dir,epoch,'train' if is_train else 'val',human_id[batch_save]),
                        utils.torch2np(255 * utils.lrgb2srgb(torch.cat([utils.clip_img((mask * torch.sum(source_diffuse_shading_wo_shadows,dim=1))[batch_save:batch_save+1],mask[batch_save:batch_save+1])[0],
                                                                        utils.clip_img(torch.einsum('bdhw,bdchw->bchw',
                                                                                        source_hard_shadows,
                                                                                        source_diffuse_shading_wo_shadows.float())[batch_save:batch_save+1],mask[batch_save:batch_save+1])[0],
                                                                        utils.clip_img(torch.einsum('bdhw,bdchw->bchw',
                                                                                        source_soft_shadows,
                                                                                        source_diffuse_shading_wo_shadows.float())[batch_save:batch_save+1],mask[batch_save:batch_save+1])[0],
                                                                        utils.clip_img(source_diffuse_shading_w_shadow_pred[batch_save:batch_save+1],mask[batch_save:batch_save+1])[0],
                                                                        utils.clip_img(source_diffuse_shading_w_shadow_gt[batch_save:batch_save+1],mask[batch_save:batch_save+1])[0]],
                                                                        dim=3)[0,[2,1,0]].permute(1,2,0))))
            checkpoint = {
                'model_state_dict': self.net.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
            }
            torch.save(checkpoint, '%s/%04depoch.pth' % (self.checkpoint_save_dir,epoch))
            
    def training(self):
        for epoch in range(self.opts.n_epoch):
            self.train_val(epoch,is_train=True)
            if ((epoch + 1) / self.opts.t_max) % 2 == 1:
                self.train_val(epoch,is_train=False)
    
def main():
    parser = ArgumentParser()
    parser.add_argument('--device', default='cuda') # 'cuda' or 'cpu'
    parser.add_argument('--out_dir', default='./outputs/train_refineshadow')
    parser.add_argument('--t_max', default=20, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--n_epoch', default=1000, type=int)
    parser.add_argument('--dataset_dir', default=None)
    parser.add_argument('--n_train_samples', default=6000, type=int)
    parser.add_argument('--lr_max', default=0.001, type=float)
    parser.add_argument('--lr_min', default=0.00001, type=float)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--d_all_min', default=2.873195, type=float)
    parser.add_argument('--d_all_max', default=5.990806, type=float)
    parser.add_argument('--camera_distance', default=4.0, type=float)
    parser.add_argument('--focal_length', default=50, type=float)
    parser.add_argument('--resolution', default=1024, type=int)
    parser.add_argument('--resolution_optimize', default=256, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--als_dir', default=None)
    parser.add_argument('--n_light', default=16, type=int)
    parser.add_argument('--window', default=2, type=float)
    parser.add_argument('--depth_threshold', default=100000, type=float)
    parser.add_argument('--shadow_threshold', default=0.005, type=float)
    parser.add_argument('--eps', default=1e-6, type=float)
    parser.add_argument('--range_sigma', default=20, type=float)
    parser.add_argument('--use_other_dataset', action='store_true')
    parser.add_argument('--other_dataset_dir', default=None)
    parser.add_argument('--use_prepare_shadow', action='store_true')
    parser.add_argument('--shadow_dir', default=None)
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--checkpoint_path', default=None)
    opts = parser.parse_args()

    np.random.seed(opts.seed)
    random.seed(opts.seed)
    torch.manual_seed(opts.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(opts.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.use_deterministic_algorithms = False
    if opts.amp:
        torch.autograd.set_detect_anomaly(False)
    else:
        torch.autograd.set_detect_anomaly(True)

    if not opts.use_prepare_shadow and opts.amp:
        print('amp is not supported when use_prepare_shadow is False because of the shadow mapping using nvdiffrast.')
        exit()

    TrainRefineShadow(opts).training()

if __name__ == '__main__':
    main()