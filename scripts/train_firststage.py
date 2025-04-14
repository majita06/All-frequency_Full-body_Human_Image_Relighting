import os
from argparse import ArgumentParser
from tqdm import tqdm
from torch.utils.data import DataLoader
from collections import defaultdict
import torch
import torch.nn.functional as F
import numpy as np
from datasets.images_dataset import ImagesDataset
from models.unet_multi import UNet_multi
from utils import utils, metrics
import cv2
import random
from shaders.lambert import Lambert
from shaders.disney import Disney


class TrainFirstStage:
    def __init__(self, opts):
        self.opts = opts
        os.makedirs(self.opts.out_dir, exist_ok=True)

        self.net = UNet_multi(self.opts,in_channels=4).to(self.opts.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.opts.lr_max)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.opts.t_max, eta_min=self.opts.lr_min)

        g = torch.Generator()
        g.manual_seed(self.opts.seed)
        self.train_dataset = ImagesDataset(opts=self.opts,train_or_val='train',id='firststage')
        self.val_dataset = ImagesDataset(opts=self.opts,train_or_val='val',id='firststage')
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

        self.list_loss_name = ['l1_loss_albedo','vgg_loss_albedo','l1_loss_normal','l1_loss_specular','l1_loss_roughness','l1_loss_als',
                               'l1_loss_source_diffuse_shading_wo_shadow','l1_loss_source_specular_shading_wo_shadow','vgg_loss_source_specular_shading_wo_shadow',
                               'l1_loss_source_rendering_wo_shadow_w_specular','vgg_loss_source_rendering_wo_shadow_w_specular',
                               'l1_loss_target_diffuse_shading_wo_shadow','l1_loss_target_specular_shading_wo_shadow','vgg_loss_target_specular_shading_wo_shadow',
                               'l1_loss_target_rendering_wo_shadow_w_specular','vgg_loss_target_rendering_wo_shadow_w_specular']
        # self.list_metric_name = ['rmse_w_mask_albedo', 'ssim_w_mask_albedo', 'lpips_albedo',
        #                          'rmse_w_mask_normal', 'ssim_w_mask_normal', 'lpips_normal',
        #                          'rmse_w_mask_specular', 'ssim_w_mask_specular', 'lpips_specular',
        #                          'rmse_w_mask_roughness', 'ssim_w_mask_roughness', 'lpips_roughness',
        #                          'rmse_w_mask_als', 'ssim_w_mask_als', 'lpips_als',
        #                          'rmse_w_mask_source_rendering_wo_shadow_w_specular', 'ssim_w_mask_source_rendering_wo_shadow_w_specular', 'lpips_source_rendering_wo_shadow_w_specular',
        #                          'rmse_w_mask_target_rendering_wo_shadow_w_specular', 'ssim_w_mask_target_rendering_wo_shadow_w_specular', 'lpips_target_rendering_wo_shadow_w_specular']
        self.list_metric_name = ['rmse_w_mask_albedo', 'ssim_w_mask_albedo',
                                 'rmse_w_mask_normal', 'ssim_w_mask_normal',
                                 'rmse_w_mask_specular', 'ssim_w_mask_specular',
                                 'rmse_w_mask_roughness', 'ssim_w_mask_roughness',
                                 'rmse_w_mask_als', 'ssim_w_mask_als',
                                 'rmse_w_mask_source_rendering_wo_shadow_w_specular', 'ssim_w_mask_source_rendering_wo_shadow_w_specular',
                                 'rmse_w_mask_target_rendering_wo_shadow_w_specular', 'ssim_w_mask_target_rendering_wo_shadow_w_specular']
        utils.generate_log_txt(self.log_train_path,['epoch'] + self.list_loss_name)
        utils.generate_log_txt(self.log_val_path,['epoch'] + self.list_metric_name)

        #self.lpips = metrics.LPIPS(self.opts).lpips
        from losses.vgg_loss import VGGLoss
        self.vgg_loss = VGGLoss(self.opts)

        self.lambert = Lambert().lambert
        self.disney = Disney(self.opts).disney
        if self.opts.train_sigma:
            from shaders.shadow_mapping import ShadowMapping
            self.shadow_mapping = ShadowMapping(self.opts,resolution=self.opts.resolution)

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
                    idx_train_specular = np.array([idx for idx, h_id in enumerate(human_id) if not 'mesh' in h_id])

                    mask = data_dict['mask'].to(self.opts.device, non_blocking=True)
                    mask_bg = data_dict['mask_bg'].to(self.opts.device, non_blocking=True)
                    mask_eye_and_shoe = data_dict['mask_eye_and_shoe'].to(self.opts.device, non_blocking=True)
                    img_w_bg = data_dict['img_w_bg'].to(self.opts.device, non_blocking=True)

                    albedo_gt = data_dict['albedo'].to(self.opts.device, non_blocking=True)
                    normal_gt = data_dict['normal'].to(self.opts.device, non_blocking=True)
                    specular_gt = data_dict['specular'].to(self.opts.device, non_blocking=True)
                    roughness_gt = data_dict['roughness'].to(self.opts.device, non_blocking=True)

                    source_als = data_dict['source_als'].to(self.opts.device, non_blocking=True)
                    source_diffuse_shading_wo_shadow_gt = data_dict['source_diffuse_shading_wo_shadow'].to(self.opts.device, non_blocking=True)
                    source_specular_shading_wo_shadow_gt = data_dict['source_specular_shading_wo_shadow'].to(self.opts.device, non_blocking=True)
                    source_rendering_wo_shadow_w_specular_gt = data_dict['source_rendering_wo_shadow_w_specular'].to(self.opts.device, non_blocking=True)

                    target_als = data_dict['target_als'].to(self.opts.device, non_blocking=True)
                    target_diffuse_shading_wo_shadow_gt = data_dict['target_diffuse_shading_wo_shadow'].to(self.opts.device, non_blocking=True)
                    target_specular_shading_wo_shadow_gt = data_dict['target_specular_shading_wo_shadow'].to(self.opts.device, non_blocking=True)
                    target_rendering_wo_shadow_w_specular_gt = data_dict['target_rendering_wo_shadow_w_specular'].to(self.opts.device, non_blocking=True)

                    if self.opts.train_sigma:
                        source_diffuse_shading_w_shadow_gt = data_dict['source_diffuse_shading_w_shadow'].to(self.opts.device, non_blocking=True)

                    if np.random.rand() > 0.5:
                        mask = torch.flip(mask, [3])
                        mask_bg = torch.flip(mask_bg, [3])
                        mask_eye_and_shoe = torch.flip(mask_eye_and_shoe, [3])
                        img_w_bg = torch.flip(img_w_bg, [3])

                        albedo_gt = torch.flip(albedo_gt, [3])
                        normal_gt = torch.flip(normal_gt, [3])
                        normal_gt[:,0] *= -1
                        specular_gt = torch.flip(specular_gt, [3])
                        roughness_gt = torch.flip(roughness_gt, [3])

                        source_als[:,:,0] *= -1
                        source_diffuse_shading_wo_shadow_gt = torch.flip(source_diffuse_shading_wo_shadow_gt, [3])
                        source_specular_shading_wo_shadow_gt = torch.flip(source_specular_shading_wo_shadow_gt, [3])
                        source_rendering_wo_shadow_w_specular_gt = torch.flip(source_rendering_wo_shadow_w_specular_gt, [3])

                        target_als[:,:,0] *= -1
                        target_diffuse_shading_wo_shadow_gt = torch.flip(target_diffuse_shading_wo_shadow_gt, [3])
                        target_specular_shading_wo_shadow_gt = torch.flip(target_specular_shading_wo_shadow_gt, [3])
                        target_rendering_wo_shadow_w_specular_gt = torch.flip(target_rendering_wo_shadow_w_specular_gt, [3])

                        if self.opts.train_sigma:
                            source_diffuse_shading_w_shadow_gt = torch.flip(source_diffuse_shading_w_shadow_gt, [3])

                    img_w_bg = mask_bg * img_w_bg

                    albedo_gt = mask * albedo_gt
                    normal_gt = mask * normal_gt
                    specular_gt = mask * specular_gt
                    roughness_gt = mask * roughness_gt

                    source_diffuse_shading_wo_shadow_gt = mask * source_diffuse_shading_wo_shadow_gt
                    source_specular_shading_wo_shadow_gt = mask * source_specular_shading_wo_shadow_gt
                    source_rendering_wo_shadow_w_specular_gt = mask * source_rendering_wo_shadow_w_specular_gt

                    target_diffuse_shading_wo_shadow_gt = mask * target_diffuse_shading_wo_shadow_gt
                    target_specular_shading_wo_shadow_gt = mask * target_specular_shading_wo_shadow_gt
                    target_rendering_wo_shadow_w_specular_gt = mask * target_rendering_wo_shadow_w_specular_gt


                    net_input = torch.cat([img_w_bg,mask],dim=1)
                    albedo_pred, normal_pred, specular_pred, roughness_pred, source_als_pred = self.net(net_input)
                    albedo_pred = mask * albedo_pred
                    normal_pred = mask * normal_pred
                    specular_pred = mask * specular_pred
                    roughness_pred = mask * roughness_pred

                    if self.opts.train_sigma:
                        _source_als_pred = torch.cat([source_als_pred[:,:,0:3].detach(),
                                                    source_als_pred[:,:,3:4],
                                                    source_als_pred[:,:,4:7].detach()],dim=2)
                        mask_vis = torch.einsum('bxhw,bdx->bdhw',normal_gt,_source_als_pred[:,:,0:3]).clamp(0,1)
                        mask_vis[mask_vis>0] = 1
                        source_soft_shadows_pred = self.shadow_mapping(depth_gt, _source_als_pred, mask, train=True)[1]
                        source_soft_shadows_pred = mask_vis * source_soft_shadows_pred
                        source_diffuse_shading_w_shadow_pred = torch.einsum('bdhw,bdchw->bchw',
                                                                            source_soft_shadows_pred,
                                                                            self.lambert(normal_gt, _source_als_pred, per_light=True))
                        source_diffuse_shading_w_shadow_pred = mask * source_diffuse_shading_w_shadow_pred

                    source_diffuse_shading_wo_shadow_predals = self.lambert(normal_gt, source_als_pred)
                    source_diffuse_shading_wo_shadow_predals = mask * source_diffuse_shading_wo_shadow_predals
                    source_specular_shading_wo_shadow_predals = self.disney(normal_gt, source_als_pred, specular_gt, roughness_gt)
                    source_specular_shading_wo_shadow_predals = mask * source_specular_shading_wo_shadow_predals

                    source_diffuse_shading_wo_shadow_pred = self.lambert(normal_pred, source_als)
                    source_diffuse_shading_wo_shadow_pred = mask * source_diffuse_shading_wo_shadow_pred
                    source_specular_shading_wo_shadow_pred = self.disney(normal_pred, source_als, specular_pred, roughness_pred)
                    source_specular_shading_wo_shadow_pred = mask * source_specular_shading_wo_shadow_pred

                    target_diffuse_shading_wo_shadow_pred = self.lambert(normal_pred, target_als)
                    target_diffuse_shading_wo_shadow_pred = mask * target_diffuse_shading_wo_shadow_pred
                    target_specular_shading_wo_shadow_pred = self.disney(normal_pred, target_als, specular_pred, roughness_pred)
                    target_specular_shading_wo_shadow_pred = mask * target_specular_shading_wo_shadow_pred

                    source_rendering_wo_shadow_w_specular_pred = albedo_pred * source_diffuse_shading_wo_shadow_pred + source_specular_shading_wo_shadow_pred
                    target_rendering_wo_shadow_w_specular_pred = albedo_pred * target_diffuse_shading_wo_shadow_pred + target_specular_shading_wo_shadow_pred

                    if is_train:
                        loss = 0
                        for loss_name in self.list_loss_name:
                            if loss_name == 'l1_loss_albedo':
                                _loss = F.l1_loss(utils.lrgb2srgb(albedo_gt.clamp(self.opts.eps,None)), utils.lrgb2srgb(albedo_pred.clamp(self.opts.eps,None)))
                                list_log[loss_name].append(_loss.item())
                                loss = loss + _loss
                            elif loss_name == 'vgg_loss_albedo':
                                _loss = self.vgg_loss(utils.lrgb2srgb(albedo_gt.clamp(self.opts.eps,None)), utils.lrgb2srgb(albedo_pred.clamp(self.opts.eps,None)))
                                list_log[loss_name].append(_loss.item())
                                loss = loss + _loss
                            elif loss_name == 'l1_loss_normal':
                                _loss = F.l1_loss(normal_gt, normal_pred)
                                list_log[loss_name].append(_loss.item())
                                loss = loss + _loss
                            elif loss_name == 'l1_loss_specular':
                                if len(idx_train_specular) != 0:
                                    _loss = F.l1_loss((mask_eye_and_shoe * specular_gt)[idx_train_specular], (mask_eye_and_shoe * specular_pred)[idx_train_specular])
                                    list_log[loss_name].append(_loss.item())
                                    loss = loss + _loss
                            elif loss_name == 'l1_loss_roughness':
                                if len(idx_train_specular) != 0:
                                    _loss = F.l1_loss((mask_eye_and_shoe * roughness_gt)[idx_train_specular], (mask_eye_and_shoe * roughness_pred)[idx_train_specular])
                                    list_log[loss_name].append(_loss.item())
                                    loss = loss + _loss
                            elif loss_name == 'l1_loss_als':
                                _loss = F.l1_loss(utils.lrgb2srgb(source_diffuse_shading_wo_shadow_gt.clamp(self.opts.eps,None)),
                                                  utils.lrgb2srgb(source_diffuse_shading_wo_shadow_predals.clamp(self.opts.eps,None)))
                                if len(idx_train_specular) != 0:
                                    _loss = _loss + F.l1_loss(utils.lrgb2srgb(source_specular_shading_wo_shadow_gt.clamp(self.opts.eps,None))[idx_train_specular],
                                                              utils.lrgb2srgb(source_specular_shading_wo_shadow_predals.clamp(self.opts.eps,None))[idx_train_specular])
                                list_log[loss_name].append(_loss.item())
                                loss = loss + _loss
                            elif loss_name == 'l1_loss_sigma':
                                _loss = F.l1_loss(utils.lrgb2srgb(source_diffuse_shading_w_shadow_gt.clamp(self.opts.eps,None)),
                                                  utils.lrgb2srgb(source_diffuse_shading_w_shadow_pred.clamp(self.opts.eps,None)))
                                list_log[loss_name].append(_loss.item())
                                loss = loss + _loss      

                            elif loss_name == 'l1_loss_source_diffuse_shading_wo_shadow':
                                _loss = F.l1_loss(utils.lrgb2srgb(source_diffuse_shading_wo_shadow_gt.clamp(self.opts.eps,None)),
                                                  utils.lrgb2srgb(source_diffuse_shading_wo_shadow_pred.clamp(self.opts.eps,None)))
                                list_log[loss_name].append(_loss.item())
                                loss = loss + _loss
                            elif loss_name == 'l1_loss_source_specular_shading_wo_shadow':
                                if len(idx_train_specular) != 0:
                                    _loss = F.l1_loss(utils.lrgb2srgb(source_specular_shading_wo_shadow_gt.clamp(self.opts.eps,None))[idx_train_specular],
                                                      utils.lrgb2srgb(source_specular_shading_wo_shadow_pred.clamp(self.opts.eps,None))[idx_train_specular])
                                    list_log[loss_name].append(_loss.item())
                                    loss = loss + _loss
                            elif loss_name == 'vgg_loss_source_specular_shading_wo_shadow':
                                if len(idx_train_specular) != 0:
                                    _loss = self.vgg_loss(utils.lrgb2srgb(source_specular_shading_wo_shadow_gt.clamp(self.opts.eps,1))[idx_train_specular],
                                                          utils.lrgb2srgb(source_specular_shading_wo_shadow_pred.clamp(self.opts.eps,1))[idx_train_specular])
                                    list_log[loss_name].append(_loss.item())
                                    loss = loss + _loss
                            elif loss_name == 'l1_loss_source_rendering_wo_shadow_w_specular':
                                if len(idx_train_specular) != 0:
                                    _loss = F.l1_loss(utils.lrgb2srgb(source_rendering_wo_shadow_w_specular_gt.clamp(self.opts.eps,None))[idx_train_specular],
                                                      utils.lrgb2srgb(source_rendering_wo_shadow_w_specular_pred.clamp(self.opts.eps,None))[idx_train_specular])
                                    list_log[loss_name].append(_loss.item())
                                    loss = loss + _loss
                            elif loss_name == 'vgg_loss_source_rendering_wo_shadow_w_specular':
                                if len(idx_train_specular) != 0:
                                    _loss = self.vgg_loss(utils.lrgb2srgb(source_rendering_wo_shadow_w_specular_gt.clamp(self.opts.eps,1))[idx_train_specular],
                                                          utils.lrgb2srgb(source_rendering_wo_shadow_w_specular_pred.clamp(self.opts.eps,1))[idx_train_specular])
                                    list_log[loss_name].append(_loss.item())
                                    loss = loss + _loss

                            elif loss_name == 'l1_loss_target_diffuse_shading_wo_shadow':
                                _loss = F.l1_loss(utils.lrgb2srgb(target_diffuse_shading_wo_shadow_gt.clamp(self.opts.eps,None)),
                                                  utils.lrgb2srgb(target_diffuse_shading_wo_shadow_pred.clamp(self.opts.eps,None)))
                                list_log[loss_name].append(_loss.item())
                                loss = loss + _loss
                            elif loss_name == 'l1_loss_target_specular_shading_wo_shadow':
                                if len(idx_train_specular) != 0:
                                    _loss = F.l1_loss(utils.lrgb2srgb(target_specular_shading_wo_shadow_gt.clamp(self.opts.eps,None))[idx_train_specular],
                                                      utils.lrgb2srgb(target_specular_shading_wo_shadow_pred.clamp(self.opts.eps,None))[idx_train_specular])
                                    list_log[loss_name].append(_loss.item())
                                    loss = loss + _loss
                            elif loss_name == 'vgg_loss_target_specular_shading_wo_shadow':
                                if len(idx_train_specular) != 0:
                                    _loss = self.vgg_loss(utils.lrgb2srgb(target_specular_shading_wo_shadow_gt.clamp(self.opts.eps,1))[idx_train_specular],
                                                          utils.lrgb2srgb(target_specular_shading_wo_shadow_pred.clamp(self.opts.eps,1))[idx_train_specular])
                                    list_log[loss_name].append(_loss.item())
                                    loss = loss + _loss
                            elif loss_name == 'l1_loss_target_rendering_wo_shadow_w_specular':
                                if len(idx_train_specular) != 0:
                                    _loss = F.l1_loss(utils.lrgb2srgb(target_rendering_wo_shadow_w_specular_gt.clamp(self.opts.eps,None))[idx_train_specular],
                                                      utils.lrgb2srgb(target_rendering_wo_shadow_w_specular_pred.clamp(self.opts.eps,None))[idx_train_specular])
                                    list_log[loss_name].append(_loss.item())
                                    loss = loss + _loss
                            elif loss_name == 'vgg_loss_target_rendering_wo_shadow_w_specular':
                                if len(idx_train_specular) != 0:
                                    _loss = self.vgg_loss(utils.lrgb2srgb(target_rendering_wo_shadow_w_specular_gt.clamp(self.opts.eps,1))[idx_train_specular],
                                                          utils.lrgb2srgb(target_rendering_wo_shadow_w_specular_pred.clamp(self.opts.eps,1))[idx_train_specular])
                                    list_log[loss_name].append(_loss.item())
                                    loss = loss + _loss
                        
                        self.scaler.scale(loss).backward()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        
                    else:
                        for metric_name in self.list_metric_name:
                            if metric_name == 'rmse_w_mask_albedo':
                                list_log[metric_name].append(metrics.rmse_w_mask(utils.lrgb2srgb(albedo_gt), utils.lrgb2srgb(albedo_pred), mask))
                            elif metric_name == 'ssim_w_mask_albedo':
                                list_log[metric_name].append(metrics.ssim_w_mask(utils.lrgb2srgb(albedo_gt), utils.lrgb2srgb(albedo_pred), mask))
                            elif metric_name == 'lpips_albedo':
                                list_log[metric_name].append(self.lpips(utils.lrgb2srgb(albedo_gt), utils.lrgb2srgb(albedo_pred)))
                            
                            elif metric_name == 'rmse_w_mask_normal':
                                list_log[metric_name].append(metrics.rmse_w_mask(normal_gt, normal_pred, mask))
                            elif metric_name == 'ssim_w_mask_normal':
                                list_log[metric_name].append(metrics.ssim_w_mask(normal_gt, normal_pred, mask))
                            elif metric_name == 'lpips_normal':
                                list_log[metric_name].append(self.lpips(normal_gt, normal_pred))

                            elif metric_name == 'rmse_w_mask_specular':
                                if len(idx_train_specular) != 0:
                                    list_log[metric_name].append(metrics.rmse_w_mask(specular_gt[idx_train_specular],
                                                                                     specular_pred[idx_train_specular],
                                                                                     mask[idx_train_specular]))
                            elif metric_name == 'ssim_w_mask_specular':
                                if len(idx_train_specular) != 0:
                                    list_log[metric_name].append(metrics.ssim_w_mask(specular_gt[idx_train_specular],
                                                                                     specular_pred[idx_train_specular],
                                                                                     mask[idx_train_specular]))
                            elif metric_name == 'lpips_specular':
                                if len(idx_train_specular) != 0:
                                    list_log[metric_name].append(self.lpips(specular_gt[idx_train_specular], specular_pred[idx_train_specular]))

                            elif metric_name == 'rmse_w_mask_roughness':
                                if len(idx_train_specular) != 0:
                                    list_log[metric_name].append(metrics.rmse_w_mask(roughness_gt[idx_train_specular],
                                                                                     roughness_pred[idx_train_specular],
                                                                                     mask[idx_train_specular]))
                            elif metric_name == 'ssim_w_mask_roughness':
                                if len(idx_train_specular) != 0:
                                    list_log[metric_name].append(metrics.ssim_w_mask(roughness_gt[idx_train_specular],
                                                                                     roughness_pred[idx_train_specular],
                                                                                     mask[idx_train_specular]))
                            elif metric_name == 'lpips_roughness':
                                if len(idx_train_specular) != 0:
                                    list_log[metric_name].append(self.lpips(roughness_gt[idx_train_specular], roughness_pred[idx_train_specular]))

                            elif metric_name == 'rmse_w_mask_als':
                                rmse_w_mask_als = metrics.rmse_w_mask(utils.lrgb2srgb(source_diffuse_shading_wo_shadow_gt),
                                                                      utils.lrgb2srgb(source_diffuse_shading_wo_shadow_predals),
                                                                      mask)
                                if len(idx_train_specular) != 0:
                                    rmse_w_mask_als = 0.5 * (rmse_w_mask_als + metrics.rmse_w_mask(utils.lrgb2srgb(source_specular_shading_wo_shadow_gt)[idx_train_specular],
                                                                                                   utils.lrgb2srgb(source_specular_shading_wo_shadow_predals)[idx_train_specular],
                                                                                                   mask[idx_train_specular]))
                                list_log[metric_name].append(rmse_w_mask_als)
                            elif metric_name == 'ssim_w_mask_als':
                                ssim_w_mask_als = metrics.ssim_w_mask(utils.lrgb2srgb(source_diffuse_shading_wo_shadow_gt),
                                                                      utils.lrgb2srgb(source_diffuse_shading_wo_shadow_predals),
                                                                      mask)
                                if len(idx_train_specular) != 0:
                                    ssim_w_mask_als = 0.5 * (ssim_w_mask_als + metrics.ssim_w_mask(utils.lrgb2srgb(source_specular_shading_wo_shadow_gt)[idx_train_specular],
                                                                                                   utils.lrgb2srgb(source_specular_shading_wo_shadow_predals)[idx_train_specular],
                                                                                                   mask[idx_train_specular]))
                                list_log[metric_name].append(ssim_w_mask_als)
                            elif metric_name == 'lpips_als':
                                lpips_als = self.lpips(utils.lrgb2srgb(source_diffuse_shading_wo_shadow_gt), utils.lrgb2srgb(source_diffuse_shading_wo_shadow_predals))
                                if len(idx_train_specular) != 0:
                                    lpips_als = 0.5 * (lpips_als + self.lpips(utils.lrgb2srgb(source_specular_shading_wo_shadow_gt)[idx_train_specular],
                                                                              utils.lrgb2srgb(source_specular_shading_wo_shadow_predals)[idx_train_specular]))
                                list_log[metric_name].append(lpips_als)

                            elif metric_name == 'rmse_w_mask_source_rendering_wo_shadow_w_specular':
                                if len(idx_train_specular) != 0:
                                    list_log[metric_name].append(metrics.rmse_w_mask(utils.lrgb2srgb(source_rendering_wo_shadow_w_specular_gt)[idx_train_specular],
                                                                                     utils.lrgb2srgb(source_rendering_wo_shadow_w_specular_pred)[idx_train_specular],
                                                                                     mask[idx_train_specular]))
                            elif metric_name == 'ssim_w_mask_source_rendering_wo_shadow_w_specular':
                                if len(idx_train_specular) != 0:
                                    list_log[metric_name].append(metrics.ssim_w_mask(utils.lrgb2srgb(source_rendering_wo_shadow_w_specular_gt)[idx_train_specular],
                                                                                     utils.lrgb2srgb(source_rendering_wo_shadow_w_specular_pred)[idx_train_specular],
                                                                                     mask[idx_train_specular]))
                            elif metric_name == 'lpips_source_rendering_wo_shadow_w_specular':
                                if len(idx_train_specular) != 0:
                                    list_log[metric_name].append(self.lpips(utils.lrgb2srgb(source_rendering_wo_shadow_w_specular_gt)[idx_train_specular],
                                                                            utils.lrgb2srgb(source_rendering_wo_shadow_w_specular_pred)[idx_train_specular]))
                            
                            elif metric_name == 'rmse_w_mask_target_rendering_wo_shadow_w_specular':
                                if len(idx_train_specular) != 0:
                                    list_log[metric_name].append(metrics.rmse_w_mask(utils.lrgb2srgb(target_rendering_wo_shadow_w_specular_gt)[idx_train_specular],
                                                                                     utils.lrgb2srgb(target_rendering_wo_shadow_w_specular_pred)[idx_train_specular],
                                                                                     mask[idx_train_specular]))
                            elif metric_name == 'ssim_w_mask_target_rendering_wo_shadow_w_specular':
                                if len(idx_train_specular) != 0:
                                    list_log[metric_name].append(metrics.ssim_w_mask(utils.lrgb2srgb(target_rendering_wo_shadow_w_specular_gt)[idx_train_specular],
                                                                                     utils.lrgb2srgb(target_rendering_wo_shadow_w_specular_pred)[idx_train_specular],
                                                                                     mask[idx_train_specular]))
                            elif metric_name == 'lpips_target_rendering_wo_shadow_w_specular':
                                if len(idx_train_specular) != 0:
                                    list_log[metric_name].append(self.lpips(utils.lrgb2srgb(target_rendering_wo_shadow_w_specular_gt)[idx_train_specular],
                                                                            utils.lrgb2srgb(target_rendering_wo_shadow_w_specular_pred)[idx_train_specular]))
                
        
        if is_train:
            self.scheduler.step()
  
        with open(self.log_train_path if is_train else self.log_val_path,"a") as f:
            f.write('%d' % epoch)
            for key in self.list_loss_name if is_train else self.list_metric_name:
                f.write(',%.7f' % np.mean(np.array(list_log[key])))
            f.write('\n')
        if ((epoch + 1) / self.opts.t_max) % 2 == 1:
            batch_save = 0
            save_pred = torch.cat([utils.clip_img(utils.lrgb2srgb(albedo_pred)[batch_save:batch_save+1],mask[batch_save:batch_save+1])[0],
                                utils.clip_img(0.5*(normal_pred+1)[batch_save:batch_save+1],mask[batch_save:batch_save+1])[0],
                                utils.clip_img(specular_pred.expand(-1,3,-1,-1)[batch_save:batch_save+1],mask[batch_save:batch_save+1])[0],
                                utils.clip_img(roughness_pred.expand(-1,3,-1,-1)[batch_save:batch_save+1],mask[batch_save:batch_save+1])[0],
                                utils.clip_img(utils.lrgb2srgb(source_diffuse_shading_wo_shadow_predals)[batch_save:batch_save+1],mask[batch_save:batch_save+1])[0],
                                utils.clip_img(utils.lrgb2srgb(source_rendering_wo_shadow_w_specular_pred)[batch_save:batch_save+1],mask[batch_save:batch_save+1])[0],
                                utils.clip_img(utils.lrgb2srgb(target_rendering_wo_shadow_w_specular_pred)[batch_save:batch_save+1],mask[batch_save:batch_save+1])[0]],
                                dim=3)
            save_gt = torch.cat([utils.clip_img(utils.lrgb2srgb(albedo_gt)[batch_save:batch_save+1],mask[batch_save:batch_save+1])[0],
                                utils.clip_img(0.5*(normal_gt+1)[batch_save:batch_save+1],mask[batch_save:batch_save+1])[0],
                                utils.clip_img(specular_gt.expand(-1,3,-1,-1)[batch_save:batch_save+1],mask[batch_save:batch_save+1])[0],
                                utils.clip_img(roughness_gt.expand(-1,3,-1,-1)[batch_save:batch_save+1],mask[batch_save:batch_save+1])[0],
                                utils.clip_img(utils.lrgb2srgb(source_diffuse_shading_wo_shadow_gt)[batch_save:batch_save+1],mask[batch_save:batch_save+1])[0],
                                utils.clip_img(utils.lrgb2srgb(source_rendering_wo_shadow_w_specular_gt)[batch_save:batch_save+1],mask[batch_save:batch_save+1])[0],
                                utils.clip_img(utils.lrgb2srgb(target_rendering_wo_shadow_w_specular_gt)[batch_save:batch_save+1],mask[batch_save:batch_save+1])[0]],
                                dim=3)

            cv2.imwrite('%s/%04depoch_%s_%s.png' % (self.opts.out_dir,epoch,'train' if is_train else 'val',human_id[batch_save]),
                        utils.torch2np(255 * torch.cat([save_pred,save_gt],dim=2)[0,[2,1,0]].permute(1,2,0)))
            checkpoint = {'model_state_dict': self.net.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'scheduler_state_dict': self.scheduler.state_dict()}
            torch.save(checkpoint, '%s/%04depoch.pth' % (self.checkpoint_save_dir,epoch))

    def training(self):
        for epoch in range(self.opts.n_epoch):
            self.train_val(epoch,is_train=True)
            if ((epoch + 1) / self.opts.t_max) % 2 == 1:
                self.train_val(epoch,is_train=False)
    
def main():
    parser = ArgumentParser()
    parser.add_argument('--device', default='cuda') # 'cuda' or 'cpu'
    parser.add_argument('--out_dir', default='/home/tajima/All-frequency_Full-body_Human_Image_Relighting/outputs/train_firststage')
    parser.add_argument('--t_max', default=20, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--n_epoch', default=2000, type=int)
    parser.add_argument('--dataset_dir', default='/home/tajima/dataset/humgen_dataset')
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
    parser.add_argument('--als_dir', default='/home/tajima/dataset/EG25/als')
    parser.add_argument('--n_light', default=16, type=int)
    parser.add_argument('--window', default=2, type=float)
    parser.add_argument('--depth_threshold', default=100000, type=float)
    parser.add_argument('--shadow_threshold', default=0.005, type=float)
    parser.add_argument('--eps', default=1e-6, type=float)
    parser.add_argument('--range_sigma', default=20, type=float)
    parser.add_argument('--use_other_dataset', action='store_true')
    parser.add_argument('--other_dataset_dir', default='/home/tajima/dataset/mesh_dataset')
    parser.add_argument('--train_sigma', action='store_true')
    parser.add_argument('--amp', action='store_true')

    
    opts = parser.parse_args()

    np.random.seed(opts.seed)
    random.seed(opts.seed)
    torch.manual_seed(opts.seed)
    torch.cuda.manual_seed(opts.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.use_deterministic_algorithms = False
    if opts.amp:
        torch.autograd.set_detect_anomaly(False)
    else:
        torch.autograd.set_detect_anomaly(True)
    TrainFirstStage(opts).training()

if __name__ == '__main__':
    main()