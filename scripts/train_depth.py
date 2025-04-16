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
from losses.gradient_loss import Gradient_loss

class TrainDepth:
    def __init__(self, opts):
        self.opts = opts
        os.makedirs(self.opts.out_dir, exist_ok=True)

        self.net = UNet(self.opts,in_channels=4,out_channels=1).to(self.opts.device)
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
        self.train_dataset = ImagesDataset(opts=self.opts,train_or_val='train',id='depth')
        self.val_dataset = ImagesDataset(opts=self.opts,train_or_val='val',id='depth')
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

        self.list_loss_name = ['l1_loss_si','grad_loss']
        self.list_metric_name = ['rmse_w_mask', 'ssim_w_mask']#, 'lpips']
        utils.generate_log_txt(self.log_train_path,['epoch'] + self.list_loss_name)
        utils.generate_log_txt(self.log_val_path,['epoch'] + self.list_metric_name)

        #self.lpips = metrics.LPIPS(self.opts).lpips
        self.gradient_loss = Gradient_loss(self.opts).gradient_loss

        self.checkpoint_save_dir = '%s/checkpoints' % self.opts.out_dir
        os.makedirs(self.checkpoint_save_dir, exist_ok=True)

    def seed_worker(self, worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    def train_val(self, epoch, is_train):
        self.net.train() if is_train else self.net.eval()
        list_log = defaultdict(list)
        for itr, data_dict in enumerate(tqdm(self.train_dataloader if is_train else self.val_dataloader,desc='%d' % epoch,ncols=50)):
            with torch.set_grad_enabled(is_train):
                if self.opts.debug and itr >= 2: #TODO
                    break
                if is_train:
                    self.optimizer.zero_grad(set_to_none=True)

                human_id = data_dict['human_id']
                img_w_bg = data_dict['img_w_bg'].to(self.opts.device, non_blocking=True)
                mask = data_dict['mask'].to(self.opts.device, non_blocking=True)
                mask_bg = data_dict['mask_bg'].to(self.opts.device, non_blocking=True)
                depth_norm_gt = data_dict['depth_norm'].to(self.opts.device, non_blocking=True)

                if np.random.rand() > 0.5:
                    img_w_bg = torch.flip(img_w_bg, [3])
                    mask = torch.flip(mask, [3])
                    mask_bg = torch.flip(mask_bg, [3])
                    depth_norm_gt = torch.flip(depth_norm_gt, [3])
                depth_norm_gt = mask * depth_norm_gt
                img_w_bg = mask_bg * img_w_bg 

                net_input = torch.cat([img_w_bg, mask], dim=1)
                depth_norm_pred = self.net(net_input)
                depth_norm_pred = torch.sigmoid(depth_norm_pred)

                with torch.no_grad():
                    sum_depth_gt = torch.sum(mask * depth_norm_gt, dim=[1,2,3])
                    sum_depth_pred = torch.sum(mask * depth_norm_pred.detach(), dim=[1,2,3])
                    c_si = ((sum_depth_gt - sum_depth_pred) / torch.sum(mask, dim=[1,2,3]))[:,None,None,None]
                depth_norm_pred = mask * (c_si + depth_norm_pred)
                if is_train:
                    loss = 0
                    for loss_name in self.list_loss_name:
                        if loss_name == 'l1_loss_si':
                            _loss = F.l1_loss(depth_norm_gt, depth_norm_pred)
                            list_log[loss_name].append(_loss.item())
                            loss = loss + _loss
                        if loss_name == 'grad_loss':
                            _loss = self.gradient_loss(depth_norm_gt, depth_norm_pred)
                            list_log[loss_name].append(_loss.item())
                            loss = loss + 0.01 * _loss   
                    loss.backward()
                    self.optimizer.step()
                    
                else:
                    for metric_name in self.list_metric_name:
                        if metric_name == 'rmse_w_mask':
                            list_log[metric_name].append(metrics.rmse_w_mask(depth_norm_gt, depth_norm_pred, mask))
                        elif metric_name == 'ssim_w_mask':
                            list_log[metric_name].append(metrics.ssim_w_mask(depth_norm_gt, depth_norm_pred, mask))
                        elif metric_name == 'lpips':
                            list_log[metric_name].append(self.lpips(depth_norm_gt, depth_norm_pred))
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
                        np.concatenate([utils.torch2np(255 * utils.clip_img(img_w_bg[batch_save:batch_save+1],mask[batch_save:batch_save+1])[0][0,[2,1,0]].permute(1,2,0)),
                                        cv2.applyColorMap((255 * utils.torch2np(torch.cat([utils.clip_img(depth_norm_pred[batch_save:batch_save+1],mask[batch_save:batch_save+1])[0],
                                                                                            utils.clip_img(depth_norm_gt[batch_save:batch_save+1],mask[batch_save:batch_save+1])[0]],
                                                                                            dim=3)[0,0])).astype(np.uint8),
                                                            cv2.COLORMAP_INFERNO)], axis=1))
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
    parser.add_argument('--out_dir', default='./outputs/train_depth')
    parser.add_argument('--t_max', default=20, type=int)
    parser.add_argument('--batch_size', default=28, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--n_epoch', default=2000, type=int)
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
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--eps', default=1e-6, type=float)
    parser.add_argument('--use_other_dataset', action='store_true')
    parser.add_argument('--other_dataset_dir', default=None)
    parser.add_argument('--checkpoint_path', default=None)
    opts = parser.parse_args()

    np.random.seed(opts.seed)
    random.seed(opts.seed)
    torch.manual_seed(opts.seed)
    torch.cuda.manual_seed(opts.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.use_deterministic_algorithms = False
    torch.autograd.set_detect_anomaly(True)

    TrainDepth(opts).training()

if __name__ == '__main__':
    main()