from torch.utils.data import Dataset
from utils import utils
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="True"
import numpy as np
import cv2
import random
import torch
from glob import glob

class ImagesDataset(Dataset):
    def __init__(self, opts, train_or_val, id):
        self.opts = opts
        self.train_or_val = train_or_val
        self.id = id

        self.human_dirs_1 = sorted(glob('%s/*' % opts.dataset_dir))
        n_human_1 = len(self.human_dirs_1)
        n_human_all = n_human_1
        if self.opts.use_other_dataset:
            self.human_dirs_2 = sorted(glob('%s/*' % opts.other_dataset_dir))
            n_human_2 = len(self.human_dirs_2)
            n_human_all += n_human_2

        if n_human_all < self.opts.n_train_samples:
            print('n_train_samples is less than all human data.')
            print('n_train_samples: %d, n_human_all: %d' % (self.opts.n_train_samples, n_human_all))
            exit()
        
        if not self.opts.use_other_dataset:
            self.human_dirs = self.human_dirs_1[:self.opts.n_train_samples] if self.train_or_val == 'train' else self.human_dirs_1[self.opts.n_train_samples:]
        else:
            self.human_dirs = (self.human_dirs_1[:int((n_human_1/n_human_all)*self.opts.n_train_samples)]+self.human_dirs_2[:int((n_human_2/n_human_all)*self.opts.n_train_samples)] if self.train_or_val == 'train' else
                               self.human_dirs_1[int((n_human_1/n_human_all)*self.opts.n_train_samples):]+self.human_dirs_2[int((n_human_2/n_human_all)*self.opts.n_train_samples):])
        
        self.dummy = np.ones((1,self.opts.resolution, self.opts.resolution), dtype=np.float32)
        self.dummy3 = np.ones((3,self.opts.resolution, self.opts.resolution), dtype=np.float32)

    def __len__(self):
        return len(self.human_dirs)

    def __getitem__(self, index):
        data_dict = {}
        human_dir = self.human_dirs[index]
        humanenv_dirs = glob('%s/*/' % human_dir)
        data_dict['human_id'] = os.path.basename(human_dir)
        data_dict['mask'] = cv2.imread('%s/mask.png' % human_dir, cv2.IMREAD_GRAYSCALE)[None].astype(np.float32)/255.

        if self.id == 'firststage' or self.id == 'depth':
            data_dict['mask_bg'] = self.get_random_mask_bg(data_dict['mask'])

        if self.id == 'firststage':
            data_dict['albedo'] = cv2.imread('%s/albedo.png' % human_dir, cv2.IMREAD_COLOR).transpose(2,0,1)[[2,1,0]].astype(np.float32)/255.
            if 'mesh' in data_dict['human_id']:
                data_dict['specular'] = self.dummy
                data_dict['roughness'] = self.dummy
                data_dict['mask_eye_and_shoe'] = self.dummy
            else:
                data_dict['specular'] = cv2.imread('%s/specular.png' % human_dir, cv2.IMREAD_GRAYSCALE)[None].astype(np.float32)/255.
                data_dict['roughness'] = cv2.imread('%s/roughness.png' % human_dir, cv2.IMREAD_GRAYSCALE)[None].astype(np.float32)/255.
                if os.path.exists('%s/label.png' % human_dir):
                    data_dict['mask_eye_and_shoe'] = cv2.imread('%s/label.png' % human_dir, cv2.IMREAD_COLOR).transpose(2,0,1)[[2,1,0]].astype(np.float32)/255.
                    data_dict['mask_eye_and_shoe'] = np.where(((data_dict['mask_eye_and_shoe'][0] == 0) & (data_dict['mask_eye_and_shoe'][1] == 1) & (data_dict['mask_eye_and_shoe'][2] == 0) | 
                                                ((data_dict['mask_eye_and_shoe'][0] == 0) & (data_dict['mask_eye_and_shoe'][1] == 1) & (data_dict['mask_eye_and_shoe'][2] == 1))),
                                                0, 1)[None].astype(np.float32)
                else:
                    data_dict['mask_eye_and_shoe'] = self.dummy
        if self.id == 'firststage' or self.id == 'refineshadow':
            data_dict['normal'] = cv2.imread('%s/normal.png' % human_dir, cv2.IMREAD_COLOR).transpose(2,0,1)[[2,1,0]].astype(np.float32)/255.
            data_dict['normal'][0] = 2.*data_dict['normal'][0]-1.
            data_dict['normal'][1] = -(2.*data_dict['normal'][1]-1.)
            data_dict['normal'] = data_dict['normal'] / np.linalg.norm(data_dict['normal'], axis=0, keepdims=True).clip(self.opts.eps, None)
        
        if self.id == 'depth' or self.id == 'refineshadow':
            data_dict['depth'] = cv2.imread('%s/depth.exr' % human_dir, -1)[None,...,0].astype(np.float32)            
            if self.id == 'depth':
                data_dict['depth_norm'] = data_dict['depth'] + (self.opts.camera_distance - np.median(data_dict['depth'][data_dict['mask'] == 1]))
                data_dict['depth_norm'] = (data_dict['depth_norm'] - self.opts.d_all_min) / (self.opts.d_all_max - self.opts.d_all_min)
            

        source_humanenv_dir, target_humanenv_dir = random.sample(humanenv_dirs, 2)
        source_humanenv_dir = source_humanenv_dir[:-len('/')]
        target_humanenv_dir = target_humanenv_dir[:-len('/')]

        if self.id == 'firststage' or self.id == 'depth':    
            data_dict['img_w_bg'] = cv2.imread('%s/rendering_w_shadow_w_specular.png' % source_humanenv_dir, cv2.IMREAD_COLOR).transpose(2,0,1)[[2,1,0]].astype(np.float32)/255.

        if self.id == 'firststage' or self.id == 'refineshadow':
            source_env_name_theta = os.path.basename(source_humanenv_dir)
            source_env_name, source_theta = source_env_name_theta.split('__')
            source_theta = float(source_theta)
            data_dict['source_env_name'] = source_env_name
            data_dict['source_theta'] = source_theta
            data_dict['source_als'] = np.load('%s/%s/als.npy' % (self.opts.als_dir, source_env_name))
            data_dict['source_als'] = self.rotation_als(data_dict['source_als'], source_theta)
            
            if self.id == 'firststage':
                target_env_name_theta = os.path.basename(target_humanenv_dir)
                target_env_name, target_theta = target_env_name_theta.split('__')
                target_theta = float(target_theta)
                data_dict['target_env_name'] = target_env_name
                data_dict['target_theta'] = target_theta
                data_dict['target_als'] = np.load('%s/%s/als.npy' % (self.opts.als_dir, target_env_name))
                data_dict['target_als'] = self.rotation_als(data_dict['target_als'], target_theta)

                data_dict['source_diffuse_shading_wo_shadow'] = cv2.imread('%s/diffuse_shading_wo_shadow.exr' % source_humanenv_dir, -1).transpose(2,0,1)[[2,1,0]].astype(np.float32)
                if 'mesh' in data_dict['human_id']:
                    data_dict['source_specular_shading_wo_shadow'] = self.dummy3
                    data_dict['source_rendering_wo_shadow_w_specular'] = self.dummy3
                else:
                    data_dict['source_specular_shading_wo_shadow'] = cv2.imread('%s/specular_shading_wo_shadow.exr' % source_humanenv_dir, -1).transpose(2,0,1)[[2,1,0]].astype(np.float32)
                    data_dict['source_rendering_wo_shadow_w_specular'] = cv2.imread('%s/rendering_wo_shadow_w_specular.exr' % source_humanenv_dir, -1).transpose(2,0,1)[[2,1,0]].astype(np.float32)

                data_dict['target_diffuse_shading_wo_shadow'] = cv2.imread('%s/diffuse_shading_wo_shadow.exr' % target_humanenv_dir, -1).transpose(2,0,1)[[2,1,0]].astype(np.float32)
                if 'mesh' in data_dict['human_id']:
                    data_dict['target_specular_shading_wo_shadow'] = self.dummy3
                    data_dict['target_rendering_wo_shadow_w_specular'] = self.dummy3
                else:
                    data_dict['target_specular_shading_wo_shadow'] = cv2.imread('%s/specular_shading_wo_shadow.exr' % target_humanenv_dir, -1).transpose(2,0,1)[[2,1,0]].astype(np.float32)
                    data_dict['target_rendering_wo_shadow_w_specular'] = cv2.imread('%s/rendering_wo_shadow_w_specular.exr' % target_humanenv_dir, -1).transpose(2,0,1)[[2,1,0]].astype(np.float32)
                
            if self.id == 'refineshadow' or (self.id == 'firststage' and self.opts.train_sigma):
                data_dict['source_diffuse_shading_w_shadow'] = cv2.imread('%s/diffuse_shading_w_shadow.exr' % source_humanenv_dir, -1).transpose(2,0,1)[[2,1,0]].astype(np.float32)

        if self.id == 'refineshadow' and self.opts.use_prepare_shadow:
            data_dict['source_hard_shadows'] = np.zeros((self.opts.n_light, self.opts.resolution, self.opts.resolution), dtype=np.float32)
            data_dict['source_soft_shadows'] = np.zeros((self.opts.n_light, self.opts.resolution, self.opts.resolution), dtype=np.float32)
            for i_light in range(self.opts.n_light):
                data_dict['source_hard_shadows'][i_light] = cv2.imread('%s/%s/%s/hard_shadows_%d.png' % (self.opts.shadow_dir, data_dict['human_id'], source_env_name_theta, i_light),
                                                                       cv2.IMREAD_GRAYSCALE).astype(np.float32)/255.
                data_dict['source_soft_shadows'][i_light] = cv2.imread('%s/%s/%s/soft_shadows_%d.png' % (self.opts.shadow_dir, data_dict['human_id'], source_env_name_theta, i_light),
                                                                       cv2.IMREAD_GRAYSCALE).astype(np.float32)/255.            

        return data_dict

    def rotation_als(self, als, theta):
        rot_mat = np.array([[np.cos(theta),0,-np.sin(theta)],
                                [0,1,0],
                                [np.sin(theta),0,np.cos(theta)]])
        als[:,0:3] = np.einsum('yx,dx->dy',rot_mat,als[:,0:3])
        return als.astype(np.float32)

    def get_random_mask_bg(self, mask):
        mask_bg = np.zeros_like(mask[0])
        _,b = np.where(mask[0])
        min_ind = random.randint(0, np.min(b))
        max_ind = random.randint(np.max(b), self.opts.resolution-1)
        mask_bg[:,min_ind:max_ind] = 1
        return mask_bg[None]