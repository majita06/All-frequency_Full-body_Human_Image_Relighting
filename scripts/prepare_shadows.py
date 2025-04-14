import os
from argparse import ArgumentParser
from tqdm import tqdm
import torch
import torch.nn.functional as F
import numpy as np
from utils import utils
import cv2
from shaders.shadow_mapping import ShadowMapping
from glob import glob

class GenerateShadows:
    def __init__(self, opts):
        self.opts = opts
        self.shadow_mapping = ShadowMapping(self.opts,resolution=self.opts.resolution)

        self.human_dirs_1 = sorted(glob('%s/*' % opts.dataset_dir))
        if self.opts.use_other_dataset:
            self.human_dirs_2 = sorted(glob('%s/*' % opts.other_dataset_dir))
            self.human_dirs = self.human_dirs_1 + self.human_dirs_2
        else:
            self.human_dirs = self.human_dirs_1
        self.human_dirs = self.human_dirs[opts.start:opts.end]

    def generate(self):
        for human_dir in tqdm(self.human_dirs,ncols=50):
            human_id = os.path.basename(human_dir)
            mask = utils.np2torch(cv2.imread('%s/mask.png' % human_dir, cv2.IMREAD_GRAYSCALE).astype(np.float32),self.opts.device)[None,None]/255.
            depth_gt = utils.np2torch(cv2.imread('%s/depth.exr' % human_dir, -1).astype(np.float32),self.opts.device)[None,None,...,0]
            normal_gt = utils.np2torch(cv2.imread('%s/normal.png' % human_dir, -1).astype(np.float32),self.opts.device).permute(2,0,1)[None,[2,1,0]]/255.
            normal_gt[:,0] = 2.*normal_gt[:,0]-1.
            normal_gt[:,1] = -(2.*normal_gt[:,1]-1.)
            normal_gt = F.normalize(normal_gt,dim=1)


            humanenv_dirs = [humanenv_dir[:-len('/')] for humanenv_dir in glob('%s/*/' % human_dir)]
            for humanenv_dir in humanenv_dirs:
                env_name_theta = os.path.basename(humanenv_dir)
                out_dir = '%s/%s/%s' % (self.opts.out_dir, human_id, env_name_theta)
                os.makedirs(out_dir, exist_ok=True)

                env_name, theta = env_name_theta.split('__')
                theta = float(theta)
                als = utils.np2torch(np.load('%s/%s/als.npy' % (self.opts.als_dir, env_name)),self.opts.device)[None] #[1,16,7]
                als = utils.rotation_als(als, theta)

                mask_vis = torch.einsum('bxhw,bdx->bdhw',normal_gt,als[:,:,0:3]).clamp(0,1)
                mask_vis[mask_vis>0] = 1
                with torch.no_grad():
                    hard_shadows, soft_shadows = self.shadow_mapping(depth_gt, als, mask, train=False)
                hard_shadows = mask_vis * hard_shadows
                soft_shadows = mask_vis * soft_shadows
                
                for i_light in range(self.opts.n_light):
                    cv2.imwrite('%s/hard_shadows_%d.png' % (out_dir, i_light),
                                utils.torch2np(255 * hard_shadows[0,i_light]))
                    cv2.imwrite('%s/soft_shadows_%d.png' % (out_dir, i_light),
                                utils.torch2np(255 * soft_shadows[0,i_light]))


def main():
    parser = ArgumentParser()
    parser.add_argument('--start', default=0, type=int)
    parser.add_argument('--end', default=99999, type=int)
    parser.add_argument('--out_dir', default='/home/tajima/dataset/shadow_dataset')
    parser.add_argument('--dataset_dir', default='/home/tajima/dataset/humgen_dataset') 
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
    opts = parser.parse_args()

    torch.manual_seed(opts.seed)
    torch.cuda.manual_seed(opts.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.use_deterministic_algorithms = False
    torch.autograd.set_detect_anomaly(True)

    GenerateShadows(opts).generate()

if __name__ == '__main__':
    main()