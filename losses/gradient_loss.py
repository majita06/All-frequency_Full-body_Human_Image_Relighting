import torch
import numpy as np
import torch.nn.functional as F
import cv2
from utils import utils
import torch.nn as nn
class Gradient_loss():
    def __init__(self,opts):
        self.opts = opts
        kernel_x = torch.tensor([[0., 0., 0.],
                                [-0.5, 0., 0.5],
                                [0., 0., 0.]], dtype=torch.float32)
        kernel_y = torch.tensor([[0., -0.5, 0.],
                                [0., 0., 0.],
                                [0., 0.5, 0.]], dtype=torch.float32)
        self.conv_x = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_y = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_x.weight.data = kernel_x[None,None]
        self.conv_y.weight.data = kernel_y[None,None]
        self.conv_x.weight.requires_grad = False
        self.conv_y.weight.requires_grad = False
        self.conv_x = self.conv_x.to(self.opts.device)
        self.conv_y = self.conv_y.to(self.opts.device)

        fov = 2*np.arctan(0.5*36/self.opts.focal_length)
        ones = torch.ones((self.opts.batch_size,1,self.opts.resolution,self.opts.resolution)).to(self.opts.device)
        coef = 2*self.opts.camera_distance*np.tan(fov*0.5)/self.opts.resolution
        self.normal_z = coef * ones 

    def gradient_loss(self, depth, depth_hat):

        _depth = depth*(self.opts.d_all_max-self.opts.d_all_min)+self.opts.d_all_min # 元の深度値に戻す
        dzdx = self.conv_x(_depth)
        dzdy = self.conv_y(_depth)
        normal = torch.cat([dzdx,dzdy,self.normal_z],1)
        normal = F.normalize(normal,dim=1)

        _depth_hat = depth_hat*(self.opts.d_all_max-self.opts.d_all_min)+self.opts.d_all_min # 元の深度値に戻す
        dzdx_hat = self.conv_x(_depth_hat)
        dzdy_hat = self.conv_y(_depth_hat)
        normal_hat = torch.cat([dzdx_hat,dzdy_hat,self.normal_z],1)
        normal_hat = F.normalize(normal_hat,dim=1)

        loss = F.l1_loss(normal,normal_hat)

        return loss