
import torch
import torch.nn.functional as F
import numpy as np


class RepulsiveLoss():
    def __init__(self,threshold=0.65):
        self.threshold = threshold

    def repulsive_loss(self, light):
        light_direction = light[:,:,0:3].clone()
        n_light = light_direction.shape[1]
        loss = 0
        for i_light in range(1,n_light):
            light_direction_roll = torch.roll(light_direction,shifts=i_light,dims=1)
            _loss = ((torch.sum(light_direction * light_direction_roll, dim=2)-self.threshold).clamp(0,None)/(1-self.threshold))**2
            loss = loss + torch.mean(_loss)
        return loss/(n_light-1)
