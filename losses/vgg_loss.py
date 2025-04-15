"""A VGG-based perceptual loss function for PyTorch."""

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models, transforms


class VGGLoss(nn.Module):

    def __init__(self, opts, model='vgg16', layer=8, reduction='mean'):
        super().__init__()
        self.opts = opts
        self.reduction = reduction
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
        if model == 'vgg16':
            self.model = models.vgg16(pretrained=True).features[:layer+1]
        elif model == 'vgg19':
            self.model = models.vgg19(pretrained=True).features[:layer+1]
        self.model.to(self.opts.device)
        self.model.eval()
        self.model.requires_grad_(False)

    def forward(self, input, target):
        sep = input.shape[0]
        batch = torch.cat([input, target])
        feats = self.model(self.normalize(batch))
        input_feats, target_feats = feats[:sep], feats[sep:]
        return F.mse_loss(input_feats, target_feats, reduction=self.reduction)