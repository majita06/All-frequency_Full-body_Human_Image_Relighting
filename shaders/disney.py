import torch
import torch.nn.functional as F
import numpy as np

class Disney:
    def __init__(self,opts):
        self.opts = opts
        self.V = torch.tensor([0,0,1]).to(self.opts.device)[None,None,:,None,None].to(torch.float32) #[1,1,x,1,1]
        self.eps = 1e-6

    def schlickWeight(self,A):
        out = (1-A).clamp(self.eps,1)
        return out**5

    def D_GGX(self,NH,alpha):
        t = 1 + (alpha**2 - 1) * NH**2
        return alpha**2/(np.pi * t**2)

    def _G_GGX(self,NX,alpha):
        a2 = alpha**2
        b2 = NX**2
        return 1/(NX + torch.sqrt((a2+b2-a2*b2).clamp(self.eps,None)))
    
    def G_GGX(self,NL,NV,alpha):
        out = self._G_GGX(NL,alpha)*self._G_GGX(NV,alpha)
        return out

    def disney(self, N, L, specular, roughness, per_light=False, view=None):
        if view is not None:
            wo = view[:,None] #[b,1,x,h,w]
        else:
            wo = self.V #[1,1,x,1,1]

        L = L[...,None,None] #[b,d,7] -> [b,d,7,1,1]
        N = N[:,None] #[b,x,h,w] -> [b,1,x,h,w]
        specular = specular[:,None] #[b,1,h,w] -> [b,1,1,h,w]
        roughness = roughness[:,None] #[b,1,h,w] -> [b,1,1,h,w]
        
        H = F.normalize(L[:,:,0:3]+wo,dim=2)
        NL = torch.sum(N * L[:,:,0:3],dim=2,keepdim=True).clamp(0,1)
        NV = torch.sum(N * wo, dim=2, keepdim=True).clamp(0,1)
        LH = torch.sum(L[:,:,0:3] * H, dim=2, keepdim=True)
        NH = torch.sum(N * H, dim=2, keepdim=True).clamp(0,1)

        Cspec0 = specular * 0.08
        fresnel = Cspec0 + (1-Cspec0)*self.schlickWeight(LH)
        
        alpha_specular = (roughness**2).clamp(0.001,None) 

        D = self.D_GGX(NH,alpha_specular)
        G = self.G_GGX(NL,NV,alpha_specular)

        fr = fresnel * D * G

        shading_specular = NL * fr * L[:,:,4:7]
        
        if not per_light:
            shading_specular = torch.sum(shading_specular, dim=1) #[b,d,c,h,w] -> [b,c,h,w]
        
        return shading_specular





