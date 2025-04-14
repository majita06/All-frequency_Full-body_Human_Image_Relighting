import torch
import kornia


def rmse_w_mask(img_a,img_b,mask):
    if img_a.shape[0] != mask.shape[0]:
        raise ValueError("Input tensor and mask must have the same batch size.")
    mse = torch.sum(mask * (img_a - img_b)**2) / (torch.sum(mask.expand(-1,img_a.shape[1],-1,-1)))
    rmse = torch.sqrt(mse).item()
    return rmse

def ssim_w_mask(img_a,img_b,mask):
    if img_a.shape[0] != mask.shape[0]:
        raise ValueError("Input tensor and mask must have the same batch size.")
    ssim_map = kornia.losses.ssim_loss(img_a,img_b,window_size=11,reduction='none')
    dssim = torch.sum(mask * ssim_map)/(torch.sum(mask.expand(-1,img_a.shape[1],-1,-1)))
    ssim = 1-2*dssim.item()
    return ssim

class LPIPS():
    def __init__(self,opts):
        self.opts = opts
        import lpips
        self.lp_loss = lpips.LPIPS(net='alex').to(self.opts.device)
    def lpips(self,img_a,img_b):
        loss = torch.mean(self.lp_loss.forward(img_a.clamp(0,1),
                                               img_b.clamp(0,1),
                                               normalize=True))
        return loss.item()