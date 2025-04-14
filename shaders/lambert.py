
import torch
import torch.nn.functional as F
import numpy as np

class Lambert():
    def __init__(self):
        pass
    def lambert(self, normal, als, per_light=False):
        if per_light:
            shading = torch.einsum('bdhw,bdc->bdchw',
                                    torch.einsum('bdx,bxhw->bdhw',
                                                als[:,:,0:3],
                                                normal).clamp(0,1),
                                    als[:,:,4:7])
        else:
            shading = torch.einsum('bdhw,bdc->bchw',
                                    torch.einsum('bdx,bxhw->bdhw',
                                                als[:,:,0:3],
                                                normal).clamp(0,1),
                                    als[:,:,4:7])
        return (1/np.pi) * shading