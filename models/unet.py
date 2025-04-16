import torch.nn as nn
import torch.nn.functional as F
import torch


class UNet(nn.Module):
    def __init__(self, opts, in_channels,out_channels,n_layer=4,n_group=16):
        super(UNet, self).__init__()
        self.n_layer = n_layer
        list_ch = [2**(5+i) for i in range(n_layer+1)]

        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        
        self.start = nn.Sequential(nn.Conv2d(in_channels, list_ch[0], 3, padding=1, bias=False),
                                   nn.GroupNorm(n_group, list_ch[0]),
                                   nn.ReLU(inplace=True))

        self.enc = nn.ModuleList([])
        for i in range(n_layer):
            self.enc.append(nn.Sequential(self.avgpool,
                                          nn.Conv2d(list_ch[i], list_ch[i+1], 3, padding=1, bias=False),
                                          nn.GroupNorm(n_group, list_ch[i+1]),
                                          nn.ReLU(inplace=True)))

        self.mid = nn.Sequential(nn.Conv2d(list_ch[n_layer], list_ch[n_layer], 3, padding=1, bias=False),
                                nn.GroupNorm(n_group, list_ch[n_layer]),
                                nn.ReLU(inplace=False))

        self.dec = nn.ModuleList([])
        for i in range(n_layer):
            self.dec.append(nn.Sequential(nn.Conv2d(2 * list_ch[n_layer-i], list_ch[n_layer-i-1], 3, padding=1, bias=False),
                                          nn.GroupNorm(n_group, list_ch[n_layer-i-1]),
                                          nn.ReLU(inplace=True)))

        self.end = nn.Conv2d(2 * list_ch[0], out_channels, 3, padding=1, bias=True)

    def forward(self, x):
        x = self.start(x)

        out_enc = [x]
        for i in range(self.n_layer):
            x = self.enc[i](x)
            out_enc.append(x)

        x = self.avgpool(out_enc[-1])
        x = x + self.mid(x)

        for i in range(self.n_layer):
            x = self.dec[i](torch.cat([out_enc[self.n_layer-i],
                                       self.upsample(x)],1))

        x = self.end(torch.cat([out_enc[0],
                                self.upsample(x)],
                                1))
        return x