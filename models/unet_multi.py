import torch.nn as nn
import torch.nn.functional as F
import torch


class UNet_multi(nn.Module):
    def __init__(self, opts, in_channel=4, n_group=16):
        super(UNet_multi, self).__init__()
        self.opts = opts
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.enc = nn.ModuleList([])

        list_ch = [32,64,128,256,512]
        self.bottleneck_ch = [512,256,128]

        self.enc.append(nn.Sequential(nn.Conv2d(in_channel, list_ch[0], 3, padding=1, bias=False),
                                   nn.GroupNorm(n_group, list_ch[0]),
                                   nn.ReLU(inplace=True)))

        self.enc.append(nn.Sequential(self.avgpool,
                                        nn.Conv2d(list_ch[0], list_ch[1], 3, padding=1, bias=False),
                                        nn.GroupNorm(n_group, list_ch[1]),
                                        nn.ReLU(inplace=True)))
        self.enc.append(nn.Sequential(self.avgpool,
                                        nn.Conv2d(list_ch[1], list_ch[2], 3, padding=1, bias=False),
                                        nn.GroupNorm(n_group, list_ch[2]),
                                        nn.ReLU(inplace=True)))
        self.enc.append(nn.Sequential(self.avgpool,
                                        nn.Conv2d(list_ch[2], list_ch[3], 3, padding=1, bias=False),
                                        nn.GroupNorm(n_group, list_ch[3]),
                                        nn.ReLU(inplace=True)))
        self.enc.append(nn.Sequential(self.avgpool,
                                        nn.Conv2d(list_ch[3], list_ch[4], 3, padding=1, bias=False),
                                        nn.GroupNorm(n_group, list_ch[4]),
                                        nn.ReLU(inplace=True)))
        
        
        self.mid = nn.ModuleList([])
        self.mid.append(nn.Sequential(nn.Conv2d(list_ch[4], list_ch[4], 3, padding=1, bias=False),
                                nn.GroupNorm(n_group, list_ch[4]),
                                nn.ReLU(inplace=False)))
        self.mid.append(nn.Sequential(nn.Conv2d(list_ch[4], self.bottleneck_ch[0]+self.bottleneck_ch[1]+self.bottleneck_ch[2], 3, padding=1, bias=False),
                                nn.GroupNorm(n_group, self.bottleneck_ch[0]+self.bottleneck_ch[1]+self.bottleneck_ch[2]),
                                nn.ReLU(inplace=False)))


        self.dec_asr = nn.ModuleList([])
        self.dec_asr.append(nn.Sequential(nn.Conv2d(self.bottleneck_ch[0], list_ch[3], 3, padding=1, bias=False),
                                        nn.GroupNorm(n_group, list_ch[3]),
                                        nn.ReLU(inplace=True),
                                        self.upsample))
        self.dec_asr.append(nn.Sequential(nn.Conv2d(2*list_ch[3], list_ch[2], 3, padding=1, bias=False),
                                        nn.GroupNorm(n_group, list_ch[2]),
                                        nn.ReLU(inplace=True),
                                        self.upsample))
        self.dec_asr.append(nn.Sequential(nn.Conv2d(2*list_ch[2], list_ch[1], 3, padding=1, bias=False),
                                        nn.GroupNorm(n_group, list_ch[1]),
                                        nn.ReLU(inplace=True),
                                        self.upsample))

        self.dec_asr.append(nn.Sequential(nn.Conv2d(list_ch[2], list_ch[1], 3, padding=1, bias=False),
                                        nn.GroupNorm(n_group, list_ch[1]),
                                        nn.ReLU(inplace=True)))
        self.dec_asr.append(nn.Sequential(nn.Conv2d(list_ch[1], list_ch[0], 3, padding=1, bias=False),
                                        nn.GroupNorm(n_group, list_ch[0]),
                                        nn.ReLU(inplace=True),
                                        self.upsample))
        self.dec_asr.append(nn.Conv2d(list_ch[0], 5, 3, padding=1, bias=True))


        self.dec_n = nn.ModuleList([])
        self.dec_n.append(nn.Sequential(nn.Conv2d(self.bottleneck_ch[1], list_ch[3], 3, padding=1, bias=False),
                                        nn.GroupNorm(n_group, list_ch[3]),
                                        nn.ReLU(inplace=True),
                                        self.upsample))
        self.dec_n.append(nn.Sequential(nn.Conv2d(2*list_ch[3], list_ch[2], 3, padding=1, bias=False),
                                        nn.GroupNorm(n_group, list_ch[2]),
                                        nn.ReLU(inplace=True),
                                        self.upsample))
        self.dec_n.append(nn.Sequential(nn.Conv2d(2*list_ch[2], list_ch[1], 3, padding=1, bias=False),
                                        nn.GroupNorm(n_group, list_ch[1]),
                                        nn.ReLU(inplace=True),
                                        self.upsample))

        self.dec_n.append(nn.Sequential(nn.Conv2d(list_ch[2], list_ch[1], 3, padding=1, bias=False),
                                        nn.GroupNorm(n_group, list_ch[1]),
                                        nn.ReLU(inplace=True)))
        self.dec_n.append(nn.Sequential(nn.Conv2d(list_ch[1], list_ch[0], 3, padding=1, bias=False),
                                        nn.GroupNorm(n_group, list_ch[0]),
                                        nn.ReLU(inplace=True),
                                        self.upsample))
        self.dec_n.append(nn.Conv2d(list_ch[0], 3, 3, padding=1, bias=True))


        self.dec_als = nn.ModuleList([])
        self.dec_als.append(nn.Sequential(nn.Conv2d(self.bottleneck_ch[2], 1, 1, padding=0, bias=True),
                                        nn.ReLU(inplace=True)))
        self.dec_als.append(nn.Linear((self.opts.resolution // 2**(len(list_ch)-1))**2, self.opts.n_light*7))

    def forward(self, c0):
        c0 = self.enc[0](c0)#->32,1024^2
        c0 = self.enc[1](c0)#->64,512^2
        c1 = self.enc[2](c0)#->128,256^2
        c2 = self.enc[3](c1)#->256,128^2
        c3 = self.enc[4](c2)#->512,64^2

        
        m = c3 + self.mid[0](c3)#->512,64^2
        m = self.mid[1](m) #->896,64^2


        dc_asr = self.dec_asr[0](m[:,0:self.bottleneck_ch[0]])#512,64^2->256,128^2
        dc_asr = self.dec_asr[1](torch.cat([c2, dc_asr], 1))#512,128^2->128,256^2
        dc_asr = self.dec_asr[2](torch.cat([c1, dc_asr], 1))#256,256^2->64,512^2
        dc_asr = self.dec_asr[3](torch.cat([c0, dc_asr], 1)) #128,512^2->64,512^2
        dc_asr = self.dec_asr[4](dc_asr) #64,512^2->32,1024^2
        dc_asr = self.dec_asr[5](dc_asr) #32,1024^2->5,1024^2
        dc_asr = torch.sigmoid(dc_asr)

        dc_n = self.dec_n[0](m[:,self.bottleneck_ch[0]:self.bottleneck_ch[0]+self.bottleneck_ch[1]])#256,64^2->256,128^2
        dc_n = self.dec_n[1](torch.cat([c2, dc_n], 1))#512,128^2->128,256^2
        dc_n = self.dec_n[2](torch.cat([c1, dc_n], 1))#256,256^2->64,512^2
        dc_n = self.dec_n[3](torch.cat([c0, dc_n], 1)) #128,512^2->64,512^2
        dc_n = self.dec_n[4](dc_n) #64,512^2->32,1024^2
        dc_n = self.dec_n[5](dc_n) #32,1024^2->5,1024^2
        dc_n = F.normalize(dc_n, dim=1) 

        dc_als = self.dec_als[0](m[:,self.bottleneck_ch[0]+self.bottleneck_ch[1]:self.bottleneck_ch[0]+self.bottleneck_ch[1]+self.bottleneck_ch[2]])#128,64^2->1,64^2
        dc_als = self.dec_als[1](dc_als.view(self.opts.batch_size,-1))#1,64^2->[b,64^2]->[b,n_light*7]
        dc_als = dc_als.view(self.opts.batch_size,self.opts.n_light,7)#[b,n_light*7]->[b,n_light,7]
        dc_als = torch.cat([F.normalize(dc_als[:,:,0:3], dim=2),
                            self.opts.range_sigma * torch.sigmoid(dc_als[:,:,3:4]),
                            F.softplus(dc_als[:,:,4:7])],dim=2)

        return dc_asr[:,0:3], dc_n, dc_asr[:,3:4], dc_asr[:,4:5], dc_als