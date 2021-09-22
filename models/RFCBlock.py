import torch
from torch import nn
from torch.nn import functional as F
import math

from models.SRFCModule import SRFCModule
from models.APU import APUModule, generate_coordinate, generate_masks


class ConvBlock(nn.Module):
    """Basic convolutional block"""
    def __init__(self, in_c, out_c, k, s=1, p=0):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, stride=s, padding=p)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        return self.bn(self.conv(x))


class SpatialAttn(nn.Module):
    """Spatial Attention """
    def __init__(self, in_channels, number=1):
        super(SpatialAttn, self).__init__()
        self.conv = ConvBlock(in_channels, number, 1)

    def forward(self, x):
        x = self.conv(x) #[bs, 1, h, w]
        a = torch.sigmoid(x)
        return a 


class RFCBlock(nn.Module):
    def __init__(self, in_channels, size, use_gpu=True):
        super(RFCBlock, self).__init__()
        self.in_channels = in_channels
        self.number = 6
        self.inter_channels = in_channels // 2
        conv_nd = nn.Conv2d
        bn = nn.BatchNorm2d
        self.use_gpu = use_gpu

        # APU
        self.APU = APUModule(in_channels, s=32, h=size[0], w=size[1], use_gpu=use_gpu)

        # spatial attention 
        self.SA = SpatialAttn(in_channels, number=1)

        # reduce dim
        self.conv_reduce = conv_nd(in_channels, self.inter_channels, kernel_size=1)

        # SRFC Module
        self.SRFC = SRFCModule(in_channels=self.inter_channels)

        # extend dim
        self.conv_extend = nn.Sequential(
                    conv_nd(self.inter_channels, in_channels, kernel_size=1),
                    bn(self.in_channels)
            )

        # init
        for m in self.modules():
            if isinstance(m, conv_nd):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, bn):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        nn.init.constant_(self.conv_extend[1].weight.data, 0.0)
        nn.init.constant_(self.conv_extend[1].bias.data, 0.0)


    def forward(self, x):
        inputs = x
        b, c, h, w = x.size()

        # APU
        keypoint_logit, keypoint_location = self.APU(x) 
        part_coordinate = generate_coordinate(x, keypoint_location, use_gpu=self.use_gpu) 

        part_masks = generate_masks(x, keypoint_location, use_gpu=self.use_gpu) 
        part_masks = part_masks.detach().view(b, 6, -1)

        # Apply the foreground maps
        foreground = self.SA(x) 
        maps = foreground.view(b, 1, -1) 
        maps = torch.div(maps, torch.sum(maps, dim=2, keepdim=True))
        x = (maps.view(b, 1, h, w) + 1) * x

        # reduce dim
        x_reduce = self.conv_reduce(x).view(b, self.inter_channels, -1) 

        # extract the parts
        part_masks_norm = part_masks / torch.sum(part_masks, dim=2, keepdim=True)
        x = torch.bmm(part_masks_norm, x_reduce.transpose(1, 2)) 

        # SRFC Module
        y, distance_list = self.SRFC(x, part_coordinate) 

        # reverse projection
        part_masks_rproj = part_masks.transpose(1, 2) 
        x = torch.bmm(part_masks_rproj, y) 
        x = x.transpose(1, 2).view(b, self.inter_channels, h, w)

        # expand dim
        x = self.conv_extend(x) + inputs
        return x, foreground, keypoint_logit, distance_list
