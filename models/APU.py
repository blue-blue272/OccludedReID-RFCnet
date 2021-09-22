import torch
from torch import nn
from torch.nn import functional as F
import math


class ConvBlock(nn.Module):
    """Basic convolutional block"""
    def __init__(self, in_c, out_c, k, s=1, p=0):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, stride=s, padding=p)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        return self.bn(self.conv(x))


class APUModule(nn.Module):
    def __init__(self, in_channels, s, h, w, use_gpu):
        super(APUModule, self).__init__()
        self.use_gpu = use_gpu
        self.h, self.w = h, w
        self.inter_channels = in_channels//s

        self.conv = ConvBlock(in_channels, self.inter_channels, 1)
        self.relu = nn.ReLU(inplace=True)

        self.cls_ver = nn.Conv2d(self.inter_channels*h, h-3 + h-2 + h-1, 1)
        self.cls_hor = nn.Conv2d(self.inter_channels*w, w-1, 1)

    def forward(self, x):
        h, w = self.h, self.w
        x = self.conv(x) 
        x_feature = self.relu(x)

        # predict the vertical key points
        x = x_feature.mean(3)
        x = x.transpose(1, 2).contiguous() 
        x = x.view(x.size(0), -1, 1, 1) 
        logit = self.cls_ver(x).squeeze(-1).squeeze(-1)
        logit1 = logit[:, :h-3] 
        logit2 = logit[:, h-3: h-3+h-2] 
        logit3 = logit[:, h-3+h-2:] 

        # predict the hortical key points
        x = x_feature.mean(2) 
        x = x.transpose(1, 2).contiguous() 
        x = x.view(x.size(0), -1, 1, 1) 
        logit4 = self.cls_hor(x).squeeze(-1).squeeze(-1) 

        # predict the key points parameter
        index1 = torch.argmax(logit1, dim=1) 

        index2 = torch.argmax(logit2, dim=1) 
        index2 = torch.max(index2, index1 + 1)

        index3 = torch.argmax(logit3, dim=1) 
        index3 = torch.max(index3, index2 + 1)

        index4 = torch.argmax(logit4, dim=1)

        return [logit1, logit2, logit3, logit4], [index1, index2, index3, index4]


def generate_coordinate(x, split_location, use_gpu):
    bs, c, h, w = x.size()
    split_ver1, split_ver2, split_ver3, split_hori = split_location
    split_ver1 = split_ver1.type(x.dtype).unsqueeze(1) + 1
    split_ver2 = split_ver2.type(x.dtype).unsqueeze(1) + 1
    split_ver3 = split_ver3.type(x.dtype).unsqueeze(1) + 1
    split_hori = split_hori.type(x.dtype).unsqueeze(1) + 1

    zeros = torch.zeros((bs, 1)) 
    ones = torch.ones((bs, 1)) 
    if use_gpu:
        zeros, ones = zeros.cuda(), ones.cuda()

    part1 = torch.cat((zeros, zeros, split_ver1 / h, ones), 1).unsqueeze(1) 
    part2 = torch.cat((split_ver1 / h, zeros, (split_ver2 - split_ver1) / h, ones), 1).unsqueeze(1)
    part3 = torch.cat((split_ver2 / h, zeros, (split_ver3 - split_ver2) / h, split_hori / w), 1).unsqueeze(1)
    part4 = torch.cat((split_ver2 / h, split_hori / w, (split_ver3 - split_ver2) / h, 1 - split_hori / w), 1).unsqueeze(1)
    part5 = torch.cat((split_ver3 / h, zeros, 1 - split_ver3 / h, split_hori / w), 1).unsqueeze(1)
    part6 = torch.cat((split_ver3 / h, split_hori / w, 1 - split_ver3 / h, 1 - split_hori / w), 1).unsqueeze(1)

    part = torch.cat([part1, part2, part3, part4, part5, part6], 1)

    return part


def generate_masks(x, split_location, use_gpu):
    bs, c, h, w = x.size()
    split_ver1, split_ver2, split_ver3, split_hori = split_location

    index_ver = torch.arange(h)
    index_hori = torch.arange(w)
    index_ver = index_ver.unsqueeze(0).repeat(bs, 1) 
    index_hori = index_hori.unsqueeze(0).repeat(bs, 1) 
    if use_gpu:
        index_ver, index_hori = index_ver.cuda(), index_hori.cuda()

    # generate the head mask
    mask_head = torch.le(index_ver, split_ver1.unsqueeze(1)).type(x.dtype) 

    # generate the upper mask
    mask_upper = torch.le(index_ver, split_ver2.unsqueeze(1)).type(x.dtype) 
    mask_upper = mask_upper - mask_head 

    # generate the bottom mask
    mask_bottom = torch.le(index_ver, split_ver3.unsqueeze(1)).type(x.dtype)
    mask_bottom = mask_bottom - mask_head - mask_upper

    # generate the shoes mask
    mask_shoes = 1 - mask_bottom - mask_head - mask_upper  

    # repeat the generated masks
    mask_head = mask_head.unsqueeze(2).repeat(1, 1, w) 
    mask_upper = mask_upper.unsqueeze(2).repeat(1, 1, w) 
    mask_bottom = mask_bottom.unsqueeze(2).repeat(1, 1, w) 
    mask_shoes = mask_shoes.unsqueeze(2).repeat(1, 1, w) 

    # generate the left mask
    mask_left = torch.le(index_hori, split_hori.unsqueeze(1)).type(x.dtype) 
    mask_left = mask_left.unsqueeze(1).repeat(1, h, 1) 

    # generate the final mask
    mask_bottom_left = mask_bottom * mask_left
    mask_bottom_right = mask_bottom * (1 - mask_left)
    mask_shoes_left = mask_shoes * mask_left
    mask_shoes_right = mask_shoes * (1 - mask_left)

    mask_total = []
    for m in [mask_head, mask_upper, mask_bottom_left, mask_bottom_right, mask_shoes_left, mask_shoes_right]:
        mask_total.append(m.unsqueeze(1))
    
    mask_total = torch.cat(mask_total, 1)
    return mask_total.type(x.dtype)