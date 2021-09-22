import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class SRFCModule(nn.Module):
    def __init__(self, in_channels):
        super(SRFCModule, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 3, 1)
        self.conv2 = nn.Sequential(
                nn.Conv2d(4, in_channels, 1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, 3, 1)
            )

        self.conv = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
            )

        self.conv3= nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
            )

    def get_distance1(self, x, z):
        x_norm = F.normalize(x, p=2, dim=2, eps=1e-12)
        d1 = torch.bmm(x_norm, x_norm.transpose(1, 2)) 

        z_norm = F.normalize(z, p=2, dim=2, eps=1e-12)
        d2 = torch.bmm(z_norm, z_norm.transpose(1, 2)) 
        return [d2, d1]


    def get_distance2(self, part_coordinate, z):
        yc = part_coordinate[:, :, 0] + part_coordinate[:, :, 2] // 2 
        xc = part_coordinate[:, :, 1] + part_coordinate[:, :, 3] // 2 
        dy = torch.pow(yc.unsqueeze(1) - yc.unsqueeze(2), 2)
        dx = torch.pow(xc.unsqueeze(1) - xc.unsqueeze(2), 2) 
        d1 = torch.sqrt(dx + dy)
        d1 = 1 - 2 * d1

        z_norm = F.normalize(z, p=2, dim=2, eps=1e-12)
        d2 = torch.bmm(z_norm, z_norm.transpose(1, 2)) 
        return [d2, d1]

    def forward(self, x, part_coordinate):
        bs, n, c = x.size()
        eps = 1e-12

        z1 = self.conv1(x.view(-1, c, 1, 1)).view(bs, n, -1) 
        z1 = F.softmax(z1, 2) 
        distance_list1 = self.get_distance1(x, z1)

        z2 = self.conv2(part_coordinate.view(-1, 4, 1, 1)).view(bs, n, -1) 
        z2 = F.softmax(z2, 2)
        distance_list2 = self.get_distance2(part_coordinate, z2)

        z = z1 + z2
        encode_matrix = z / (torch.sum(z, 1, keepdim=True) + eps) 
        decode_matrix = z / (torch.sum(z, 2, keepdim=True) + eps) 

        u = torch.bmm(encode_matrix.transpose(1, 2), x)
        u = self.conv(u.view(-1, c, 1, 1)).view(u.size()) 
        y = torch.bmm(decode_matrix, u) 
        y = self.conv3(y.view(-1, c, 1, 1)).view(y.size())
        y = y + x
        return y, [distance_list1, distance_list2]