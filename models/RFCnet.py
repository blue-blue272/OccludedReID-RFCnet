from __future__ import absolute_import
from __future__ import division

import torch
from torch import nn
from torch.nn import functional as F
import torchvision
from models.RFCBlock import RFCBlock


class RFCnet(nn.Module):
    def __init__(self, num_classes, use_gpu=True):
        super(RFCnet, self).__init__()
        resnet50 = torchvision.models.resnet50(pretrained=True)
        resnet50.layer4[0].conv2.stride=(1, 1)
        resnet50.layer4[0].downsample[0].stride=(1, 1) 

        self.conv1 = resnet50.conv1
        self.bn1 = resnet50.bn1
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = resnet50.maxpool

        self.layer1 = resnet50.layer1 
        self.layer2 = resnet50.layer2 
        self.layer3 = resnet50.layer3 
        self.layer4 = resnet50.layer4 

        self.RFC1 = RFCBlock(in_channels=512, size=(32, 16))
        self.RFC2 = RFCBlock(in_channels=1024, size=(16, 8))

        self.feat_dim = 2048
        self.bn = nn.BatchNorm1d(self.feat_dim)
        self.classifier = nn.Linear(self.feat_dim, num_classes)

    def forward(self, x):
        x = self.bn1(self.conv1(x))
        x = self.maxpool(self.relu(x)) 

        x1 = self.layer1(x) 
        x2 = self.layer2(x1) 
        x2, fore1, keypoint_logit1, distance_list1 = self.RFC1(x2) 
        x3 = self.layer3(x2)
        x3, fore2, keypoint_logit2, distance_list2 = self.RFC2(x3)
        x4 = self.layer4(x3) 

        f = F.avg_pool2d(x4, x4.size()[2:])
        f = f.view(f.size(0), -1)
        f = self.bn(f)

        fore = [fore1, fore2]
        keypoint_logit = [keypoint_logit1, keypoint_logit2]
        distance_list = [distance_list1, distance_list2]

        if not self.training:
            return f

        y = self.classifier(f)
        return y, f, fore, keypoint_logit, distance_list
