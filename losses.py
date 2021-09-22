from __future__ import absolute_import
from __future__ import division

import sys

import torch
from torch import nn
from torch.nn import functional as F


def DeepSupervision(criterion, xs, y):
    """
    Args:
    - criterion: loss function
    - xs: tuple of inputs
    - y: ground truth
    """
    loss = 0.
    for x in xs:
        loss += criterion(x, y)
    loss /= len(xs)
    return loss


class ClusterLoss(nn.Module):
    def __init__(self):
        super(ClusterLoss, self).__init__()
        self.loss = nn.L1Loss(reduction='mean')

    def forward(self, distance_list):
        d1, d2 = distance_list #[bs, 6, 6]

        d1 = d1.view(-1, d1.size(-1)) #[N, 36]
        d2 = d2.view(-1, d2.size(-1)) #[N, 36]

        output = self.loss(d1, d2) * d1.size(-1) 
        return output


class MaskLoss(nn.Module):
    """L2 or L1 loss or cross entropy loss with average with all elements.
    """
    def __init__(self):
        super(MaskLoss, self).__init__()
        self.loss = nn.BCELoss()

    def forward(self, inputs, targets):
        """
        Args:
        - inputs: prediction spatial map with shape (batch_size, 1, h, w)
        - targets: ground truth labels with shape (batch_size, 1, h1, w1)
        """
        b, c, h, w = inputs.size()
        targets = F.interpolate(targets, (h, w), mode='bilinear', align_corners=True)
        inputs = inputs.view(b, -1)
        targets = targets.view(b, -1)
        return self.loss(inputs, targets)


class HardSplitLoss(nn.Module):
    def __init__(self):
        super(HardSplitLoss, self).__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, part_logit, targets):
        """
        Args:
        - y1, y2, y3, y4: prediction matrix (before softmax) with shape (batch_size, num_classes)
        - targets: ground truth labels with shape (batch_size, 4)
        """
        y1, y2, y3, y4 = part_logit
        l1 = self.loss(y1, targets[:, 0])
        l2 = self.loss(y2, targets[:, 1])
        l3 = self.loss(y3, targets[:, 2])
        l4 = self.loss(y4, targets[:, 3])
        return (l1 + l2 + l3 + l4) / 4


class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.

    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.

    Args:
    - num_classes (int): number of classes.
    - epsilon (float): weight.
    """
    def __init__(self, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
        - inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
        - targets: ground truth labels with shape (batch_size)
        """
        self.num_classes = inputs.size(-1)
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss


class TripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.

    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.

    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.

    Args:
    - margin (float): margin for triplet.
    """
    def __init__(self, margin=0.5, distance='consine', use_gpu=True):
        super(TripletLoss, self).__init__()
        if distance not in ['euclidean', 'consine']:
            raise KeyError('Unsupported distance: {}'.format(distance))
        self.distance = distance
        self.margin = margin
        self.use_gpu = use_gpu
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        """
        Args:
        - inputs: feature matrix with shape (batch_size, feat_dim)
        - targets: ground truth labels with shape (num_classes)
        """
        n = inputs.size(0)
        
        # Compute pairwise distance, replace by the official when merged
        if self.distance == 'euclidean':
            dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
            dist = dist + dist.t()
            dist.addmm_(1, -2, inputs, inputs.t())
            dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        elif self.distance == 'consine':
            fnorm = torch.norm(inputs, p=2, dim=1, keepdim=True)
            l2norm = inputs.div(fnorm.expand_as(inputs))
            dist = - torch.mm(l2norm, l2norm.t())

        if self.use_gpu: targets = targets.cuda()
        
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        
        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        return loss
