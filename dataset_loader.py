from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
from PIL import Image
import numpy as np
import random
import math
import os.path as osp
import lmdb
import io

import torch
from torch.utils.data import Dataset


def read_image(img_path, mode='RGB'):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert(mode)
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class ImageDataset(Dataset):
    """Image Person ReID Dataset"""
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid = self.dataset[index]
        img = read_image(img_path)
        if self.transform is not None:
            self.transform.randomize_parameters()
            img = self.transform(img)
        return img, pid, camid


class ImageDataset_hardSplit_seg(Dataset):
    """Image Person ReID Dataset"""
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        path, pid, camid, splits = self.dataset[index]
        img_path, foreground_path = path
        split_layer1, split_layer2, split_layer3, split_layer4 = splits

        img = read_image(img_path, mode='RGB')
        foreground = read_image(foreground_path, mode='L')

        sequence = [img, foreground, split_layer1, split_layer2, split_layer3, split_layer4]

        if self.transform is not None:
            self.transform.randomize_parameters()
            sequence = [self.transform(img) for img in sequence]

        img, foreground, split_layer1, split_layer2, split_layer3, split_layer4 = sequence
        split_layer1 = np.array(split_layer1[2:])
        split_layer2 = np.array(split_layer2[2:])
        split_layer3 = np.array(split_layer3[2:])
        split_layer4 = np.array(split_layer4[2:])

        return img, foreground, pid, camid, split_layer1, split_layer2, split_layer3, split_layer4