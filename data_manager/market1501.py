from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob
import re
import sys
import urllib
import tarfile
import zipfile
import json
import os.path as osp
from scipy.io import loadmat
import numpy as np
import h5py

from .base import BaseImgDataset


class Market1501(BaseImgDataset):
    """
    Market1501

    Reference:
    Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.

    URL: http://www.liangzheng.org/Project/project_reid.html
    
    Dataset statistics:
    # identities: 1501 (+1 for background)
    # images: 12936 (train) + 3368 (query) + 15913 (gallery)
    """
    def __init__(self, dataset_dir, fore_dir, verbose=True):
        super(Market1501, self).__init__()
        self.dataset_dir = dataset_dir
        self.key_points_dir = './key_points/'

        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')
        
        self.key_points_train = osp.join(self.key_points_dir, 'market_train_key_points.json')
        self.fore_maps_train = fore_dir

        self._check_before_run()

        train, num_train_pids, num_train_imgs = self._process_dir(self.train_dir, self.key_points_train, self.fore_maps_train, relabel=True)
        query, num_query_pids, num_query_imgs = self._process_dir(self.query_dir, relabel=False)
        gallery, num_gallery_pids, num_gallery_imgs = self._process_dir(self.gallery_dir, relabel=False)
        num_total_pids = num_train_pids + num_query_pids
        num_total_imgs = num_train_imgs + num_query_imgs + num_gallery_imgs

        if verbose:
            print("=> Market1501 loaded")
            print("Dataset statistics:")
            print("  ------------------------------")
            print("  subset   | # ids | # images")
            print("  ------------------------------")
            print("  train    | {:5d} | {:8d}".format(num_train_pids, num_train_imgs))
            print("  query    | {:5d} | {:8d}".format(num_query_pids, num_query_imgs))
            print("  gallery  | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_imgs))
            print("  ------------------------------")
            print("  total    | {:5d} | {:8d}".format(num_total_pids, num_total_imgs))
            print("  ------------------------------")

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids = num_train_pids
        self.num_query_pids = num_query_pids
        self.num_gallery_pids = num_gallery_pids

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _generate_split(self, split_param, target_size):
        h, w = target_size  
        split_h1 = round(split_param[0] * h) - 1
        split_h1 = max(0, split_h1)
        split_h1 = min(split_h1, h-4)

        split_h2 = round(split_param[1] * h) - 1
        split_h2 = max(0, split_h2)
        split_h2 = min(split_h2, h-3)

        split_h3 = round(split_param[2] * h) - 1
        split_h3 = max(0, split_h3)
        split_h3 = min(split_h3, h-2)

        split_w = round(split_param[3] * w) - 1
        split_w = max(0, split_w)
        split_w = min(split_w, w-2)
        return [h, w, split_h1, split_h2, split_h3, split_w]

    def _process_dir(self, dir_path, key_points_file=None, fore_maps_dir=None, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')
        if key_points_file is not None:
            key_points_params = json.load(open(key_points_file, 'r'))

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid:label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            assert 0 <= pid <= 1501  # pid == 0 means background
            assert 1 <= camid <= 6
            camid -= 1 # index starts from 0
            if relabel: pid = pid2label[pid]

            if key_points_file is None:
                dataset.append((img_path, pid, camid))
            else:
                split_param = key_points_params[img_path.split('/')[-1]]
                split_layer1 = self._generate_split(split_param, target_size=(64, 32))
                split_layer2 = self._generate_split(split_param, target_size=(32, 16))
                split_layer3 = self._generate_split(split_param, target_size=(16, 8))
                split_layer4 = self._generate_split(split_param, target_size=(16, 8))

                assert fore_maps_dir is not None
                img_name = img_path.split('/')[-1].split('.')[0]
                foreground_path = osp.join(fore_maps_dir, img_name, 'foreground.png')

                assert os.path.exists(foreground_path)
                img_path = [img_path, foreground_path]
                dataset.append((img_path, pid, camid, [split_layer1, split_layer2, split_layer3, split_layer4]))

        num_pids = len(pid_container)
        num_imgs = len(dataset)
        return dataset, num_pids, num_imgs


if __name__ == '__main__':
    data = Market1501()

