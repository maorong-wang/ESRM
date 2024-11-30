from __future__ import print_function
from PIL import Image
import torchvision.datasets as datasets
import torch.utils.data as data
import torch
import math
import numpy as np 

from src.datasets.ImageNet100 import ImageNet100
from src.datasets.sync_in100 import SYNC_ImageNet100

class MIXED_IN100(ImageNet100):
    def __init__(self, root, syn_root, sync_versions=None, portion=.5, train=True, transform=None):
        super().__init__(root, train=True, transform=transform)
        self.portion = portion
        ## sort self.data and self.targets by self.targets
        if sync_versions is None:
            self.sync_versions = ['xl']
        else:
            self.sync_versions = sync_versions
        
        self.sync_datasets = []
        for sync_version in self.sync_versions:
            self.sync_datasets.append(SYNC_ImageNet100(syn_root, sync_version=sync_version, transform=transform))

        self.idx, cnt = torch.unique(torch.tensor(self.sync_datasets[0].targets), return_counts=True)
        self.cum = np.cumsum(cnt.numpy())
        self.idx = np.concatenate(([-1], self.idx))
        self.cum = np.concatenate(([0], self.cum))

    def __getitem__(self, index):
        percental = np.interp(index, self.cum, self.idx) - np.floor(np.interp(index, self.cum, self.idx))
        if percental < self.portion:
            img, label = self.sync_datasets[0][index]
            return img, label, index, True
        img, label = super().__getitem__(index)
        return img, label, index, False


