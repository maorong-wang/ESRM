from __future__ import print_function
from PIL import Image
import torchvision.datasets as datasets
import torch.utils.data as data
import torch
import math
import numpy as np

from src.datasets.tinyImageNet import TinyImageNet
from src.datasets.sync_tiny import SYNC_TINY

class MIXED_TINY(TinyImageNet):
    def __init__(self, root, syn_root, sync_versions=None, portion=.5, train=True, transform=None):
        super().__init__(root, train=True, download=False, transform=transform)
        self.portion = portion
        ## sort self.data and self.targets by self.targets
        self.targets = np.array(self.samples)[:,1].astype(int)
        samples = np.array(self.samples)[self.targets.argsort()]
        self.targets = self.targets[self.targets.argsort()]
        self.samples = samples.tolist()

        if sync_versions is None:
            self.sync_versions = ['v1.4']
        else:
            self.sync_versions = sync_versions
        
        self.sync_datasets = []
        for sync_version in self.sync_versions:
            self.sync_datasets.append(SYNC_TINY(syn_root, sync_version=sync_version, transform=transform))

    def __getitem__(self, index):
        if index % 500 < 500 * self.portion:
            sync_data_idx = math.floor(index % 500 / 500 / self.portion * len(self.sync_datasets))
            img, label, idx = self.sync_datasets[sync_data_idx][index]
            return img, label, idx, True
        img, label = super().__getitem__(index)
        label = torch.tensor(label)
        return img, label, index, False

if __name__ == '__main__':
    dataset = MIXED_TINY(root='/home/ma_wang/python/syn_ocl/data/tiny-imagenet-200', sync_versions=['v1.4'], portion=.5, train=True, transform=None)