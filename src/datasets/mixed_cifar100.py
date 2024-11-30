from __future__ import print_function
from PIL import Image
import torchvision.datasets as datasets
import torch.utils.data as data
import torch
import math

from src.datasets.cifar100 import CIFAR100
from src.datasets.sync_cifar100 import SYNC_CIFAR100

class MIXED_CIFAR100(CIFAR100):
    def __init__(self, root, syn_root, sync_versions=None, portion=.5, train=True, transform=None):
        super().__init__(root, train=True, download=False, transform=transform)
        self.portion = portion
        ## sort self.data and self.targets by self.targets
        self.data = self.data[self.targets.argsort()]
        self.targets = self.targets[self.targets.argsort()]

        if sync_versions is None:
            self.sync_versions = ['v1.4']
        else:
            self.sync_versions = sync_versions
        
        self.sync_datasets = []
        for sync_version in self.sync_versions:
            self.sync_datasets.append(SYNC_CIFAR100(syn_root, sync_version=sync_version, transform=transform))

    def __getitem__(self, index):
        if index % 500 < 500 * self.portion:
            sync_data_idx = math.floor(index % 500 / 500 / self.portion * len(self.sync_datasets))
            img, label, idx = self.sync_datasets[sync_data_idx][index]
            return img, label, idx, True
        img, label, idx = super().__getitem__(index)
        return img, label, idx, False


