import torch

from src.utils.utils import filter_labels
from src.datasets.sync_cifar100 import SYNC_CIFAR100


class SplitSYNC_CIFAR100(SYNC_CIFAR100):
    def __init__(self, sync_version=None, train=True, transform=None, selected_labels=[0]):
        super().__init__(sync_version=sync_version, transform=transform)  # versions and portions are temporary hanging here
        self.selected_labels = selected_labels
        self.targets = torch.Tensor(self.targets)
        self.indexes = torch.nonzero(filter_labels(self.targets, self.selected_labels)).flatten()

    def __getitem__(self, index):
        img, label, _ = super().__getitem__(self.indexes[index])
        return img, label, self.indexes[index], True

    def __len__(self):
        return len(self.indexes)