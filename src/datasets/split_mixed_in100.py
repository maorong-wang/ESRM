import torch

from src.utils.utils import filter_labels
from src.datasets.mixed_in100 import MIXED_IN100


class SplitMIXED_IN100(MIXED_IN100):
    def __init__(self, root, syn_root, sync_versions=None, portion=.5, train=True, transform=None, selected_labels=[0]):
        super().__init__(root=root, syn_root=syn_root, sync_versions=sync_versions, portion=portion, transform=transform)  # versions and portions are temporary hanging here
        self.selected_labels = selected_labels
        self.targets = torch.Tensor(self.targets)
        self.indexes = torch.nonzero(filter_labels(self.targets, self.selected_labels)).flatten()

    def __getitem__(self, index):
        img, label, _, isSyn = super().__getitem__(self.indexes[index])
        return img, label, self.indexes[index], isSyn

    def __len__(self):
        return len(self.indexes)