from __future__ import print_function
from PIL import Image
import torchvision.datasets as datasets
import torch.utils.data as data
import torch


class CIFAR100(datasets.cifar.CIFAR100):
    """CIFAR100 Instance Dataset.
    """
    def __init__(self, data_root_dir, train, download, transform):
        super().__init__(data_root_dir, train=train, download=download, transform=transform)
        self.targets = torch.Tensor(self.targets)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        # if self.train:
        img, target = self.data[index], self.targets[index]
        # else:
            # img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index


if __name__ == '__main__':
    dataset = CIFAR100(data_root_dir='data', train=True, download=True, transform=None)
    import pdb
    pdb.set_trace()