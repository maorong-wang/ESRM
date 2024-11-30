from __future__ import print_function
from PIL import Image
from torchvision import transforms
import torchvision.datasets as datasets
import torch.utils.data as data
import torch
import numpy as np
import os

class SYNC_CIFAR10(torch.utils.data.Dataset):
    def __init__(self, baseroot, sync_version='v1.4', transform=None):
        self.transform = transform
        self.transform_in = transforms.Compose([
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float32),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229,0.224,0.aaaaoaaaad225]), # use imagenet normalization
        ])
        self.baseroot = baseroot
        self.data = []
        self.targets = []
        self.path = []

        if sync_version == 'xl_test':
            for class_idx in range(10):
                for image_idx in range(1000):
                    path = self.get_path(sync_version, class_idx, image_idx)
                    # img_p = Image.open(path)
                    # img = np.array(img_p)
                    self.path.append(path)
                    self.targets.append(class_idx)
                    # img_p.close()
        else:
            for class_idx in range(10):
                for image_idx in range(5000):
                    path = self.get_path(sync_version, class_idx, image_idx)
                    # img_p = Image.open(path)
                    # img = np.array(img_p)
                    self.path.append(path)
                    self.targets.append(class_idx)
                    # img_p.close()

    def get_path(self, version, class_idx, image_idx):
        return os.path.join(self.baseroot, f"{version}/{class_idx}/{image_idx}.png")

    def __len__(self):
        return len(self.path)
        
    def __getitem__(self, index):
        img_p = Image.open(self.path[index])
        img = np.array(img_p)
        img_p.close()
        img = self.transform_in(img)
        label = torch.tensor(self.targets[index])
        return img, label, index

if __name__ == '__main__':
    dataset = SYNC_CIFAR10(sync_version='v1.4', transform=None)
    import pdb
    pdb.set_trace()