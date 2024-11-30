import os
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder

# ImageNet100
class SYNC_ImageNet100(ImageFolder):
    def __init__(self, baseroot, sync_version=['xl'], transform=None):
        root = os.path.join(baseroot, sync_version)
        super().__init__(root, transform)
        self.transform_in = transforms.Compose([
            transforms.ConvertImageDtype(torch.float32),
            transforms.Resize(size=(224,224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229,0.224,0.225]),
        ])
    
    def __getitem__(self, idx):
        img, label = super().__getitem__(idx)
        img = self.transform_in(img)
        label = torch.tensor(label)
        return img, label
        
