import torch
import time
import torch.nn as nn
import random as r
import numpy as np
import os
import pandas as pd
import wandb
import matplotlib.pyplot as plt
import kornia as K
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, confusion_matrix

from src.learners.baseline.base import BaseLearner
from src.utils.losses import SupConLoss
from src.buffers.reservoir import Reservoir
from src.models.resnet import ResNet18, ImageNet_ResNet18, ChunkNorm_ResNet18
from src.utils.metrics import forgetting_line   
from src.utils.utils import get_device
from src.utils.losses import SupConLoss
from src.buffers.syn_res import SynRes

device = get_device()

def kl_loss(logits_stu, logits_tea, temperature=4.0):
    """
    Args:
        logits_stu: student logits
        logits_tea: teacher logits
        temperature: temperature
    Returns:
        distillation loss
    """
    pred_teacher = F.softmax(logits_tea / temperature, dim=1)
    log_pred_student = F.log_softmax(logits_stu / temperature, dim=1)
    loss_kd = F.kl_div(
        log_pred_student,
        pred_teacher,
        reduction='none'
    ).sum(1).mean(0) * (temperature ** 2)
    return loss_kd
    
class ERLearner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self.buffer = SynRes(
            max_size=self.params.mem_size,
            img_size=self.params.img_size,
            nb_ch=self.params.nb_channels,
            n_classes=self.params.n_classes,
            drop_method=self.params.drop_method,
        )
        self.iter = 0
    
    def load_criterion(self):
        return nn.CrossEntropyLoss()
    
    def load_model(self, **kwargs):
        if self.params.norm_layer == 'cn' or self.params.norm_layer == 'cn+real':
            return ChunkNorm_ResNet18(
                dim_in=self.params.dim_in,
                nclasses=self.params.n_classes,
                nf=self.params.nf
            ).to(device)
        if self.params.dataset == 'cifar10' or self.params.dataset == 'cifar100' or self.params.dataset == 'tiny' or self.params.dataset == 'mixed_cifar100' or self.params.dataset == 'mixed_tiny' or self.params.dataset == 'mixed_cifar10' or self.params.dataset == 'concat_cifar100':
            return ResNet18(
                dim_in=self.params.dim_in,
                nclasses=self.params.n_classes,
                nf=self.params.nf,
                normalization = self.params.norm_layer,
                num_groups = self.params.num_groups
            ).to(device)
        elif self.params.dataset == 'imagenet' or self.params.dataset == 'imagenet100' or self.params.dataset == 'mixed_in100':
            return ImageNet_ResNet18(
                dim_in=self.params.dim_in,
                nclasses=self.params.n_classes,
                nf=self.params.nf
            ).to(device)

    
    def train(self, dataloader, **kwargs):
        task_name  = kwargs.get('task_name', 'unknown task')
        task_id    = kwargs.get('task_id', 0)
        dataloaders = kwargs.get('dataloaders', None)
        self.model = self.model.train()
        supcon = SupConLoss(temperature=0.07)

        for j, batch in enumerate(dataloader):
            # Stream data
            batch_x, batch_y = batch[0], batch[1]
            if self.params.dataset == 'mixed_cifar100' or self.params.dataset == 'mixed_tiny' or self.params.dataset == 'mixed_cifar10' or self.params.dataset == 'concat_cifar100' or self.params.dataset == 'mixed_in100':
                isSyn = batch[3]
            else:
                isSyn = torch.tensor([False, False, False, False, False, False, False, False, False, False])
            self.stream_idx += len(batch_x)
            
            for _ in range(self.params.mem_iters):
                mem_x, mem_y, mem_isSyn = self.buffer.random_retrieve(n_imgs=self.params.mem_batch_size)

                if mem_x.size(0) > 0:
                    # Combined batch
                    combined_x, combined_y = self.combine(batch_x, batch_y, mem_x, mem_y)  # (batch_size, nb_channel, img_size, img_size)
                    combined_x = combined_x.to(device)
                    combined_y = combined_y.to(device)
                    # Augment
                    combined_aug = self.transform_train(combined_x)

                    # Inference
                    feature_aug, logits_aug = self.model.full(combined_aug)

                    # Loss
                    loss = self.criterion(logits_aug, combined_y.long())
                    self.loss = loss.item()
                    print(f"Loss {self.loss:.3f}  batch {j}", end="\r")
                    self.optim.zero_grad()
                    loss.backward()
                    self.optim.step()

                    self.iter += 1
            
            # Update reservoir buffer

            if batch_x.size(0) > 0:
                self.buffer.update(imgs=batch_x, labels=batch_y, isSyn=isSyn)

            if (j == (len(dataloader) - 1)) and (j > 0):
                print(
                    f"Task : {task_name}   batch {j}/{len(dataloader)}   Loss : {loss.item():.4f}    time : {time.time() - self.start:.4f}s"
                )
    
    def get_mem_rep_labels_syn(self, eval=True, use_proj=False):
        """Compute every representation -labels pairs from memory
        Args:
            eval (bool, optional): Whether to turn the mdoel in evaluation mode. Defaults to True.
        Returns:
            representation - labels pairs
        """
        # if eval: 
            # self.model.eval()
        self.model.eval()
        mem_imgs, mem_labels, mem_isSyn = self.buffer.get_all_syn()
        batch_s = 10
        n_batch = len(mem_imgs) // batch_s
        all_reps = []
        for i in range(n_batch):
            mem_imgs_b = mem_imgs[i*batch_s:(i+1)*batch_s].to(self.device)
            mem_imgs_b = self.transform_test(mem_imgs_b)
            mem_representations_b = self.model(mem_imgs_b)
            all_reps.append(mem_representations_b)
        mem_representations= torch.cat(all_reps, dim=0)
        return mem_representations, mem_labels, mem_isSyn

    def encode_fea_syn(self, dataloader, nbatches=-1):
        i = 0
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            for sample in dataloader:
                if nbatches != -1 and i >= nbatches:
                    break
                inputs = sample[0]
                labels = sample[1]
                isSyn = sample[3]
                
                inputs = inputs.to(device)

                feat = self.model(self.transform_test(inputs))

                if i == 0:
                    all_labels = labels.cpu().numpy()
                    all_feat= feat.cpu().numpy()
                    all_isSyn = isSyn
                    i += 1
                else:
                    all_labels = np.hstack([all_labels, labels.cpu().numpy()])
                    all_feat= np.vstack([all_feat, feat.cpu().numpy()])
                    all_isSyn = np.hstack([all_isSyn, isSyn])

        return all_feat, all_labels, all_isSyn

    def print_results(self, task_id):
        n_dashes = 20
        pad_size = 8
        print('-' * n_dashes + f"TASK {task_id + 1} / {self.params.n_tasks}" + '-' * n_dashes)
        
        print('-' * n_dashes + "ACCURACY" + '-' * n_dashes)        
        for line in self.results:
            print('Acc'.ljust(pad_size) + ' '.join(f'{value:.4f}'.ljust(pad_size) for value in line), f"{np.nanmean(line):.4f}")
    
    def combine(self, batch_x, batch_y, mem_x, mem_y):
        # mem_x, mem_y = mem_x.to(self.device), mem_y.to(self.device)
        # batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
        combined_x = torch.cat([mem_x, batch_x])
        combined_y = torch.cat([mem_y, batch_y])
        if self.params.memory_only:
            return mem_x, mem_y
        else:
            return combined_x, combined_y
        