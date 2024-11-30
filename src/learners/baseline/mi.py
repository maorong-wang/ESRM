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

from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, confusion_matrix

from src.learners.baseline.base import BaseLearner
from src.utils.losses import SupConLoss
from src.buffers.reservoir import Reservoir
from src.models.resnet import ResNet18, ImageNet_ResNet18, ChunkNorm_ResNet18, OCMResnet, ImageNet_OCMResnet
from src.utils.metrics import forgetting_line   
from src.utils.utils import get_device
from src.utils.losses import SupConLoss
from src.buffers.syn_res import SynRes


device = get_device()
    

class MILearner(BaseLearner):
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
    
    def infoNCE(self, x1, x2, label1, label2, tau=0.07):
        x1 = torch.nn.functional.normalize(x1, dim=1)
        x2 = torch.nn.functional.normalize(x2, dim=1)
        N1 = x1.size(0)
        N2 = x2.size(0)
        sim = torch.mm(x1, x2.t()) # batch_size x batch_size
        sim_max = torch.max(sim, dim=1, keepdim=True)[0]
        sim = torch.exp(sim / tau - sim_max.detach())
        denom = torch.sum(sim, dim=1, keepdim=True)
        sim = torch.log(sim / denom)
        label1 = label1.view(N1, 1)
        label2 = label2.view(N2, 1)
        mask = label1 == label2.t()
        sim = sim * mask.float()
        return -sim.mean()

    
    def load_model(self, **kwargs):
        if self.params.dataset == 'cifar10' or self.params.dataset == 'cifar100' or self.params.dataset == 'tiny' or self.params.dataset == 'mixed_cifar100':
            return OCMResnet(
                head='mlp',
                dim_in=self.params.dim_in,
                proj_dim=self.params.proj_dim,
                n_classes=self.params.n_classes
            ).to(device)
        elif self.params.dataset == 'imagenet' or self.params.dataset == 'imagenet100':
            # for imagenet experiments, the 80 gig memory is not enough, so do it in a data parallel way
            model = ImageNet_OCMResnet(
                head='mlp',
                dim_in=self.params.dim_in,
                proj_dim=self.params.proj_dim,
                n_classes=self.params.n_classes
            )
            return model.to(device)

    
    def train(self, dataloader, **kwargs):
        task_name  = kwargs.get('task_name', 'unknown task')
        task_id    = kwargs.get('task_id', 0)
        dataloaders = kwargs.get('dataloaders', None)
        self.model = self.model.train()
        supcon = SupConLoss(temperature=0.07)

        for j, batch in enumerate(dataloader):
            # Stream data
            batch_x, batch_y = batch[0], batch[1]
            if self.params.dataset == 'mixed_cifar100' or self.params.dataset == 'mixed_tiny':
                isSyn = batch[3]
            else:
                isSyn = torch.tensor([False, False, False, False, False, False, False, False, False, False])
            self.stream_idx += len(batch_x)
            
            for _ in range(self.params.mem_iters):
                mem_x, mem_y, mem_isSyn = self.buffer.random_retrieve(n_imgs=self.params.mem_batch_size)

                if mem_x.size(0) > 0:
                    # Combined batch
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    mem_x, mem_y = mem_x.to(device), mem_y.to(device)
                    combined_x, combined_y = self.combine(batch_x, batch_y, mem_x, mem_y)  # (batch_size, nb_channel, img_size, img_size)
                    # combined_x = combined_x.to(device)
                    # combined_y = combined_y.to(device)
                    # Augment
                    combined_aug = self.transform_train(combined_x)

                    # Inference
                    if self.params.norm_layer == 'cn':
                        with torch.no_grad():
                            self.model.eval()
                            combined_x = combined_x.to(device)
                            combined_x = self.transform_test(combined_x)
                            feat = self.model.logits(combined_x)
                            feat = torch.nn.functional.softmax(feat, dim=1)
                            ent = -torch.sum(feat * torch.log(feat), dim=1)
                            fifty_percentail = torch.quantile(ent, 0.5)
                            chunk_mask = torch.tensor(ent < fifty_percentail, dtype=torch.int).to(device)
                            self.model.train()
                        _, logits_aug = self.model.full(combined_aug, chunk_mask)
                    elif self.params.norm_layer == 'cn+real':
                        combined_isSyn = torch.cat([isSyn, mem_isSyn])
                        _, logits_aug = self.model.full(combined_aug, combined_isSyn)
                    else:
                        _, logits_aug = self.model.full(combined_aug)
                        _, feature_aug = self.model(batch_x, is_simclr=True)
                        _, feature_mem = self.model(mem_x, is_simclr=True)


                    with torch.no_grad():
                        self.model.eval()
                        batch_x = batch_x.to(device)
                        batch_x = self.transform_test(batch_x)
                        feat = self.model.logits(batch_x)
                        feat = torch.nn.functional.softmax(feat, dim=1)
                        ent = -torch.sum(feat * torch.log(feat), dim=1)
                        percentail = torch.quantile(ent, 0.5)
                        self.model.train()
                    # Loss
                    loss = self.criterion(logits_aug, combined_y.long()) \
                        + self.infoNCE(feature_aug[ent >= percentail], feature_aug[ent < percentail], batch_y[ent >= percentail], batch_y[ent < percentail]) \
                        + self.infoNCE(feature_aug, feature_mem, batch_y, mem_y)
                    self.loss = loss.item()
                    print(f"Loss {self.loss:.3f}  batch {j}", end="\r")
                    self.optim.zero_grad()
                    loss.backward()
                    self.optim.step()

                    self.iter += 1
            
            # Update reservoir buffer
            if self.params.dataset == 'mixed_cifar100' or self.params.dataset == 'mixed_tiny':
                if self.params.mem_strat == 'real':
                    batch_x = batch_x[isSyn==0]
                    batch_y = batch_y[isSyn==0]
                    isSyn = isSyn[isSyn==0]
                elif self.params.mem_strat == 'syn':
                    batch_x = batch_x[isSyn==1]
                    batch_y = batch_y[isSyn==1]
                    isSyn = isSyn[isSyn==1]
                elif self.params.mem_strat == 'entropy':
                    with torch.no_grad():
                        self.model.eval()
                        batch_x = batch_x.to(device)
                        batch_x = self.transform_test(batch_x)
                        feat = self.model.logits(batch_x)
                        feat = torch.nn.functional.softmax(feat, dim=1)
                        ent = -torch.sum(feat * torch.log(feat), dim=1)
                        percentail = torch.quantile(ent, 0.5)
                        batch_x = batch_x[ent > percentail]
                        batch_y = batch_y[(ent > percentail).cpu()]
                        isSyn = isSyn[(ent > percentail).cpu()]
                        self.model.train()
                elif self.params.mem_strat == 'entropy.89':
                    with torch.no_grad():
                        self.model.eval()
                        batch_x = batch_x.to(device)
                        batch_x = self.transform_test(batch_x)
                        feat = self.model.logits(batch_x)
                        feat = torch.nn.functional.softmax(feat, dim=1)
                        ent = -torch.sum(feat * torch.log(feat), dim=1)
                        percentail = torch.quantile(ent, 0.89)
                        batch_x = batch_x[ent > percentail]
                        batch_y = batch_y[(ent > percentail).cpu()]
                        isSyn = isSyn[(ent > percentail).cpu()]
                        self.model.train()

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
        