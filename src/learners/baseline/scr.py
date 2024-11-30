import torch
import time
import torch.nn as nn
import sys
import logging as lg
import pandas as pd

from copy import deepcopy

from torch.utils.data import DataLoader

from src.learners.baseline.base import BaseLearner
from src.utils.losses import SupConLoss
from src.buffers.reservoir import Reservoir
from src.utils import name_match
from src.utils.augment import MixUpAugment, CutMixAugment
from src.models.resnet import OCMResnet, ImageNet_OCMResnet
from src.utils.utils import filter_labels
from src.utils.utils import get_device
from src.models.resnet import ResNet18, ImageNet_ResNet18
from src.buffers.syn_res import SynRes

device = get_device()

class SCRLearner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        # In SCR they use the images from memory for evaluation
        if self.params.training_type != 'uni':
            self.params.eval_mem = True
        self.params.supervised = True
        self.buffer = SynRes(
            max_size=self.params.mem_size,
            img_size=self.params.img_size,
            nb_ch=self.params.nb_channels,
            n_classes=self.params.n_classes,
            drop_method=self.params.drop_method,
        )
        self.tf_seq = {}
        for i in range(2):
            self.tf_seq[f"aug{i}"] = self.transform_train.to(device)
        # if self.params.n_mixup >= 1:
        #     self.tf_seq["mixup"] = MixUpAugment(
        #             min_mix=self.params.min_mix,
        #             max_mix=self.params.max_mix
        #             )
        # if self.params.n_cutmix >= 1:
        #     self.tf_seq["cutmix"] = CutMixAugment(
        #         min_mix=self.params.min_mix,
        #         max_mix=self.params.max_mix
        #         )


    def load_model(self, **kwargs):
        return OCMResnet(
            head='mlp',
            dim_in=self.params.dim_in,
            dim_int=self.params.dim_in,
            proj_dim=self.params.proj_dim,
            n_classes=self.params.n_classes
        ).to(device)


    def load_criterion(self):
        return SupConLoss(self.params.temperature)
    
    def train(self, dataloader, **kwargs):
        self.model = self.model.train()
        if self.params.training_type == 'inc' or self.params.training_type == "blurry":
            self.train_inc(dataloader, **kwargs)
        elif self.params.training_type == "uni":
            self.train_uni(dataloader, **kwargs)

    def train_uni(self, dataloader, **kwargs):
        raise NotImplementedError("Uni training not implemented")

    def train_inc(self, dataloader, **kwargs):
        self.model = self.model.train()
        task_name = kwargs.get('task_name', None)
        task_id = kwargs.get('task_id', None)
        dataloaders = kwargs.get("dataloaders")

        for j, batch in enumerate(dataloader):
            # Stream data
            batch_x, batch_y = batch[0], batch[1]
            if self.params.dataset == 'mixed_cifar100':
                isSyn = batch[3]
            else:
                isSyn = [False, False, False, False, False, False, False, False, False, False] 

            self.stream_idx += len(batch_x)
            
            for _ in range(self.params.mem_iters):
                mem_x, mem_y, *_ = self.buffer.random_retrieve(n_imgs=self.params.mem_batch_size)

                if mem_x.size(0) > 0:
                    # Combined batch
                    combined_x, combined_y = self.combine(batch_x, batch_y, mem_x, mem_y)  # (batch_size, nb_channel, img_size, img_size)

                    # Augment
                    augmentations = self.augment(combined_x=combined_x, mem_x=mem_x.to(device), batch_x=batch_x.to(device))

                    self.model.train()
                    projs = []
                    loss_cls = 0
                    for a in augmentations:
                        _, p = self.model(a, is_simclr=True)
                        projs.append(p.unsqueeze(1))
                        logits = self.model.logits(a)
                        loss_cls += nn.CrossEntropyLoss()(logits, combined_y.long())

                    projections = torch.cat(projs, dim=1)

                    # Loss
                    loss = self.criterion(features=projections, labels=combined_y)
                    loss = loss.mean() + loss_cls

                    self.loss = loss.item()
                    print(f"Loss {self.loss:.3f}  batch {j}", end="\r")
                    self.optim.zero_grad()
                    loss.backward()
                    self.optim.step()
                    
            # Update buffer
            self.buffer.update(imgs=batch_x, labels=batch_y, isSyn=isSyn)

            # Plot to tensorboard
            if self.params.tensorboard:
                self.plot()
            if (j == (len(dataloader) - 1)) and (j > 0):
                lg.info(
                    f"Phase : {task_name}   batch {j}/{len(dataloader)}   Loss : {self.loss:.4f}    time : {time.time() - self.start:.4f}s"
                )
    
    def plot(self):
        self.writer.add_scalar("loss", self.loss, self.stream_idx)
        self.writer.add_scalar("n_added_so_far", self.buffer.n_added_so_far, self.stream_idx)
        percs = self.buffer.get_labels_distribution()
        for i in range(self.params.n_classes):
            self.writer.add_scalar(f"labels_distribution/{i}", percs[i], self.stream_idx)

    def combine(self, batch_x, batch_y, mem_x, mem_y):
        mem_x, mem_y = mem_x.to(self.device), mem_y.to(self.device)
        batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
        combined_x = torch.cat([mem_x, batch_x])
        combined_y = torch.cat([mem_y, batch_y])
        if self.params.memory_only:
            return mem_x, mem_y
        else:
            return combined_x, combined_y

    def get_n_repeat(self, augmentation):
        if augmentation == 'style':
            return self.params.n_styles
        if augmentation == 'cutmix':
            return self.params.n_cutmix
        if augmentation == 'mixup':
            return self.params.n_mixup
        return 1
    
    def augment(self, combined_x, mem_x, batch_x, **kwargs):
        with torch.no_grad():
            augmentations = []
            for key in self.tf_seq:
                if 'aug' in key:
                    augmentations.append(self.tf_seq[key](combined_x))
                else:
                    n_repeat = self.get_n_repeat(key)
                    batch1 = combined_x
                    batch2 = batch_x if self.params.daa else combined_x
                    for _ in range(n_repeat):
                        augmentations.append(self.tf_seq[key](batch1, batch2, model=self.model))
            augmentations.append(combined_x)
            return augmentations

    def after_eval(self, **kwargs):
        if self.params.review:
            self.model = deepcopy(self.model_backup)
            self.optim = self.load_optim()