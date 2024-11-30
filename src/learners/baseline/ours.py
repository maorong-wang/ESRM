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
import torchvision.transforms.v2 as v2
from copy import deepcopy

from scipy.linalg import svd
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.distributions import Categorical

from src.learners.baseline.base import BaseLearner
from src.utils.losses import SupConLoss
from src.buffers.reservoir import Reservoir
from src.models.resnet import ResNet18, ImageNet_ResNet18, ChunkNorm_ResNet18, OCMResnet, ImageNet_OCMResnet
from src.utils.metrics import forgetting_line   
from src.utils.utils import get_device
from src.utils.losses import SupConLoss
from src.buffers.syn_res import SynRes
from src.buffers.ent_res import EntRes


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

class OursLearner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self.buffer = EntRes(
            max_size=self.params.mem_size,
            img_size=self.params.img_size,
            nb_ch=self.params.nb_channels,
            n_classes=self.params.n_classes,
            drop_method=self.params.drop_method,
        )
        self.iter = 0
        self.syn_results = []
        self.syn_results_clustering = []
        self.syn_results_forgetting = []
        self.syn_results_clustering_forgetting = []

    def load_criterion(self):
        return nn.CrossEntropyLoss()
    
    def infoNCE(self, x1, x2, label1, label2, tau=0.07):
        x1 = torch.nn.functional.normalize(x1, dim=1)
        x2 = torch.nn.functional.normalize(x2, dim=1)
        N1 = x1.size(0)
        N2 = x2.size(0)
        if N1 == 0 or N2 == 0:
            return 0
        sim = torch.mm(x1, x2.t()) # batch_size x batch_size
        sim = sim / tau
        sim_max = torch.max(sim, dim=1, keepdim=True)[0]
        sim = torch.exp(sim - sim_max.detach())
        denom = torch.sum(sim, dim=1, keepdim=True)
        sim = torch.log(sim / denom)
        label1 = label1.view(N1, 1)
        label2 = label2.view(N2, 1)
        mask = label1 == label2.t()
        sim = sim * mask.float()
        return -sim.mean()

    
    def load_model(self, **kwargs):
        if 'cifar10' in self.params.dataset or 'tiny' in self.params.dataset:
            return OCMResnet(
                head='mlp',
                dim_in=self.params.dim_in,
                proj_dim=self.params.proj_dim,
                n_classes=self.params.n_classes
            ).to(device)
        elif self.params.dataset == 'imagenet' or self.params.dataset == 'imagenet100' or self.params.dataset == 'mixed_in100':
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
            if self.params.dataset == 'mixed_cifar100' or self.params.dataset == 'mixed_tiny' or self.params.dataset == 'mixed_cifar10' or self.params.dataset == 'concat_cifar100' or self.params.dataset == 'mixed_in100':
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
                    _, logits = self.model.full(combined_x)
                    _, logits_aug = self.model.full(combined_aug)
                    _, feature_aug = self.model(batch_x, is_simclr=True)
                    _, feature_mem = self.model(mem_x, is_simclr=True)


                    with torch.no_grad():
                        # by entropy
                        self.model.eval()
                        batch_x = batch_x.to(device)
                        batch_x = self.transform_test(batch_x)
                        feat = self.model.logits(batch_x)
                        feat = torch.nn.functional.softmax(feat, dim=1)
                        ent = -torch.sum(feat * torch.log(feat), dim=1)
                        u_th = torch.quantile(ent, self.params.entropy_percentail_upper) # percentile upper is set to one so u_th will not filter any data
                        l_th = torch.quantile(ent, self.params.entropy_percentail_lower)
                        real = (ent <= u_th) * (ent > l_th)
                        fake = ~real
                        self.model.train()
                    

                    # Loss

                    loss = self.criterion(logits_aug, combined_y.long()) + self.criterion(logits, combined_y.long()) + kl_loss(logits, logits_aug, temperature=4.0) \
                        + self.params.kd_lambda * (self.infoNCE(feature_aug[real], feature_aug[fake], batch_y[real], batch_y[fake]) + self.infoNCE(feature_aug[fake], feature_aug[real], batch_y[fake], batch_y[real])\
                        + self.infoNCE(feature_aug, feature_mem, batch_y, mem_y) + self.infoNCE(feature_mem, feature_aug, mem_y, batch_y))

                    self.loss = loss.item()
                    print(f"Loss {self.loss:.3f}  batch {j}", end="\r")
                    self.optim.zero_grad()
                    loss.backward()
                    self.optim.step()

                    self.iter += 1
            
            # Update reservoir buffer
            if self.params.mem_strat == 'real':
                real = ~isSyn
                fake = isSyn
            elif self.params.mem_strat == 'fake':
                real = isSyn
                fake = ~isSyn
            elif self.params.mem_strat == 'random':
                real = torch.ones_like(isSyn).bool()
            if self.params.dataset == 'mixed_cifar100' or self.params.dataset == 'mixed_tiny' or self.params.dataset == 'mixed_in100' or self.params.dataset == 'mixed_cifar10':
                if mem_x.size(0) > 0:
                    batch_x = batch_x[real]
                    batch_y = batch_y[(real).cpu()]
                    isSyn = isSyn[(real).cpu()]

            if batch_x.size(0) > 0:
                with torch.no_grad():
                    self.model.eval()
                    logits = self.model.logits(self.transform_test(batch_x.to(device)))
                    feat = torch.nn.functional.softmax(logits, dim=1)
                    ent = -torch.sum(feat * torch.log(feat), dim=1).cpu()
                    self.model.train()
                self.buffer.update(imgs=batch_x, labels=batch_y, isSyn=isSyn, ent=ent)
                _, _, all_isSyn = self.buffer.get_all_syn()
                wandb.log({"Synthetic": all_isSyn.sum() / len(all_isSyn)})

            if (j == (len(dataloader) - 1)) and (j > 0):
                print(
                    f"Task : {task_name}   batch {j}/{len(dataloader)}   Loss : {loss.item():.4f}    time : {time.time() - self.start:.4f}s"
                )
        
        self.update_ent()
    
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
    
    def evaluate(self, dataloaders, task_id, eval_ema=False, **kwargs):
        self.model.eval()
        with torch.no_grad():
            num_data = 0
            acc_entropy = 0
            real_entropy = []
            syn_entropy = []
            syn_index = []
            with torch.no_grad():
                for i, sample in enumerate(dataloaders[f'train{task_id}']):
                    data = sample[0]
                    labels = sample[1]
                    try:
                        isSyn = sample[3]
                        imgidx = sample[2]
                    except:
                        isSyn = torch.zeros_like(labels)
                    
                    data = data.to(device)
                    outputs = self.model.logits(data)
                    prob = torch.softmax(outputs, dim=1)
                    entropy = Categorical(probs = prob).entropy()
                    acc_entropy += entropy.sum()

                    num_data += labels.size(0)
                    real_entropy += entropy[isSyn == 0].tolist()
                    syn_entropy += entropy[isSyn == 1].tolist()
                
            acc_entropy /= num_data

            fig = plt.figure()
            plt.hist(real_entropy, bins=100, alpha=0.5, label='real')
            plt.hist(syn_entropy, bins=100, alpha=0.5, label='syn')
            plt.legend()
            wandb.log({f"Entropy_dist_train_{task_id}": wandb.Image(fig)})
        
        with torch.no_grad():
            ret_imgs, ret_labels, ret_isSyn = self.buffer.get_all_syn()
            ret_imgs = self.transform_test(ret_imgs).to(device)
            batch_size = 100
            n_batches = len(ret_imgs) // batch_size
            syn_entropy = []
            real_entropy = []
            for i in range(n_batches):
                batch_imgs = ret_imgs[i*batch_size:(i+1)*batch_size]
                logits = self.model.logits(batch_imgs)
                prob = torch.softmax(logits, dim=1)
                entropy = Categorical(probs = prob).entropy()
                acc_entropy += entropy.sum()

                num_data += len(batch_imgs)
                real_entropy += entropy[ret_isSyn[i*batch_size:(i+1)*batch_size] == 0].tolist()
                syn_entropy += entropy[ret_isSyn[i*batch_size:(i+1)*batch_size] == 1].tolist()
        
        fig = plt.figure()
        plt.hist(real_entropy, bins=100, alpha=0.5, label='real')
        plt.hist(syn_entropy, bins=100, alpha=0.5, label='syn')
        plt.legend()
        wandb.log({f"Entropy_dist_buffer_whole_{task_id}": wandb.Image(fig)})

        stepsize = int(self.params.n_classes / self.params.n_tasks)
        current_classes = torch.tensor(self.params.labels_order[task_id * stepsize: (task_id + 1) * stepsize], dtype=torch.long)
        with torch.no_grad():
            ret_imgs, ret_labels, ret_isSyn = self.buffer.get_all_syn()
            mask = torch.zeros_like(ret_labels, dtype=torch.bool)
            for c in current_classes:
                mask = mask | (ret_labels == c)
            ret_imgs = ret_imgs[mask]
            ret_labels = ret_labels[mask]
            ret_isSyn = ret_isSyn[mask]
            ret_imgs = self.transform_test(ret_imgs).to(device)
            batch_size = 100
            n_batches = len(ret_imgs) // batch_size
            syn_entropy = []
            real_entropy = []
            for i in range(n_batches):
                batch_imgs = ret_imgs[i*batch_size:(i+1)*batch_size]
                logits = self.model.logits(batch_imgs)
                prob = torch.softmax(logits, dim=1)
                entropy = Categorical(probs = prob).entropy()
                acc_entropy += entropy.sum()

                num_data += len(batch_imgs)
                real_entropy += entropy[ret_isSyn[i*batch_size:(i+1)*batch_size] == 0].tolist()
                syn_entropy += entropy[ret_isSyn[i*batch_size:(i+1)*batch_size] == 1].tolist()
        
        fig = plt.figure()
        plt.hist(real_entropy, bins=100, alpha=0.5, label='real')
        plt.hist(syn_entropy, bins=100, alpha=0.5, label='syn')
        plt.legend()
        wandb.log({f"Entropy_dist_buffer_task_{task_id}": wandb.Image(fig)})


        return self._evaluate(dataloaders, task_id, eval_ema, **kwargs)

    def update_ent(self):
        with torch.no_grad():
            ret_imgs, ret_idx = self.buffer.get_all_idx()
            batch_size = 100
            n_batches = len(ret_imgs) // batch_size
            for i in range(n_batches):
                batch_imgs = ret_imgs[i*batch_size:(i+1)*batch_size]
                batch_imgs = self.transform_test(batch_imgs).to(device)
                logits = self.model.logits(batch_imgs)
                prob = torch.softmax(logits, dim=1)
                ent = -torch.sum(prob * torch.log(prob), dim=1)
                self.buffer.update_ent(ret_idx[i*batch_size:(i+1)*batch_size], ent)