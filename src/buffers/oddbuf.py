from copyreg import pickle
import torch
import numpy as np
import random as r
import logging as lg

from torch.utils.data import DataLoader, Subset, ConcatDataset
from torchvision import transforms

from src.utils.data import get_color_distortion
from src.utils.utils import timing, get_device
from src.datasets.memory import MemoryDataset

class OddBuffer(torch.nn.Module):
    def __init__(self, max_size=200, shape=(3,32,32), n_classes=10, ipc=None):
        """
        The first part of OddBuffer will behave as usual, storing the incoming data in a reservoir.
        The second part will be storing the geenrated data.
        """
        super().__init__()
        self.n_classes = n_classes
        self.max_size = max_size
        self.shape = shape
        self.n_seen_so_far = 0
        self.n_added_so_far = 0
        self.n_class_seen_so_far = 0
        self.class_seen_so_far = []
        self.classwise_ipc = {}
        self.n_distilled_data_slot = 0
        self.n_distilled_data = 0
        self.n_optimized_data = 0
        self.device = get_device()
        if ipc is None:
            self.ipc = max_size // n_classes
        else:
            self.ipc = ipc
        if self.shape is not None:
            if len(self.shape) == 3:
                self.register_buffer('buffer_imgs', torch.FloatTensor(self.max_size, self.shape[0], self.shape[1], self.shape[2]).fill_(0))
            elif len(self.shape) == 1:
                self.register_buffer('buffer_imgs', torch.FloatTensor(self.max_size, self.shape[0]).fill_(0))
        self.register_buffer('buffer_labels', torch.LongTensor(self.max_size).fill_(-1))

    def distill_data_retrieve(self):         
        ret_imgs = self.buffer_imgs[self.max_size-self.n_distilled_data:self.max_size-self.n_optimized_data]
        ret_labels = self.buffer_labels[self.max_size-self.n_distilled_data:self.max_size-self.n_optimized_data]
        return ret_imgs, ret_labels
    
    def distill_data_write(self, imgs, labels):
        cur = 0
        for data, label in zip(imgs, labels):
            self.n_distilled_data += 1
            cur += 1
            self.write_data((self.max_size-self.n_distilled_data+cur-1), data, label)
        
        assert self.n_distilled_data == self.n_optimized_data
    
    def distill_data_update(self, imgs, labels):
        cur = 0
        for data, label in zip(imgs, labels):
            self.n_optimized_data += 1
            cur += 1
            self.write_data((self.max_size-self.n_distilled_data+cur-1), data, label)
        
        assert self.n_optimized_data == self.n_distilled_data

    def update(self, imgs, labels=None):
        for stream_img, stream_label in zip(imgs, labels):
            if stream_label not in self.class_seen_so_far:
                self.class_seen_so_far.append(stream_label)
                self.n_class_seen_so_far += 1
                self.n_distilled_data_slot += self.ipc
                '''
                add the first image
                '''
                self.classwise_ipc[int(stream_label)] = 1
                self.write_data((self.max_size-self.n_distilled_data-1), stream_img, stream_label)
                self.n_distilled_data += 1
            elif self.classwise_ipc[int(stream_label)] < self.ipc:
                self.classwise_ipc[int(stream_label)] += 1
                self.write_data((self.max_size-self.n_distilled_data-1), stream_img, stream_label)
                self.n_distilled_data += 1
            
            reservoir_idx = int(r.random() * (self.n_seen_so_far + 1))
            if self.n_seen_so_far < (self.max_size - self.n_distilled_data_slot):
                reservoir_idx = self.n_added_so_far
            if reservoir_idx < (self.max_size - self.n_distilled_data_slot):
                self.replace_data(reservoir_idx, stream_img, stream_label)
            self.n_seen_so_far += 1

    def stack_data(self, img, label):
        if self.n_seen_so_far < self.max_size:
            self.buffer_imgs[self.n_seen_so_far] = img
            self.buffer_labels[self.n_seen_so_far] = label
            self.n_added_so_far += 1

    def replace_data(self, idx, img, label):
        self.buffer_imgs[idx] = img
        self.buffer_labels[idx] = label
        self.n_added_so_far += 1
    

    def write_data(self, idx, img, label):
        self.buffer_imgs[idx] = img
        self.buffer_labels[idx] = label

    def is_empty(self):
        return self.n_added_so_far == 0
    
    def random_retrieve(self, n_imgs=100):
        if self.n_added_so_far < n_imgs:
            lg.debug(f"""Cannot retrieve the number of requested images from memory {self.n_added_so_far}/{n_imgs}""")
            return self.buffer_imgs[:self.n_added_so_far], self.buffer_labels[:self.n_added_so_far]
        
        ret_indexes = r.sample(np.arange(min(self.n_added_so_far, self.max_size)).tolist(), n_imgs)
        ret_imgs = self.buffer_imgs[ret_indexes]
        ret_labels = self.buffer_labels[ret_indexes]
        
        return ret_imgs, ret_labels
    
    def only_retrieve(self, n_imgs, desired_labels):
        """Retrieve images belonging only to the set of desired labels

        Args:
            n_imgs (int):                    Number of images to retrieve 
            desired_labels (torch.Tensor): tensor of desired labels to retrieve from
        """
        desired_labels = torch.tensor(desired_labels)
        valid_indexes = torch.isin(self.buffer_labels, desired_labels).nonzero().view(-1)
        n_out = min(n_imgs, len(valid_indexes))
        out_indexes = np.random.choice(valid_indexes, n_out)
        
        return self.buffer_imgs[out_indexes], self.buffer_labels[out_indexes]
    
    def except_retrieve(self, n_imgs, undesired_labels):
        """Retrieve images except images of undesired labels

        Args:
            n_imgs (int):                  Number of images to retrieve 
            desired_labels (torch.Tensor): tensor of desired labels to retrieve from
        """
        undesired_labels = torch.tensor(undesired_labels + [-1])
        valid_indexes = (~torch.isin(self.buffer_labels, undesired_labels)).nonzero().view(-1)
        n_out = min(n_imgs, len(valid_indexes))
        out_indexes = np.random.choice(valid_indexes, n_out)
        
        return self.buffer_imgs[out_indexes], self.buffer_labels[out_indexes]
    
    def dist_retrieve(self, means, model, n_imgs=100):
        if self.n_added_so_far < n_imgs:
            lg.debug(f"""Cannot retrieve the number of requested images from memory {self.n_added_so_far}/{n_imgs}""")
            return self.buffer_imgs[:self.n_added_so_far], self.buffer_labels[:self.n_added_so_far]
        
        # model.eval()
        with torch.no_grad():
            _, p_mem = model(self.buffer_imgs[:self.n_added_so_far].to(self.device))

        m = torch.zeros((p_mem.shape[1], self.n_classes)).to(self.device)
        for c in means:
            m[:, int(float(c))] = means[f'{c}']

        dists = p_mem @ m
        # Get distances from kown classes only
        dists = dists[torch.arange(dists.size(0)), self.buffer_labels[:self.n_added_so_far]]
        sorted_idx = dists.sort(descending=True).indices
        ret_indexes = []
        # ensuring we get some of each class
        for c in self.buffer_labels[:self.n_added_so_far].unique():
            idx = torch.where((self.buffer_labels[:self.n_added_so_far][sorted_idx] == c))[0][:int(n_imgs/len(self.buffer_labels[:self.n_added_so_far].unique()))]
            ret_indexes.append(idx)
        ret_indexes = torch.cat(ret_indexes)
        ret_imgs = self.buffer_imgs[ret_indexes]
        ret_labels = self.buffer_labels[ret_indexes]

        return ret_imgs, ret_labels
    
    def dist_update(self, means, model, imgs, labels, **kwargs):
        # model.eval()
        # with torch.no_grad():
        #     _, p_mem = model(self.buffer_imgs[:self.n_added_so_far].to(self.device))
        
        # m = torch.zeros((p_mem.shape[1], self.n_classes)).to(self.device)
        # for c in means:
        #     m[:, int(float(c))] = means[f'{c}']

        # dists = p_mem @ m
        for stream_data, stream_label in zip(imgs, labels):
            if self.n_added_so_far < self.max_size:
                self.stack_data(stream_data, stream_label)
            else:
                max_img_per_class = self.get_max_img_per_class()
                class_indexes = self.get_indexes_of_class(stream_label)
                # Do nothing if class has reached maximum number of images
                if len(class_indexes) <= max_img_per_class:
                    # Drop img of major class if not
                    major_class = self.get_major_class()
                    class_indexes = self.get_indexes_of_class(major_class)

                    # compute distances to mean
                    model.eval()
                    with torch.no_grad():
                        _, p_mem = model(self.buffer_imgs[class_indexes.squeeze()].to(self.device))
                    
                    m = means[f'{major_class}.0'].to(self.device)

                    dists = p_mem @ m
                    # idx = class_indexes.squeeze()[dists.argmax()]
                    idx = class_indexes.squeeze()[dists.argmin()]
                    self.replace_data(idx, stream_data, stream_label)
            self.n_seen_so_far += 1
    
    def bootstrap_retrieve(self, n_imgs=100):
        if self.n_added_so_far == 0:
            return torch.Tensor(), torch.Tensor() 
        ret_indexes = [r.randint(0, min(self.n_added_so_far, self.max_size)-1) for _ in range(n_imgs)]            
        ret_imgs = self.buffer_imgs[ret_indexes]
        ret_labels = self.buffer_labels[ret_indexes]

        return ret_imgs, ret_labels
        
    def n_data(self):
        return len(self.buffer_labels[self.buffer_labels >= 0])

    def get_all(self):
        return self.buffer_imgs[:min(self.n_added_so_far, self.max_size)],\
             self.buffer_labels[:min(self.n_added_so_far, self.max_size)]

    def get_indexes_of_class(self, label):
        return torch.nonzero(self.buffer_labels == label)
    
    def get_indexes_out_of_class(self, label):
        return torch.nonzero(self.buffer_labels != label)

    def is_full(self):
        return self.n_data() == self.max_size

    def get_labels_distribution(self):
        np_labels = self.buffer_labels.numpy().astype(int)
        counts = np.bincount(np_labels[self.buffer_labels >= 0], minlength=self.n_classes)
        tot_labels = len(self.buffer_labels[self.buffer_labels >= 0])
        if tot_labels > 0:
            return counts / len(self.buffer_labels[self.buffer_labels >= 0])
        else:
            return counts

    def get_major_class(self):
        np_labels = self.buffer_labels.numpy().astype(int)
        counts = np.bincount(np_labels[self.buffer_labels >= 0])
        return counts.argmax()

    def get_max_img_per_class(self):
        n_classes_in_memory = len(self.buffer_labels.unique())
        return int(len(self.buffer_labels[self.buffer_labels >= 0]) / n_classes_in_memory)

