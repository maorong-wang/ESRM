import torch
import random as r
import numpy as np

from src.buffers.buffer import Buffer
from src.utils.utils import get_device

device = get_device()


class EntRes(Buffer):
    def __init__(self, max_size=200, img_size=32, nb_ch=3, n_classes=10, shape=None, **kwargs):
        if shape is not None:
            super().__init__(max_size, shape=shape, n_classes=n_classes)
        else:
            super().__init__(max_size, shape=(nb_ch, img_size, img_size), n_classes=n_classes)
        self.img_size = img_size
        self.nb_ch = nb_ch
        self.drop_method = kwargs.get('drop_method', 'random')
        self.register_buffer('buffer_syn', torch.IntTensor(self.max_size).fill_(0))
        self.register_buffer('buffer_ent', torch.FloatTensor(self.max_size).fill_(0))

    def update(self, imgs, labels, isSyn, ent, **kwargs):
        """Update buffer with the given list of images and labels.
            Note that labels are not used update selection, only when storing the image in memory.
        Args:
            imgs (torch.tensor): stream images seen by the buffer
            labels (torch.tensor): stream labels seen by the buffer
            logits (torch.tensor): stream logits seen by the buffer
        """
        for stream_img, stream_label, stream_isSyn, stream_ent in zip(imgs, labels, isSyn, ent):
            reservoir_idx = int(r.random() * (self.n_seen_so_far + 1))
            if self.n_seen_so_far < self.max_size:
                # if the buffer is not full, add the new data at the end
                reservoir_idx = self.n_added_so_far 
            if reservoir_idx < self.max_size:
                if self.n_seen_so_far >= self.max_size:
                    # if the buffer is full, drop the data of class 'label' with smallest entropy
                    label = self.buffer_labels[reservoir_idx]
                    if (self.buffer_labels==label).sum() > 1:
                        reservoir_idx = (self.buffer_labels == label).nonzero().squeeze()[self.buffer_ent[self.buffer_labels == label].argmin()]
                self.replace_data(reservoir_idx, stream_img, stream_label, stream_isSyn, stream_ent)
            self.n_seen_so_far += 1
    
    def replace_data(self, idx, img, label, isSyn, ent):
        self.buffer_imgs[idx] = img
        self.buffer_labels[idx] = label
        self.buffer_syn[idx] = isSyn
        self.buffer_ent[idx] = ent
        self.n_added_so_far += 1

    def random_retrieve(self, n_imgs=100):
        if self.n_added_so_far < n_imgs:
            return self.buffer_imgs[:self.n_added_so_far], self.buffer_labels[:self.n_added_so_far], self.buffer_syn[:self.n_added_so_far]
        
        ret_indexes = r.sample(np.arange(min(self.n_added_so_far, self.max_size)).tolist(), n_imgs)
        ret_imgs = self.buffer_imgs[ret_indexes]
        ret_labels = self.buffer_labels[ret_indexes]
        ret_isSyn = self.buffer_syn[ret_indexes]

        return ret_imgs, ret_labels, ret_isSyn

    def get_all_syn(self):
        return self.buffer_imgs[:min(self.n_added_so_far, self.max_size)],\
             self.buffer_labels[:min(self.n_added_so_far, self.max_size)],\
             self.buffer_syn[:min(self.n_added_so_far, self.max_size)]
    
    def get_all_idx(self):
        # retrun all imgs with their indeces in the buffer
        return self.buffer_imgs[:min(self.n_added_so_far, self.max_size)],\
             np.arange(min(self.n_added_so_far, self.max_size))

    def update_ent(self, idx, ent):
        assert len(idx) < self.max_size
        for i, ent in zip(idx, ent):
            self.buffer_ent[i] = ent