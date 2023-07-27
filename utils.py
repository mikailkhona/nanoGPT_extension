import scipy
import random
import numpy as np
import torch
import os
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch
import random
from typing import Any, Callable, Dict, List, Tuple, Union
import math


# learning rate decay scheduler (cosine with warmup)
def get_cosine_warmp_lr(it, learning_rate, warmup_iters, lr_decay_iters, min_lr):
    '''
    Return lr for it'th step
    '''

    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)



class dotdict(dict):
    '''
    dot.notation access to dictionary attributes
    Wrapper around dictionary
    '''
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

class SequenceDataset(Dataset):
    '''
    Dataset and DataLoader for sequence data.
    Made specifically for autoregressive next token prediction training
    Data is integer-type for tokenizer
    '''

    def __init__(self, filepath, block_size):
        self.data = np.memmap(filepath, dtype=np.uint16, mode='r')
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        #y is 1-index shifted version of x. Everything should be integer for tokenizer.
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + 1 + self.block_size]
        return torch.from_numpy(x.astype(np.int64)), torch.from_numpy(y.astype(np.int64))

def get_dataloader(train_data_path, val_data_path, block_size, batch_size, shuffle=True, num_workers=4):
    '''
    Open data directory and get train and val dataloaders
    '''

    train_dataset = SequenceDataset(train_data_path, block_size)
    val_dataset = SequenceDataset(val_data_path, block_size)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True, num_workers=num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True, num_workers=num_workers)

    return train_dataloader, val_dataloader

