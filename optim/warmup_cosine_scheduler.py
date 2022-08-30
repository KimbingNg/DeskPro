# encoding: utf-8


import torch
import matplotlib.pyplot as plt

from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim.lr_scheduler as lrs

import math


class WarmupCosineAnnealingLR(_LRScheduler):

    def __init__(self, optimizer, multiplier, warmup_epoch, epochs, min_lr=3.5e-7, last_epoch=-1):
        self.multiplier = multiplier
        if self.multiplier < 1.:
            raise ValueError(
                'multiplier should be greater thant or equal to 1.')
        self.warmup_epoch = warmup_epoch
        self.last_epoch = last_epoch
        self.eta_min = min_lr
        self.T_max = float(epochs - warmup_epoch)
        self.epochs = epochs
        self.after_scheduler = True
        self.after_cos_epochs = 500

        super(WarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch > self.warmup_epoch - 1:
            if self.last_epoch > self.epochs:
                #print(f'{self.last_epoch}  {self.epochs}')
                e = self.last_epoch - self.epochs
                #return self.eta_min + (self.last_lr - self.eta_min) *
                if not hasattr(self, 'last_lr'):
                    # dirty hack
                    self.last_lr = [self.eta_min + (base_lr - self.eta_min) *
                                    (1 + math.cos(math.pi * (self.epochs - self.warmup_epoch) / (self.T_max - 1))) / 2
                                    for base_lr in self.base_lrs]
                return [self.eta_min + (l - self.eta_min) * (1 - e / self.after_cos_epochs) for l in self.last_lr]

            self.last_lr = [self.eta_min + (base_lr - self.eta_min) *
                    (1 + math.cos(math.pi * (self.last_epoch -
                                             self.warmup_epoch) / (self.T_max - 1))) / 2
                    for base_lr in self.base_lrs]
            return self.last_lr

        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch + 1) / self.warmup_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.warmup_epoch + 1.) for base_lr in self.base_lrs]
