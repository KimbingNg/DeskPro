import os
import numpy as np
from importlib import import_module

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from .triplet import TripletLoss, TripletSemihardLoss, CrossEntropyLabelSmooth
from .grouploss import GroupLoss
from loss.multi_similarity_loss import MultiSimilarityLoss
from loss.focal_loss import FocalLoss
from loss.osm_caa_loss import OSM_CAA_Loss
from loss.center_loss import CenterLoss


class LossFunction():
    def __init__(self, args):
        super(LossFunction, self).__init__()

        self.args = args
        self.loss = []
        for loss in args.loss.split('+'):
            weight, loss_type = loss.split('*')
            if loss_type == 'CrossEntropy':
                if args.if_labelsmooth:
                    loss_function = CrossEntropyLabelSmooth(
                        num_classes=args.num_classes)
                else:
                    loss_function = nn.CrossEntropyLoss()
            elif loss_type == 'Triplet':
                loss_function = TripletLoss(args.margin)
            elif loss_type == 'GroupLoss':
                loss_function = GroupLoss(
                    total_classes=args.num_classes, max_iter=args.T, num_anchors=args.num_anchors)
            elif loss_type == 'MSLoss':
                loss_function = MultiSimilarityLoss(margin=args.margin)
            elif loss_type == 'Focal':
                loss_function = FocalLoss(reduction='mean')
            elif loss_type == 'OSLoss':
                loss_function = OSM_CAA_Loss()
            elif loss_type == 'CenterLoss':
                loss_function = CenterLoss(
                    num_classes=args.num_classes, feat_dim=args.feats)

            self.loss.append({
                'type': loss_type,
                'weight': float(weight),
                'function': loss_function
            })

        if len(self.loss) > 1:
            self.loss.append({'type': 'Total', 'weight': 0, 'function': None})

        self.special_ce = lambda x: False
        self.special_tr = lambda x: False
        self.ce_weights = 1
        if args.kd_loss.enable:
            i = 1
            self.special_ce = lambda x: x >= (i - 1) * 7 and x < i * 7
            self.special_tr = lambda x: x >= (i - 1) * 3 and x < i * 3
            self.ce_weights = 1 - args.kd_loss.alpha
            self.tr_weights = args.kd_loss.tr_weight


    def compute(self, outputs, labels, for_loss_compute=None):
        losses = []

        for i, l in enumerate(self.loss):
            if l['type'] in ['CrossEntropy']:

                if isinstance(outputs[0], list):
                    loss = [self.ce_weights * l['function'](output, labels) if self.special_ce(idx)
                            else l['function'](output, labels)
                            for idx, output in enumerate(outputs[0])]
                elif isinstance(outputs[0], torch.Tensor):
                    raise NotImplementedError
                    loss = [l['function'](outputs[0], labels)]
                else:
                    raise TypeError(
                        'Unexpected type: {}'.format(type(outputs[0])))

                loss = sum(loss)
                effective_loss = l['weight'] * loss
                losses.append(effective_loss)


            elif l['type'] in ['Triplet', 'MSLoss']:
                if isinstance(outputs[-1], list):
                    loss = [self.tr_weights * l['function'](output, labels) if self.special_tr(idx)
                            else l['function'](output, labels)
                            for idx, output in enumerate(outputs[-1])]
                elif isinstance(outputs[-1], torch.Tensor):
                    loss = [l['function'](outputs[-1], labels)]
                else:
                    raise TypeError(
                        'Unexpected type: {}'.format(type(outputs[-1])))
                loss = sum(loss)
                effective_loss = l['weight'] * loss
                losses.append(effective_loss)

            elif l['type'] in ['GroupLoss']:
                if isinstance(outputs[-1], list):
                    loss = [l['function'](output[0], labels, output[1])
                            for output in zip(outputs[-1], outputs[0][:3])]
                elif isinstance(outputs[-1], torch.Tensor):
                    loss = [l['function'](outputs[-1], labels)]
                else:
                    raise TypeError(
                        'Unexpected type: {}'.format(type(outputs[-1])))
                loss = sum(loss)
                effective_loss = l['weight'] * loss
                losses.append(effective_loss)

            elif l['type'] in ['CenterLoss']:
                if isinstance(outputs[-1], list):
                    loss = [l['function'](output, labels)
                            for output in outputs[-1]]
                elif isinstance(outputs[-1], torch.Tensor):
                    loss = [l['function'](outputs[-1], labels)]
                else:
                    raise TypeError(
                        'Unexpected type: {}'.format(type(outputs[-1])))

                loss = sum(loss)
                effective_loss = l['weight'] * loss
                losses.append(effective_loss)

            else:
                pass

        loss_sum = sum(losses)

        return loss_sum


    def display_loss(self, batch):
        n_samples = batch + 1
        log = []
        for l, c in zip(self.loss, self.log[-1]):
            log.append('[{}: {:.6f}]'.format(l['type'], c / n_samples))

        return ''.join(log)

    def plot_loss(self, apath, epoch):
        axis = np.linspace(1, epoch, epoch)
        for i, l in enumerate(self.loss):
            label = '{} Loss'.format(l['type'])
            fig = plt.figure()
            plt.title(label)

            plt.plot(axis, self.log[:, i].numpy(), label=label)
            plt.legend()
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.savefig('{}/loss_{}.jpg'.format(apath, l['type']))
            plt.close(fig)


    def step(self):
        for l in self.get_loss_module():
            if hasattr(l, 'scheduler'):
                l.scheduler.step()

    def get_loss_module(self):
        return self.loss_module

    def save(self, apath):
        torch.save(self.state_dict(), os.path.join(apath, 'loss.pt'))
        torch.save(self.log, os.path.join(apath, 'loss_log.pt'))

    def load(self, apath, cpu=False):
        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}
        else:
            kwargs = {}

        self.load_state_dict(torch.load(
            os.path.join(apath, 'loss.pt'),
            **kwargs
        ))
        self.log = torch.load(os.path.join(apath, 'loss_log.pt'))
        for l in self.loss_module:
            if hasattr(l, 'scheduler'):
                for _ in range(len(self.log)):
                    l.scheduler.step()


def make_loss(args):
    return LossFunction(args)
