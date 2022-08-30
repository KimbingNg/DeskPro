# Copyright (c) Malong Technologies Co., Ltd.
# All rights reserved.
#
# Contact: github@malong.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.

import torch
from torch import nn


class MultiSimilarityLoss(nn.Module):
    def __init__(self, margin=0.1):
        super(MultiSimilarityLoss, self).__init__()
        self.thresh = 0.5
        self.margin = margin

        self.scale_pos = 2.0
        self.scale_neg = 40.0

    def forward(self, feats, labels):
        assert feats.size(0) == labels.size(0), \
            f"feats.size(0): {feats.size(0)} is not equal to labels.size(0): {labels.size(0)}"
        batch_size = feats.size(0)
        feats = nn.functional.normalize(feats, p=2, dim=1)

        # Shape: batchsize * batch size
        sim_mat = torch.matmul(feats, torch.t(feats))

        epsilon = 1e-5
        loss = list()

        mask = labels.expand(batch_size, batch_size).eq(
            labels.expand(batch_size, batch_size).t())

        pos_pairs:torch.Tensor = torch.ones_like(sim_mat) * float('inf')
        pos_pairs[mask] = sim_mat[mask]
        pos_pairs[pos_pairs >= 1-epsilon] = float('inf')

        neg_pairs:torch.Tensor = torch.ones_like(sim_mat) * -1 * float('inf')
        neg_pairs[mask == 0] = sim_mat[mask == 0]

        neg_pairs[neg_pairs + self.margin <= torch.min(pos_pairs, dim=1)[0]] = -float('inf')
        pos_pairs[pos_pairs - self.margin >= torch.max(neg_pairs, dim=1)[0]] = float('inf')
        #%%
        foo = 1.0 / self.scale_pos * torch.log( 1 + torch.sum(torch.exp(- self.scale_pos *( pos_pairs - self.thresh)), dim=1))
        pos_loss = torch.sum(foo)
        #%%
        bar = 1.0 / self.scale_neg * torch.log( 1 + torch.sum(torch.exp(self.scale_neg *( neg_pairs - self.thresh)), dim=1))
        neg_loss = torch.sum(bar)
        loss1 = (pos_loss + neg_loss) / batch_size
        return loss1
