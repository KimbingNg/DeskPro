import copy

import torch
from torch import nn
from torch.nn import functional as F

from model.attention import BatchFeatureErase_Top
from model.bnneck import BNNeck, BNNeck3
from model.osnet import OSBlock
from utils import model_utils


class LMBN_BaseBranch(nn.Module):
    def __init__(self, osnet):
        super(LMBN_BaseBranch, self).__init__()
        conv3 = osnet.conv3[1:]

        self.backbone2 = nn.Sequential(copy.deepcopy(
            conv3), copy.deepcopy(osnet.conv4), copy.deepcopy(osnet.conv5))

    def bnneck_return(self, fs, fea_indices):
        # cls = [f[1] for f in fs]
        # fea = [fs[idx][-1] for idx in fea_indices]
        # return cls, fea
        if not self.training:
            stacked = [f[0] for f in fs]
            stacked = torch.stack(stacked, dim=2)
            return stacked
        cls = [f[1] for f in fs]
        fea = [fs[idx][-1] for idx in fea_indices]
        return cls, fea

class GlobalBranch(LMBN_BaseBranch):
    def __init__(self, args, osnet):
        super(GlobalBranch, self).__init__(osnet)

        reduction = BNNeck3(512, args.num_classes,
                            args.feats, return_f=True)

        self.reduction_1 = copy.deepcopy(reduction)
        self.reduction_2 = copy.deepcopy(reduction)

        self.batch_drop_block = BatchFeatureErase_Top(512, OSBlock)

    def forward(self, after_backbone):
        f_glo, f_glo_drop = self.forward_global(after_backbone)
        return self.bnneck_return([f_glo, f_glo_drop], [0, 1])

    def forward_global(self, x):
        f_glo = self.backbone2(x)
        return self.forward_global2(f_glo)

    def forward_global2(self, f_glo):
        f_glo_drop, f_glo = self.batch_drop_block(f_glo)
        f_glo_drop = F.adaptive_max_pool2d(f_glo_drop, (1, 1))
        f_glo = F.adaptive_avg_pool2d(f_glo, (1,1))
        f_glo = self.reduction_1(f_glo)
        f_glo_drop = self.reduction_2(f_glo_drop)
        return f_glo, f_glo_drop

class PartBranch(LMBN_BaseBranch):

    def __init__(self, args, osnet):
        super(PartBranch, self).__init__(osnet)


        self.partial_pooling = nn.AdaptiveAvgPool2d((2, 1))

        reduction = BNNeck3(512, args.num_classes,
                            args.feats, return_f=True)

        self.reduction_1 = copy.deepcopy(reduction)
        self.reduction_2 = copy.deepcopy(reduction)
        self.reduction_3 = copy.deepcopy(reduction)

    def forward_part(self, x):
        f_p0 = self.backbone2(x)
        f_p1 = self.partial_pooling(f_p0)
        f_p0 = F.adaptive_max_pool2d(f_p0, (1, 1))
        f_p2 = f_p1[:, :, 1:2, :]
        f_p1 = f_p1[:, :, 0:1, :]
        f_p0 = self.reduction_1(f_p0)
        f_p1 = self.reduction_2(f_p1)
        f_p2 = self.reduction_3(f_p2)
        return f_p0, f_p1, f_p2

    def forward(self, after_backbone):
        return self.bnneck_return(self.forward_part(after_backbone), [0])

class ChannelBranch(LMBN_BaseBranch):

    def __init__(self, args, osnet):
        super(ChannelBranch, self).__init__(osnet)

        self.n_ch = 2
        self.chs = 512 // self.n_ch

        self.shared = nn.Sequential(nn.Conv2d(
            self.chs, args.feats, 1, bias=False), nn.BatchNorm2d(args.feats), nn.ReLU(True))

        model_utils.weights_init_kaiming(self.shared)

        self.reduction_ch_0 = BNNeck(
            args.feats, args.num_classes, return_f=True)
        self.reduction_ch_1 = BNNeck(
            args.feats, args.num_classes, return_f=True)

    def forward_channel(self, x):
        cha = self.backbone2(x)
        cha = F.adaptive_avg_pool2d(cha, (1,1))
        f_c0 = cha[:, :self.chs, :, :]
        f_c1 = cha[:, self.chs:, :, :]
        f_c0 = self.shared(f_c0)
        f_c1 = self.shared(f_c1)
        f_c0 = self.reduction_ch_0(f_c0)
        f_c1 = self.reduction_ch_1(f_c1)
        return f_c0, f_c1

    def forward(self, after_backbone):
        return self.bnneck_return(self.forward_channel(after_backbone), [])