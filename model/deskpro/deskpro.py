import cv2
import torch
import einops
from loss.kd_loss import KDLoss
from torch.nn import MSELoss

from model.backbone.lmbn.LMBN import MultiBranchBase, LMBN
from model.backbone.lmbn.branches import *
from model.osnet import osnet_x1_0

from .cbam import SpatialAttention


class NoClothNet(MultiBranchBase):
    def __init__(self, args, glob=True, part=True, channel=True):
        super(NoClothNet, self).__init__()
        osnet = osnet_x1_0(pretrained=True)

        self.branches = nn.ModuleList([])

        self.backbone = nn.Sequential(
            osnet.conv1,
            osnet.maxpool,
            osnet.conv2,
            osnet.conv3[0]
        )
        self.glob = glob
        self.part = part
        self.channel = channel

        self.spatial_attn = SpatialAttention(kernel_size=1, in_channel=384)
        self.attn_loss = MSELoss()

        if self.glob:
            self.global_branch = GlobalBranch(args, osnet)
            self.branches.append(self.global_branch)
        if self.part:
            self.part_branch = PartBranch(args, osnet)
            self.branches.append(self.part_branch)
        if self.channel:
            self.channel_branch = ChannelBranch(args, osnet)
            self.branches.append(self.channel_branch)

    def get_mask_tensor(self, nocloth_mask, shape):
        nocloth_mask = einops.rearrange(nocloth_mask, 'bs c h w -> bs h w c')
        hms = []
        for hm in nocloth_mask:
            hm = cv2.resize(hm.data.cpu().numpy(), shape)
            hms.append(hm)
        hms = einops.rearrange(hms, 'bs h w c -> bs h w c')
        head_mask =  einops.rearrange(hms[:, :, :, 0], 'bs h w -> bs 1 h w')
        return torch.tensor(head_mask).float().cuda()

    def forward(self, x):
        x, nocloth_mask = x
        after_backbone = self.backbone(x)
        nocloth_mask = self.get_mask_tensor(nocloth_mask, (after_backbone.shape[3], after_backbone.shape[2]))

        nocloth_mask[nocloth_mask == 0] = 0.1

        attn = self.spatial_attn(after_backbone)
        attn_loss = self.attn_loss(attn, nocloth_mask)
        after_backbone = after_backbone * attn

        outputs = [net(after_backbone) for net in self.branches]

        outputs = self.forward_branches(outputs)

        if self.training:
            return outputs, attn_loss
        else:
            return outputs


class DeskPro(MultiBranchBase):

    def __init__(self, args):
        super(DeskPro, self).__init__()
        self.face_net = LMBN(args)
        self.lr_face_net = LMBN(args)
        self.nocloth_net = NoClothNet(args)

        self.mse = MSELoss()
        self.kd_loss = KDLoss(args.kd_loss.T)

        self.hook_layer = self.nocloth_net.attn_loss
        self.args = args

    def forward(self, x):
        x, lr_face, face, cloth_mask = x
        kd_loss, attn_loss = 0, 0
        mode = self.args.forward_mode
        if mode in ['face', 'all']:
            lr_face_out = self.lr_face_net(lr_face) # [bs, 512, 7] (training)
            if not self.training: # Samples with no face detected are filtered out only during training.
                has_lr_face = lr_face.view(lr_face.shape[0], -1).any(dim=1)
                if (~has_lr_face).any():
                    lr_face_out[~has_lr_face] = torch.zeros(512, 7).to(lr_face_out)
            if self.training:
                with torch.no_grad():
                    face_out = self.face_net(face)
                kd_loss = self.kd_loss(lr_face_out, face_out)
            if mode == 'face':
                return {'out':self.forward_branches([lr_face_out]), 'kd_loss':kd_loss, 'attn_loss': 0}
        if mode in ['body', 'all']:
            if self.training:
                nocloth_out, attn_loss = self.nocloth_net((x, cloth_mask))
            else:
                nocloth_out = self.nocloth_net((x, cloth_mask))
            if mode == 'body':
                return {'out':self.forward_branches([nocloth_out]), 'kd_loss':kd_loss, 'attn_loss': attn_loss}
        if mode == 'all':
            return {'out':self.forward_branches([lr_face_out, nocloth_out]), 'kd_loss':kd_loss, 'attn_loss': attn_loss}