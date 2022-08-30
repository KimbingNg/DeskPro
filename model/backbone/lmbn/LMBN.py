from .branches import *
from model.osnet import osnet_x1_0
from model.utils import reg_model


class MultiBranchBase(nn.Module):

    def forward_branches(self, branch_outputs):
        # case 1:
        # [3, [bs, 512, 2 or 3]]
        # return [bs, 512, 7]
        # case 2:
        # [2, [bs, 512, 7]]
        # return [bs, 512, 14]
        if not self.training:
            return torch.cat(branch_outputs, dim=2)
        # cls, fea
        # List[arr(bs, cls)], List[arr(bs, feat_dim)]
        # len = 7             len = 3
        return list(map(lambda x: sum(x, []), zip(*branch_outputs)))


@reg_model('LMBN')
class LMBN(MultiBranchBase):
    def __init__(self, args, glob=True, part=True, channel=True):
        super(LMBN, self).__init__()

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

        if self.glob:
            self.global_branch = GlobalBranch(args, osnet)
            self.branches.append(self.global_branch)
        if self.part:
            self.part_branch = PartBranch(args, osnet)
            self.branches.append(self.part_branch)
        if self.channel:
            self.channel_branch = ChannelBranch(args, osnet)
            self.branches.append(self.channel_branch)

    def forward(self, x):
        x = self.backbone(x)
        outputs = [net(x) for net in self.branches]
        return self.forward_branches(outputs)

