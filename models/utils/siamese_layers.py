from ..builder import SIAMESE_LAYER
from ..base.blocks import ConvModule, BasicConv2d
from ..base.blocks.self_attention_block import PyramidAttentionModule
import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from models.utils import normal_init


@SIAMESE_LAYER.register_module()
class PixelDistance(nn.Module):
    """
    Pixel Distance: See :class:`torch.nn.PairwiseDistance` for details
    """

    def __init__(self, use_att=None):
        super(PixelDistance, self).__init__()
        self.use_att = use_att
        if use_att.enable:
            self.Self_Att = PyramidAttentionModule(in_channels=use_att.in_c, out_channels=use_att.in_c, sizes=[1, 2, 4, 8],
                                                   stride=use_att.stride)
            self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)

    def forward(self, n, g):
        if self.use_att.enable:
            height = n.shape[3]
            x = torch.cat((n, g), 3)
            x = self.Self_Att(x)
            n, g = x[:, :, :, 0:height], x[:, :, :, height:]
        distance = F.pairwise_distance(n, g, keepdim=True)
        return distance


@SIAMESE_LAYER.register_module()
class PixelCat(nn.Module):
    def __init__(self, in_c, ou_c):
        super(PixelCat, self).__init__()
        self.conv_modules = nn.Sequential(ConvModule(in_c*2, in_c, 3, 1, 1,
                                                     norm_cfg=dict(type='BN'),
                                                     act_cfg=dict(type='ReLU')
                                                     ),
                                          ConvModule(in_c, in_c // 2, 3, 1, 1,
                                                     norm_cfg=dict(type='BN'),
                                                     act_cfg=dict(type='ReLU')
                                                     ))
        self.final_conv = nn.Conv2d(in_c // 2, ou_c, 1)

    def forward(self, n, g):
        f = torch.cat([n, g], dim=1)
        f = self.conv_modules(f)
        f = self.final_conv(f)
        return f


@SIAMESE_LAYER.register_module()
class PixelSub(nn.Module):
    def __init__(self, in_c, ou_c, add_conv=True):
        super(PixelSub, self).__init__()
        self.add_conv = add_conv
        self.conv_modules = nn.Sequential(BasicConv2d(in_c, in_c//2, 3, 1, 1),
                                          BasicConv2d(in_c // 2, in_c // 4, 3, 1, 1))
        self.final_conv = nn.Conv2d(in_c // 4, ou_c, 1)

    def forward(self, n, g):
        f = torch.tanh(n - g)
        if self.add_conv:
            f = self.conv_modules(f)
            f = self.final_conv(f)
            return f
        return f
