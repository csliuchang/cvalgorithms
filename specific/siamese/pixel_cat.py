from .builder import SIAMESE_LAYER
import torch.nn as nn
import torch
from models.base.blocks import ConvModule, BasicConv2d

__all__ = ["PixelCat"]


@SIAMESE_LAYER.register_module()
class PixelCat(nn.Module):
    def __init__(self, in_c, ou_c, norm_cfg=dict(type='BN'), act_cfg=dict(type='ReLU')):
        super(PixelCat, self).__init__()
        self.conv_modules = nn.Sequential(ConvModule(in_c * 2, in_c, 3, 1, 1,
                                                     norm_cfg=norm_cfg,
                                                     act_cfg=act_cfg
                                                     ),
                                          ConvModule(in_c, in_c // 2, 3, 1, 1,
                                                     norm_cfg=norm_cfg,
                                                     act_cfg=act_cfg
                                                     ))
        self.final_conv = nn.Conv2d(in_c // 2, ou_c, 1)

    def forward(self, n, g):
        f = torch.cat([n, g], dim=1)
        f = self.conv_modules(f)
        f = self.final_conv(f)
        return f
