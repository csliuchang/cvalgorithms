from .builder import SIAMESE_LAYER
import torch.nn as nn
import torch
from models.base.blocks.comm_blocks import BasicConv2d

__all__ = ["PixelSub"]


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
