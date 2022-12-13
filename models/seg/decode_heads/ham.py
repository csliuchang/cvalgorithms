# SegNext
from .decode_head import BaseDecodeHead
import torch.nn as nn


class _MatrixDecomposition2DBase(nn.Module):
    def __init__(self, spatial):
        super(_MatrixDecomposition2DBase, self).__init__()
        self.spatial = spatial

    def forward(self, x):
        B, C, H, W = x.shape
        if self.spatial:
            pass


class MMF2D(_MatrixDecomposition2DBase):
    def __init__(self):
        super(MMF2D, self).__init__()
        pass


class Hamburger(nn.Module):
    def __init__(self):
        super(Hamburger, self).__init__()

    def forward(self, x):
        enjoy = self.ham_in(x)
        pass
