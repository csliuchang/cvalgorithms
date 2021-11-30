from .builder import SIAMESE_LAYER
import torch.nn as nn
import torch
import torch.nn.functional as F
from models.base.blocks.self_attention_block import PyramidAttentionModule

__all__ = ["PixelDistance"]


@SIAMESE_LAYER.register_module()
class PixelDistance(nn.Module):
    """
     the implementation of the STANet: https://www.mdpi.com/2072-4292/12/10/1662
    Pixel Distance: See :class:`torch.nn.PairwiseDistance` for details
    """

    def __init__(self, use_att=None):
        super(PixelDistance, self).__init__()
        self.use_att = use_att
        if use_att.enable:
            self.Self_Att = PyramidAttentionModule(in_channels=use_att.in_c, out_channels=use_att.in_c,
                                                   sizes=[1, 2, 4, 8],
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



