import torch
import torch.nn as nn
from deepcv2.ds_network.builder import NECKS
from deepcv2.ds_network.backbone.components.blocks import BasicConv2d


@NECKS.register_module()
class FeatureFusionNeck(nn.Module):
    def __init__(self, policy,
                 soft_conv=True,
                 in_channels=None,
                 channels=None):
        super(FeatureFusionNeck, self).__init__()
        self.policy = policy
        self.soft_conv = soft_conv
        self.in_channels = in_channels
        if self.soft_conv:
            self.lateral_convs = nn.ModuleList()
            for in_channel in in_channels:
                self.conv_modules = nn.Sequential(
                    BasicConv2d(in_channel, in_channel // 2, 3, 1, 1),
                    BasicConv2d(in_channel // 2, in_channel // 2, 3, 1, 1),
                    nn.Conv2d(in_channel // 2, in_channel // 2, 1)
                )
                self.lateral_convs.append(self.conv_modules)

    @staticmethod
    def sub(x, y):
        feature = torch.relu(x - y)
        return feature

    @staticmethod
    def distance(x, y):
        return torch.abs(x - y)

    @staticmethod
    def sum(x, y):
        return torch.relu(x - y)

    @staticmethod
    def concat(x, y):
        return torch.cat([x, y], dim=1)

    def _fusion_function(self, x, y, function):
        features = getattr(self, self.policy)(x, y)
        return function(features)

    def forward(self, feat_1, feat_2):
        assert len(feat_1) == len(feat_2) == len(self.in_channels)
        outs = []
        for idx, channel in enumerate(self.in_channels):
            outs.append(
                self._fusion_function(feat_1[idx], feat_2[idx], self.lateral_convs[idx])
            )
        return tuple(outs)








