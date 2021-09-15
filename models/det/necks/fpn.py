from models.base.layers import Conv2d, ShapeSpec
from models.base.norms import get_norm
import torch.nn.functional as F
from models.builder import NECKS
import torch.nn as nn
import math
from models.utils import c2_xavier_fill

__all__ = ["FPN"]


@NECKS.register_module()
class FPN(nn.Module):
    """

    This module implements : paper: FPN.
    example:
            input_shapes: the input feature channels such as [64, 128, 256, 512]
            in_features: the output feature scale end
    """

    def __init__(self, input_shapes, out_channels, strides, norm="", fuse_type="sum", in_features='p7',
                 top_block='LastLevelMaxPool'):
        super().__init__()
        lateral_convs = []
        output_convs = []
        if top_block == 'LastLevelMaxPool':
            top_block = LastLevelMaxPool()
        elif top_block == 'LastLevelP6P7':
            top_block = LastLevelP6P7(out_channels, out_channels, in_features=in_features)
        else:
            pass

        use_bias = norm == ""
        for idx, in_channels in enumerate(input_shapes):
            lateral_norm = get_norm(norm, out_channels)
            output_norm = get_norm(norm, out_channels)

            lateral_conv = Conv2d(
                in_channels, out_channels, kernel_size=1, bias=use_bias, norm=lateral_norm
            )
            output_conv = Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=use_bias,
                norm=output_norm,
            )

            c2_xavier_fill(lateral_conv)
            c2_xavier_fill(output_conv)
            stage = int(math.log2(strides[idx]))
            self.add_module("fpn_lateral{}".format(stage), lateral_conv)
            self.add_module("fpn_output{}".format(stage), output_conv)

            lateral_convs.append(lateral_conv)
            output_convs.append(output_conv)

        self.lateral_convs = lateral_convs[::-1]  # reverse the lateral convs
        self.output_convs = output_convs[::-1]

        self._out_feature_strides = {"p{}".format(int(math.log2(s))): s for s in strides}

        self.top_block = top_block

        if self.top_block is not None:
            for s in range(stage, stage + self.top_block.num_levels):
                self._out_feature_strides["p{}".format(s + 1)] = 2 ** (s + 1)

        self._out_features = list(self._out_feature_strides.keys())
        self._out_feature_channels = {k: out_channels for k in self._out_features}

        assert fuse_type in {"avg", "sum"}
        self._fuse_type = fuse_type

        self.rev_in_features = list(range(len(input_shapes)))[::-1]


    def forward(self, x):
        """
        :param x: a tuple contain stage of backbone,
                for example:
                [res2,res3,res4,res5]
        :return:
        """
        results = []
        prev_features = self.lateral_convs[0](x[-1])
        results.append(self.output_convs[0](prev_features))

        # Reverse feature maps into top-down order (from low to high resolution)
        for features_id, lateral_conv, output_conv in zip(self.rev_in_features[1:], self.lateral_convs[1:],
                                                          self.output_convs[1:]):
            features = x[features_id]
            top_down_features = F.interpolate(prev_features, scale_factor=2.0, mode="bilinear", align_corners=True)
            lateral_features = lateral_conv(features)
            prev_features = lateral_features + top_down_features
            if self._fuse_type == "avg":
                prev_features /= 2
            results.insert(0, output_conv.forward(prev_features))

        if self.top_block is not None:
            top_block_in_feature = results[self._out_features.index(self.top_block.in_feature)]
            results.extend(self.top_block(top_block_in_feature))

        return results

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }


class LastLevelP6P7(nn.Module):
    """
    This module is used in RetinaNet and FCOS to generate extra layers, P6 and P7 from
    C5 or P5 feature.
    """

    def __init__(self, in_channels, out_channels, in_features="res5"):
        super().__init__()
        self.num_levels = 2
        self.in_feature = in_features
        self.p6 = nn.Conv2d(in_channels, out_channels, 3, 2, 1)
        self.p7 = nn.Conv2d(out_channels, out_channels, 3, 2, 1)
        for module in [self.p6, self.p7]:
            c2_xavier_fill(module)

    def forward(self, x):
        p6 = self.p6(x)
        p7 = self.p7(F.relu(p6))
        return [p6, p7]


class LastLevelMaxPool(nn.Module):
    """
    This module is used in the original FPN to generate a downsampled
    P6 feature from P5.
    """

    def __init__(self):
        super().__init__()
        self.num_levels = 1
        self.in_feature = "p5"

    def forward(self, x):
        return [F.max_pool2d(x, kernel_size=1, stride=2, padding=0)]
