import torch
import torch.nn as nn
import torch.nn.functional as F
from deepcv2.ds_network.utils import DropPath, trunc_normal_
from deepcv2.ds_network.builder import BACKBONES
from deepcv2.ds_network.backbone.com import _get_backbone
from typing import List, Optional

from torch import Tensor, reshape, stack

from torch.nn import (
    Conv2d,
    InstanceNorm2d,
    Module,
    ModuleList,
    PReLU,
    Sequential,
    Upsample,
)


@BACKBONES.register_module()
class TinyCD(nn.Module):
    def __init__(self,
                 in_channels,
                 torch_name,
                 out_levels,
                 num_classes,
                 pretrained=True,
                 output_layer_bkbn="3",
                 freeze_backbone=False,
                 **kwargs
                 ):
        super().__init__()
        # if warp backbone, use _get_backbone
        # Load the pretrained backbone according to parameters:
        self._backbone = _get_backbone(
            torch_name, pretrained, output_layer_bkbn, freeze_backbone
        )

        self.conv_channel = Conv2d(in_channels=in_channels, out_channels=3, kernel_size=(3, 3),
                                   stride=(1, 1), padding=1)

        self._first_mix = MixingMaskAttentionBlock(in_channels * 2, in_channels,
                                                   [in_channels, 10, 5], [10, 5, 1])

        self._mixing_mask = nn.ModuleList(
            [
                MixingMaskAttentionBlock(48, 24, [24, 12, 6], [12, 6, 1]),
                MixingMaskAttentionBlock(64, 32, [32, 16, 8], [16, 8, 1]),
                MixingBlock(112, 56),
            ]
        )

        self._up = nn.ModuleList(
            [
                UpMask(64, 56, 64),
                UpMask(128, 64, 64),
                UpMask(256, 64, 32),
            ]
        )

        self._classify = PixelwiseLinear([32, 16], [16, 8], None)

    def _encode(self, ref, test) -> List[Tensor]:
        features = [self._first_mix(ref, test)]
        if ref.shape[1] | test.shape[1] < 3:
            ref, test = self.conv_channel(ref), self.conv_channel(test)
        for num, layer in enumerate(self._backbone):
            ref, test = layer(ref), layer(test)
            if num != 0:
                features.append(self._mixing_mask[num - 1](ref, test))
        return features

    def _decode(self, features) -> Tensor:
        upping = features[-1]
        for i, j in enumerate(range(-2, -5, -1)):
            upping = self._up[i](upping, features[j])
        return upping

    def forward(self, x1, x2):
        features = self._encode(x1, x2)
        latents = self._decode(features)
        out = self._classify(latents)

        return out,


class UpMask(Module):
    def __init__(
            self,
            up_dimension: int,
            nin: int,
            nout: int,
    ):
        super().__init__()
        self._upsample = Upsample(
            size=(up_dimension, up_dimension), mode="bilinear", align_corners=True
        )
        self._convolution = Sequential(
            Conv2d(nin, nin, 3, 1, groups=nin, padding=1),
            PReLU(),
            InstanceNorm2d(nin),
            Conv2d(nin, nout, kernel_size=1, stride=1),
            PReLU(),
            InstanceNorm2d(nout),
        )

    def forward(self, x: Tensor, y: Optional[Tensor] = None) -> Tensor:
        x = self._upsample(x)
        if y is not None:
            x = x * y
        return self._convolution(x)


class PixelwiseLinear(Module):
    def __init__(
            self,
            fin: List[int],
            fout: List[int],
            last_activation: Module = None,
    ) -> None:
        assert len(fout) == len(fin)
        super().__init__()

        n = len(fin)
        self._linears = Sequential(
            *[
                Sequential(
                    Conv2d(fin[i], fout[i], kernel_size=1, bias=True),
                    PReLU()
                    if i < n - 1 or last_activation is None
                    else last_activation,
                )
                for i in range(n)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        # Processing the tensor:
        return self._linears(x)


class MixingMaskAttentionBlock(nn.Module):
    def __init__(self,
                 ch_in,
                 ch_out,
                 fin: List[int],
                 fout: List[int],
                 generate_masked: bool = False):
        super(MixingMaskAttentionBlock, self).__init__()
        self._mixing = MixingBlock(ch_in, ch_out)
        self._linear = PixelwiseLinear(fin, fout)
        self._final_normalization = nn.InstanceNorm2d(ch_out) if generate_masked else None
        self._mixing_out = MixingBlock(ch_in, ch_out) if generate_masked else None

    def forward(self, x, y):
        z_mix = self._mixing(x, y)
        z = self._linear(z_mix)
        z_mix_out = 0 if self._mixing_out is None else self._mixing_out(x, y)
        return (
            z
            if self._final_normalization is None
            else self._final_normalization(z_mix_out * z)
        )


class MixingBlock(nn.Module):
    def __init__(self,
                 ch_in: int,
                 ch_out: int,
                 ):
        super(MixingBlock, self).__init__()
        self._convmix = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, 3, groups=ch_out, padding=1),
            nn.PReLU(),
            nn.InstanceNorm2d(ch_out),
        )

    def forward(self, x, y):
        mixed = torch.stack((x, y), dim=2)
        mixed = torch.reshape(mixed, (x.shape[0], -1, x.shape[2], x.shape[3]))
        return self._convmix(mixed)
