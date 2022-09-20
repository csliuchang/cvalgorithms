from .builder import SIAMESE_LAYER
import torch.nn as nn
from torch import Tensor, reshape
import torch
from typing import List, Optional


@SIAMESE_LAYER.register_module()
class PixelStack(nn.Module):
    def __init__(self, c_in, c_out):
        super(PixelStack, self).__init__()
        self._convmix = nn.Sequential(
            nn.Conv2d(
                c_in, c_out, kernel_size=(3, 3), groups=c_out, padding=(1, 1),
            ),
            nn.PReLU(),
            nn.InstanceNorm2d(c_out),
        )

    def forward(self, x1: Tensor, x2: Tensor):
        mixed = torch.stack((x1, x2), dim=2)
        mixed = reshape(mixed, (x1.shape[0], -1, x1.shape[2], x1.shape[3]))
        return self._convmix(mixed)


class PixelwiseLinear(nn.Module):
    def __init__(self, f_in: List[int], f_out: List[int], last_activation: nn.Module = None):
        super(PixelwiseLinear, self).__init__()
        assert len(f_in) == len(f_out)
        n = len(f_in)
        self._linears = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Conv2d(f_in[i], f_out[i], kernel_size=(1, 1), bias=True),
                    nn.PReLU()
                    if i < n - 1 or last_activation is None
                    else last_activation,
                )
                for i in range(n)
            ]

        )

    def forward(self, x: Tensor) -> Tensor:
        return self._linears(x)


@SIAMESE_LAYER.register_module()
class PixelStackAttention(nn.Module):
    def __init__(self, c_in, c_out, f_in, f_out, generate_masked: bool = False):
        super(PixelStackAttention, self).__init__()
        self._mixing = PixelStack(c_in, c_out)
        self._linear = PixelwiseLinear(f_in, f_out)
        self._mixing_out = PixelStack(c_in, c_out) if generate_masked else None
        self._final_normalization = nn.InstanceNorm2d(c_in, c_out) if generate_masked else None

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        z_mix = self._mixing(x, y)
        z = self._linear(z_mix)
        z_mix_out = 0 if self._mixing_out is None else self._mixing_out(x, y)
        return (
            z if self._final_normalization is None
            else self._final_normalization(z_mix_out * z)
        )


if __name__ == "__main__":
   a = torch.randn(1, 24, 128, 128)
   b = torch.randn(1, 24, 128, 128)
   model = PixelStackAttention(48, 24, [24, 12, 6], [12, 6, 1])
   print(model(a, b))