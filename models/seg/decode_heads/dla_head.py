import torch
import math
import torch.nn as nn
import numpy as np
from ...base import ConvModule, resize
from ...builder import HEADS, build_loss
from .decode_head import BaseDecodeHead
import torch.nn.functional as F


def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[:, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))


class IDAUP(nn.Module):

    def __init__(self, kernel, out_dim, channels, up_factors,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='ReLU', inplace=True),
                 ):
        super(IDAUP, self).__init__()
        self.channels = channels
        self.out_dim = out_dim

        for i, c in enumerate(channels):
            if c == out_dim:
                proj = nn.Identity()
            else:
                proj = ConvModule(c, out_dim, kernel_size=1, stride=1, norm_cfg=norm_cfg, act_cfg=act_cfg)
            f = int(up_factors[i])
            if f == 1:
                up = nn.Identity()
            else:
                up = nn.ConvTranspose2d(out_dim, out_dim, f * 2, stride=f, padding=f // 2,
                                        output_padding=0, groups=out_dim, bias=False)

                fill_up_weights(up)

            setattr(self, 'proj_' + str(i), proj)
            setattr(self, 'up_' + str(i), up)

        for i in range(1, len(channels)):
            node = ConvModule(out_dim * 2, out_dim, kernel_size=kernel, stride=1,
                              padding=kernel // 2,
                              norm_cfg=norm_cfg, act_cfg=act_cfg)

            setattr(self, 'node_' + str(i), node)

        self.init_weights()

    def forward(self, layers):
        assert len(self.channels) == len(layers), \
            '{} vs {} layers'.format(len(self.channels), len(layers))

        for i, l in enumerate(layers):
            upsample = getattr(self, 'up_' + str(i))
            project = getattr(self, 'proj_' + str(i))
            layers[i] = upsample(project(l))

        x = layers[0]
        y = []
        for i in range(1, len(layers)):
            node = getattr(self, 'node_' + str(i))
            x = node(torch.cat([x, layers[i]], dim=1))
            y.append(x)
        return x, y

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class DLAUP(nn.Module):
    def __init__(self, channels, scales=(1, 2, 4, 8, 16),
                 in_channels=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='ReLU', inplace=True),
                 ):
        super(DLAUP, self).__init__()
        if in_channels is None:
            in_channels = channels
        channels = list(channels)
        scales = np.array(scales, dtype=int)
        for i in range(len(channels) - 1):
            j = -i - 2
            setattr(self, 'ida_{}'.format(i),
                    IDAUP(3, channels[j], in_channels[j:],
                          scales[j:] // scales[j], norm_cfg=norm_cfg, act_cfg=act_cfg))
            scales[j + 1:] = scales[j]
            in_channels[j + 1:] = [channels[j] for _ in channels[j + 1:]]

    def forward(self, layers):
        layers = list(layers)
        assert len(layers) > 1
        for i in range(len(layers) - 1):
            ida = getattr(self, 'ida_{}'.format(i))
            x, y = ida(layers[-i - 2:])
            layers[-i - 1:] = y
        return x


@HEADS.register_module()
class DLAHead(BaseDecodeHead):

    def __init__(self,
                 loss,
                 down_ratio=2,
                 **kwargs):
        super(DLAHead, self).__init__(**kwargs)

        assert down_ratio in [2, 4, 8, 16]

        scales = [2 ** i for i in range(len(self.in_channels))]
        self.dla_up = DLAUP(self.in_channels, scales=scales,
                            norm_cfg=self.norm_cfg,
                            act_cfg=self.act_cfg
                            )
        self.fc = nn.Conv2d(self.in_channels[0], self.num_classes, kernel_size=(1, 1),
                            stride=(1, 1), padding=(0, 0), bias=True)

        self.loss = build_loss(loss)

        self.init_weights()

    def forward(self, x):
        x = self.dla_up(x)
        x = self.fc(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        return x

    def init_weights(self):
        for m in self.fc.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def losses(self, seg_logit, seg_label):
        loss = dict()
        loss_1 = self.loss(seg_logit, seg_label)
        loss["loss"] = loss_1
        return loss
