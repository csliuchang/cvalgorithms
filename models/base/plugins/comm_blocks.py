import torch
import torch.nn as nn
from models.base.blocks.conv_module import ConvModule
from deepcv.pytorch_networks.base.cnn.components.plugins.corrd_attention import CoordAttention


class Bottleneck(nn.Module):
    """
    A standard bottlenck from resnet

    parameters
    ----------
    channel_in: input channel dimension of bottleneckcsp structure;

    channel_out: output channel dimension of bottleneckcsp structure;

    Shortcut: whether to add a shortcut connection to the bottleneck structure. After adding, it is the ResNet module;

    g: Groups, the parameters of channel grouping, the number of input channels and the number of output channels must be divisible by groups at the same time;

    e: Expansion: the channel expansion rate of the bottleneck part in the bottleneck structure is 0.5;
    """
    def __init__(self, channel_in, channel_out, shortcut=True, g=1, e=0.5,**cfg):  # ch_in, ch_out, shortcut, groups, expansion
        super(Bottleneck, self).__init__()
        c_ = int(channel_out * e)  # hidden channels
        self.cv1 = ConvModule(channel_in, c_, 1, 1,**cfg)
        self.cv2 = ConvModule(c_, channel_out, 3, 1,1, groups=g, **cfg)
        self.add = shortcut and channel_in == channel_out
        self.attention = CoordAttention(channel_in)

    def forward(self, x):
        return x + self.cv2(self.attention(self.cv1(x))) if self.add else self.cv2(self.attention(self.cv1(x)))


class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self, input_channels, output_channels,
                 kernel_size=1, stride=1, padding=None, groups=1,
                 norm_cfg=None, act_cfg=None, **kwargs):
        super(Focus, self).__init__()
        self.conv = ConvModule(input_channels * 4, output_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding, groups=groups,
                               norm_cfg=norm_cfg,act_cfg=act_cfg)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))


class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self,  input_channels, output_channels, attention=False,
                 number=1, shortcut=True, groups=1, expansion=0.5, **cfg):
        super(C3, self).__init__()
        c_ = int(output_channels * expansion)  # hidden channels
        self.cv1 = ConvModule(input_channels, c_, 1, 1,**cfg)
        self.cv2 = ConvModule(input_channels, c_, 1, 1,**cfg)
        self.cv3 = ConvModule(2 * c_, output_channels, 1,**cfg)  # act=FReLU(c2)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, groups,
                                            attention=attention, e=1.0, **cfg) for _ in range(number)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))

