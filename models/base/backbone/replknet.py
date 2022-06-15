import torch
import torch.nn as nn
import os


class RepLKNet(nn.Module):
    def __init__(self, channels, large_kernel_sizes, layers, drop_path_rate, small_kernel,
                 dw_ratio=1, ffn_ratio=4, in_channels=3, num_classes=1000, out_indices=None,
                 use_checkpoint=False, small_kernel_merged=False, use_sync_bn=False,
                 norm_intermediate_features=False, **kwargs):
        super(RepLKNet, self).__init__()
        base_width = channels[0]
        self.stem = nn.ModuleList([
            ConvBnRelu(
                in_channels=in_channels, out_channels=base_width, kernel_size=3,
                stride=2, padding=1, groups=1),
            ConvBnRelu(
                in_channels=in_channels, out_channels=base_width, kernel_size=3,
                stride=1, padding=1, groups=base_width),
            ConvBnRelu(
                in_channels=in_channels, out_channels=base_width, kernel_size=1,
                stride=1, padding=0, groups=1),
            ConvBnRelu(
                in_channels=in_channels, out_channels=base_width, kernel_size=3,
                stride=2, padding=1, groups=base_width)],
        )
        self.stages = nn.ModuleList()
        for stage_idx in range(self.num_stages):
            layer = RepLKNetStage(

            )
            self.stages.append(layer)

            if stage_idx < len(layers) - 1:
                transition = nn.Sequential(
                    ConvBnRelu(
                        channels[stage_idx], channels[stage_idx + 1], 1, 1, 0, groups=1
                    ),
                    ConvBnRelu(
                        channels[stage_idx + 1], channels[stage_idx + 1], 3, stride=2, padding=1,
                        groups=channels[stage_idx + 1])
                )

        if num_classes is not None:
            self.norm = get_bn(
                channels[-1]
            )
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            self.head = nn.linear(channels[-1], num_classes)

    def forward(self, x):
        pass


def get_bn(channels, use_sync=False):
    if use_sync:
        return nn.SyncBatchNorm(channels)
    else:
        return nn.BatchNorm2d(channels)


class ConvBnRelu(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=None,
                 padding=None,
                 groups=None,
                 dilation=None,
                 nonlinear=True,
                 use_bn=True,
                 **kwargs,
                 ):
        super(ConvBnRelu, self).__init__()
        if type(kernel_size) is int:
            use_large_impl = kernel_size > 5
        else:
            assert len(kernel_size) == 2 and kernel_size[0] == kernel_size[1]
            use_large_impl = kernel_size[0] > 5
        has_large_impl = 'LARGE_KERNEL_CONV_IMPL' in os.environ
        if has_large_impl and in_channels == out_channels and out_channels == groups \
                and use_large_impl and stride == 1 and padding == kernel_size // 2 and dilation == 1:
            pass
        else:
            self.conv =  nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=False
            )
        if use_bn:
            self.conv.add_module('bn', get_bn(in_channels))
        if nonlinear:
            self.conv.add_module('nonlinear', nn.ReLU())

    def forward(self, x):
        return self.conv(x)


class RepLKBlock(nn.Module):
    def __init__(self, in_channels, dw_channels):
        super(RepLKBlock, self).__init__()
        self.pw1 = ConvBnRelu(in_channels, dw_channels, 1, 1, 0, groups=1)
        self.pw2 = ConvBnRelu(dw_channels, in_channels, 1, 1, 0, groups=1, nonlinear=False)
        self.large_kernel = ReparamLargeKernelConv(
        )
        pass

    def forward(self, x):
        out = self.prelkb_bn(x)
        out = self.pw1(out)
        out = self.nonlinear(out)


class ReparamLargeKernelConv(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, groups, small_kernel, small_kernel_merged=False):
        super(ReparamLargeKernelConv, self).__init__()
        self.kernel_size = kernel_size
        self.small_kernel = small_kernel

        padding = kernel_size // 2

        if small_kernel_merged:
            self.lkb_reparam = ConvBnRelu(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                          stride=stride, padding=padding, dilation=1, groups=groups, bias=True)
        else:
            self.lkb_origin = ConvBnRelu(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride, padding=padding,dilation=1, group=groups)


        




