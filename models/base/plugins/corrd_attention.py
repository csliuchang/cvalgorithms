import torch
import torch.nn as nn
import torch.nn.functional as F


# @PLUGIN_LAYERS.register_module()
# class CoordAttention(nn.Module):
#
#     def __init__(self, in_channel, reduction=32, **cfg):
#         super(CoordAttention, self).__init__()
#         # self.pool_h = AdaptiveAvgPool2d([None, 1])
#         # self.pool_w = AdaptiveAvgPool2d([1, None])
#
#         min_channel = max(8, in_channel // reduction)
#         self.conv1 = ConvModule(in_channel, min_channel,kernel_size=1, stride=1, padding=0, **cfg)
#         self.conv_h = ConvModule(min_channel, in_channel, kernel_size=1, stride=1, padding=0,norm_cfg=None, act_cfg=None)
#         self.conv_w = ConvModule(min_channel, in_channel, kernel_size=1, stride=1, padding=0,norm_cfg=None, act_cfg=None)
#
#
#     def forward(self, x):
#         identity = x
#
#         n, c, h, w = x.shape
#
#         if torch.is_tensor(h):
#             h = h.item()  # to be constant
#             w = w.item()
#
#         x_h = F.avg_pool2d(x,(1,w))
#         x_w = F.avg_pool2d(x,(h,1)).permute(0, 1, 3, 2).contiguous()
#
#         y = torch.cat([x_h, x_w], dim=2)
#         y = self.conv1(y)
#
#         x_h, x_w = torch.split(y, [h, w], dim=2)
#         x_w = x_w.permute(0, 1, 3, 2).contiguous()
#
#         a_h = self.conv_h(x_h).sigmoid()
#         a_w = self.conv_w(x_w).sigmoid()
#
#         out = identity * a_w * a_h
#
#         return out


class AdaptiveAvgPool2d(nn.Module):
    def __init__(self, sz=None):
        super().__init__()

        self.sz = sz

    def forward(self, x):
        inp_size = x.size()

        kernel_width, kernel_height = inp_size[2], inp_size[3]
        if self.sz is not None:
            if isinstance(self.sz, int):
                kernel_width = torch.ceil(inp_size[2] / self.sz)
                kernel_height = torch.ceil(inp_size[3] / self.sz)
            elif isinstance(self.sz, list) or isinstance(self.sz, tuple):
                assert len(self.sz) == 2
                self.sz[0] = self.sz[0] if self.sz[0] else kernel_height
                self.sz[1] = self.sz[1] if self.sz[1] else kernel_width
                kernel_width = inp_size[2] // self.sz[0]
                kernel_height =inp_size[3] // self.sz[1]

            return F.avg_pool2d(input=x,
                                ceil_mode=False,
                                kernel_size=(kernel_width, kernel_height))


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAttention(nn.Module):
    def __init__(self, inp, reduction=32):
        """
            https://github.com/Andrew-Qibin/CoordAttention/blob/main/coordatt.py
        """
        super(CoordAttention, self).__init__()
        # self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        # self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x


        n, c, h, w = x.size()

        if torch.is_tensor(h):
            h = h.item()  # make h and w to be constant
            w = w.item()
        pool_h = nn.AdaptiveAvgPool2d((h, 1))
        pool_w = nn.AdaptiveAvgPool2d((1, w))

        x_h = pool_h(x)
        x_w = pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out

