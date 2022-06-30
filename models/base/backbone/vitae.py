import torch
import torch.nn as nn
from functools import partial
import numpy as np
import math



def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class PRM(nn.Module):
    def __init__(self, img_size=224, kernel_size=4, downsample_ratio=4, dilations=[1, 6, 12], in_chans=3, embed_dim=64,
                 share_weights=False, op='cat'):
        super().__init__()
        self.dilations = dilations
        self.embed_dim = embed_dim
        self.downsample_ratio = downsample_ratio
        self.op = op
        self.kernel_size = kernel_size
        self.stride = downsample_ratio
        self.share_weights = share_weights
        self.outSize = img_size // downsample_ratio

        if share_weights:
            self.convolution = nn.Conv2d(in_channels=in_chans, out_channels=embed_dim, kernel_size=self.kernel_size, \
                                         stride=self.stride, padding=3 * dilations[0] // 2, dilation=dilations[0])

        else:
            self.convs = nn.ModuleList()
            for dilation in self.dilations:
                padding = math.ceil(((self.kernel_size - 1) * dilation + 1 - self.stride) / 2)
                self.convs.append(nn.Sequential(
                    *[nn.Conv2d(in_channels=in_chans, out_channels=embed_dim, kernel_size=self.kernel_size, \
                                stride=self.stride, padding=padding, dilation=dilation),
                      nn.GELU()]))

        if self.op == 'sum':
            self.out_chans = embed_dim
        elif op == 'cat':
            self.out_chans = embed_dim * len(self.dilations)

    def forward(self, x):
        if self.share_weights:
            padding = math.ceil(((self.kernel_size - 1) * self.dilations[0] + 1 - self.stride) / 2)
            y = nn.functional.conv2d(x, weight=self.convolution.weight, bias=self.convolution.bias, \
                                     stride=self.downsample_ratio, padding=padding,
                                     dilation=self.dilations[0]).unsqueeze(dim=-1)
            for i in range(1, len(self.dilations)):
                padding = math.ceil(((self.kernel_size - 1) * self.dilations[i] + 1 - self.stride) / 2)
                _y = nn.functional.conv2d(x, weight=self.convolution.weight, bias=self.convolution.bias, \
                                          stride=self.downsample_ratio, padding=padding,
                                          dilation=self.dilations[i]).unsqueeze(dim=-1)
                y = torch.cat((y, _y), dim=-1)
        else:
            y = self.convs[0](x).unsqueeze(dim=-1)
            for i in range(1, len(self.dilations)):
                _y = self.convs[i](x).unsqueeze(dim=-1)
                y = torch.cat((y, _y), dim=-1)
        B, C, W, H, N = y.shape
        if self.op == 'sum':
            y = y.sum(dim=-1).flatten(2).permute(0, 2, 1).contiguous()
        elif self.op == 'cat':
            y = y.permute(0, 4, 1, 2, 3).flatten(3).reshape(B, N * C, W * H).permute(0, 2, 1).contiguous()
        else:
            raise NotImplementedError('no such operation: {} for multi-levels!'.format(self.op))
        return y, (W, H)


class PatchEmbedding(nn.Module):
    def __init__(self, inter_channel=32, out_channels=48, img_size=None):
        self.img_size = img_size
        self.inter_channel = inter_channel
        self.out_channel = out_channels
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, inter_channel, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(inter_channel),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(inter_channel, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv3(self.conv2(self.conv1(x)))
        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1).reshape(b, h * w, c)
        return x


class WindowTransformerBlock(nn.Module):
    r""" Window Transformer Block.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, in_dim, out_dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 relative_pos=True, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.in_dim = in_dim
        self.dim = out_dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.relative_pos = relative_pos
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(in_dim)
        self.attn = WindowAttention(
            in_dim=in_dim, out_dim=out_dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, relative_pos=relative_pos)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(out_dim)
        mlp_hidden_dim = int(out_dim * mlp_ratio)
        self.mlp = Mlp(in_features=out_dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)


class ReductionCell(nn.Module):
    def __init__(self, img_size=224, in_chans=3, embed_dims=64, token_dims=64, downsample_ratios=4, kernel_size=7,
                 num_heads=1, dilations=[1,2,3,4], share_weights=False, op='cat', tokens_type='performer', group=1,
                 relative_pos=False, drop=0., attn_drop=0., drop_path=0., mlp_ratio=1.0, window_size=7):
        super().__init__()

        self.img_size = img_size
        self.window_size = window_size
        self.op = op
        self.tokens_type = tokens_type
        self.dilations = dilations
        self.num_heads = num_heads
        self.embed_dims = embed_dims
        self.token_dims = token_dims
        self.in_chans = in_chans
        self.downsample_ratios = downsample_ratios
        self.kernel_size = kernel_size
        self.outSize = img_size
        self.relative_pos = relative_pos
        PCMStride = []
        residual = downsample_ratios // 2
        for _ in range(3):
            PCMStride.append((residual > 0) + 1)
            residual = residual // 2
        assert residual == 0
        self.pool = None

        self.PCM = nn.Sequential(
                        nn.Conv2d(in_chans, embed_dims, kernel_size=(3, 3), stride=PCMStride[0], padding=(1, 1), groups=group),  # the 1st convolution
                        nn.BatchNorm2d(embed_dims),
                        nn.SiLU(inplace=True),
                        nn.Conv2d(embed_dims, embed_dims, kernel_size=(3, 3), stride=PCMStride[1], padding=(1, 1), groups=group),  # the 1st convolution
                        nn.BatchNorm2d(embed_dims),
                        nn.SiLU(inplace=True),
                        nn.Conv2d(embed_dims, token_dims, kernel_size=(3, 3), stride=PCMStride[2], padding=(1, 1), groups=group),  # the 1st convolution
                    )

        self.PRM = PRM(img_size=img_size, kernel_size=kernel_size, downsample_ratio=downsample_ratios, dilations=self.dilations,
            in_chans=in_chans, embed_dim=embed_dims, share_weights=share_weights, op=op)
        self.outSize = self.outSize // downsample_ratios

        in_chans = self.PRM.out_chans
        if tokens_type == 'performer':
            self.attn = Token_performer(dim=in_chans, in_dim=token_dims, head_cnt=num_heads, kernel_ratio=0.5)
        elif tokens_type == 'performer_less':
            self.attn = None
            self.PCM = None
        elif tokens_type == 'transformer':
            self.attn = Token_transformer(dim=in_chans, in_dim=token_dims, num_heads=num_heads, mlp_ratio=mlp_ratio, drop=drop, attn_drop=attn_drop, drop_path=drop_path)
        elif tokens_type == 'window':
            self.attn = WindowTransformerBlock(in_dim=in_chans, out_dim=token_dims, input_resolution=(self.img_size//self.downsample_ratios, self.img_size//self.downsample_ratios),
                                            num_heads=num_heads, mlp_ratio=mlp_ratio, drop=drop,
                                            attn_drop=attn_drop, drop_path=drop_path, window_size=window_size, shift_size=0, relative_pos=relative_pos)

        self.num_patches = (img_size // 2) * (img_size // 2)  # there are 3 sfot split, stride are 4,2,2 seperately

    def forward(self, x):
        if len(x.shape) < 4:
            B, N, C  = x.shape
            n = int(np.sqrt(N))
            x = x.view(B, n, n, C).contiguous()
            x = x.permute(0, 3, 1, 2)
        if self.pool is not None:
            x = self.pool(x)
        shortcut = x
        PRM_x, _ = self.PRM(x)
        if self.tokens_type == 'window':

            B, N, C = PRM_x.shape
            H, W = self.img_size // self.downsample_ratios, self.img_size // self.downsample_ratios
            b, _, c = PRM_x.shape
            assert N == H*W
            x = self.attn.norm1(PRM_x)
            padding_td = (self.window_size - H % self.window_size) % self.window_size
            padding_top = padding_td // 2
            padding_down = padding_td - padding_top
            padding_lr = (self.window_size - W % self.window_size) % self.window_size
            padding_left = padding_lr // 2
            padding_right = padding_lr - padding_left
            x = x.view(B, H, W, C)
            if (padding_td + padding_lr) > 0:
                x = x.permute(0, 3, 1, 2)
                x = nn.functional.pad(x, (padding_left, padding_right, padding_top, padding_down))
                x = x.permute(0, 2, 3, 1).contiguous()

            x_windows = window_partition(x, self.window_size)
            x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
            attn_windows = self.attn.attn(x_windows, mask=self.attn.attn_mask)  # nW*B, window_size*window_size, C
            attn_windows = attn_windows.view(-1, self.window_size, self.window_size, self.token_dims)
            shifted_x = window_reverse(attn_windows, self.window_size, H+padding_td, W+padding_lr)  # B H' W' C
            x = shifted_x
            x = x[:, padding_top:padding_top+H, padding_left:padding_left+W, :]
            x = x.reshape(B, H * W, self.token_dims)

            convX = self.PCM(shortcut)
            convX = convX.permute(0, 2, 3, 1).view(*x.shape).contiguous()
            x = x + self.attn.drop_path(convX)
            x = x + self.attn.drop_path(self.attn.mlp(self.attn.norm2(x)))
        else:
            if self.attn is None:
                return PRM_x
            convX = self.PCM(shortcut)
            x = self.attn.attn(self.attn.norm1(PRM_x))
            convX = convX.permute(0, 2, 3, 1).view(*x.shape).contiguous()
            x = x + self.attn.drop_path(convX)
            x = x + self.attn.drop_path(self.attn.mlp(self.attn.norm2(x)))

        return x


class BasicLayer(nn.Module):
    def __init__(self,
                 img_size=224,
                 in_chans=3,
                 embed_dims=64,
                 token_dims=64,
                 downsample_ratios=4,
                 kernel_size=7,
                 RC_tokens_type='performer',
                 **kwargs):
        super(BasicLayer, self).__init__()

        # build RC
        if RC_tokens_type == "stem":
            self.RC = PatchEmbedding(inter_channel=token_dims // 2, out_channels=token_dims, img_size=img_size)
        elif RC_tokens_type == "stem":
            pass

    def forward(self, x):
        x = self.RC(x)

        for nc in self.NC:
            x = nc(x)
        return x


class ViTAEv2(nn.Module):
    def __init__(self, stages=4, num_classes=1000, NC_depth=[2, 2, 6, 2], **kwargs):
        super(ViTAEv2, self).__init__()
        self.NC_depth = NC_depth
        layers = []
        self.stages = stages
        depth = np.sum(self.NC_depth)
        for i in range(stages):
            startDpr = 0 if i == 0 else self.NC.depth[i - 1]
            layers.append(

            )

        self.num_classes = num_classes

    def forward_features(self, x):
        for layer in self.layers:
            x = layer(x)
        return torch

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x
