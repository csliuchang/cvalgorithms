import torch
import math
import torch.nn as nn
import numpy as np
from models.base.utils import to_2tuple, trunc_normal_, DropPath
from models.base.transformer import window_partition, window_unpartition, Mlp, \
    add_decomposed_rel_pos, PatchEmbed, get_abs_pos
from models.builder import BACKBONES


def attention_pool(x, pool, norm=None):
    # (B, H, W, C) -> (B, C, H, W)
    x = x.permute(0, 3, 1, 2)
    x = pool(x)
    # (B, C, H1, W1) -> (B, H1, W1, C)
    x = x.permute(0, 2, 3, 1)
    if norm:
        x = norm(x)

    return x


class MultiScaleAttention(nn.Module):
    """Multiscale Multi-head Attention block."""

    def __init__(
            self,
            dim,
            dim_out,
            num_heads,
            qkv_bias=True,
            norm_layer=nn.LayerNorm,
            pool_kernel=(3, 3),
            stride_q=1,
            stride_kv=1,
            residual_pooling=True,
            window_size=0,
            use_rel_pos=False,
            rel_pos_zero_init=True,
            input_size=None,
    ):
        """
        Args:
            dim (int): Number of input channels.
            dim_out (int): Number of output channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool:  If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            pool_kernel (tuple): kernel size for qkv pooling layers.
            stride_q (int): stride size for q pooling layer.
            stride_kv (int): stride size for kv pooling layer.
            residual_pooling (bool): If true, enable residual pooling.
            use_rel_pos (bool): If True, add relative postional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            input_size (int or None): Input resolution.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim_out // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim_out * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim_out, dim_out)

        # qkv pooling
        pool_padding = [k // 2 for k in pool_kernel]
        dim_conv = dim_out // num_heads
        self.pool_q = nn.Conv2d(
            dim_conv,
            dim_conv,
            pool_kernel,
            stride=stride_q,
            padding=pool_padding,
            groups=dim_conv,
            bias=False,
        )
        self.norm_q = norm_layer(dim_conv)
        self.pool_k = nn.Conv2d(
            dim_conv,
            dim_conv,
            pool_kernel,
            stride=stride_kv,
            padding=pool_padding,
            groups=dim_conv,
            bias=False,
        )
        self.norm_k = norm_layer(dim_conv)
        self.pool_v = nn.Conv2d(
            dim_conv,
            dim_conv,
            pool_kernel,
            stride=stride_kv,
            padding=pool_padding,
            groups=dim_conv,
            bias=False,
        )
        self.norm_v = norm_layer(dim_conv)

        self.window_size = window_size
        if window_size:
            self.q_win_size = window_size // stride_q
            self.kv_win_size = window_size // stride_kv
        self.residual_pooling = residual_pooling

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            # initialize relative positional embeddings
            assert input_size[0] == input_size[1]
            size = input_size[0]
            rel_dim = 2 * max(size // stride_q, size // stride_kv) - 1
            self.rel_pos_h = nn.Parameter(torch.zeros(rel_dim, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(rel_dim, head_dim))

            if not rel_pos_zero_init:
                trunc_normal_(self.rel_pos_h, std=0.02)
                trunc_normal_(self.rel_pos_w, std=0.02)

    def forward(self, x):
        B, H, W, _ = x.shape
        # qkv with shape (3, B, nHead, H, W, C)
        qkv = self.qkv(x).reshape(B, H, W, 3, self.num_heads, -1).permute(3, 0, 4, 1, 2, 5)
        # q, k, v with shape (B * nHead, H, W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, H, W, -1).unbind(0)

        q = attention_pool(q, self.pool_q, self.norm_q)
        k = attention_pool(k, self.pool_k, self.norm_k)
        v = attention_pool(v, self.pool_v, self.norm_v)

        ori_q = q
        if self.window_size:
            q, q_hw_pad = window_partition(q, self.q_win_size)
            k, kv_hw_pad = window_partition(k, self.kv_win_size)
            v, _ = window_partition(v, self.kv_win_size)
            q_hw = (self.q_win_size, self.q_win_size)
            kv_hw = (self.kv_win_size, self.kv_win_size)
        else:
            q_hw = q.shape[1:3]
            kv_hw = k.shape[1:3]

        q = q.view(q.shape[0], np.prod(q_hw), -1)
        k = k.view(k.shape[0], np.prod(kv_hw), -1)
        v = v.view(v.shape[0], np.prod(kv_hw), -1)

        attn = (q * self.scale) @ k.transpose(-2, -1)

        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, q_hw, kv_hw)

        attn = attn.softmax(dim=-1)
        x = attn @ v

        x = x.view(x.shape[0], q_hw[0], q_hw[1], -1)

        if self.window_size:
            x = window_unpartition(x, self.q_win_size, q_hw_pad, ori_q.shape[1:3])

        if self.residual_pooling:
            x += ori_q

        H, W = x.shape[1], x.shape[2]
        x = x.view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = self.proj(x)

        return x


class MultiScaleBlock(nn.Module):
    """Multiscale Transformer blocks"""

    def __init__(
            self,
            dim,
            dim_out,
            num_heads,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_path=0.0,
            norm_layer=nn.LayerNorm,
            act_layer=nn.GELU,
            qkv_pool_kernel=(3, 3),
            stride_q=1,
            stride_kv=1,
            residual_pooling=True,
            window_size=0,
            use_rel_pos=False,
            rel_pos_zero_init=True,
            input_size=None,
    ):
        """
        Args:
            dim (int): Number of input channels.
            dim_out (int): Number of output channels.
            num_heads (int): Number of attention heads in the MViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            drop_path (float): Stochastic depth rate.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            qkv_pool_kernel (tuple): kernel size for qkv pooling layers.
            stride_q (int): stride size for q pooling layer.
            stride_kv (int): stride size for kv pooling layer.
            residual_pooling (bool): If true, enable residual pooling.
            window_size (int): Window size for window attention blocks. If it equals 0, then not
                use window attention.
            use_rel_pos (bool): If True, add relative postional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            input_size (int or None): Input resolution.
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = MultiScaleAttention(
            dim,
            dim_out,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            norm_layer=norm_layer,
            pool_kernel=qkv_pool_kernel,
            stride_q=stride_q,
            stride_kv=stride_kv,
            residual_pooling=residual_pooling,
            window_size=window_size,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim_out)
        self.mlp = Mlp(
            in_features=dim_out,
            hidden_features=int(dim_out * mlp_ratio),
            out_features=dim_out,
            act_layer=act_layer,
        )

        if dim != dim_out:
            self.proj = nn.Linear(dim, dim_out)

        if stride_q > 1:
            kernel_skip = stride_q + 1
            padding_skip = int(kernel_skip // 2)
            self.pool_skip = nn.MaxPool2d(kernel_skip, stride_q, padding_skip, ceil_mode=False)

    def forward(self, x):
        x_norm = self.norm1(x)
        x_block = self.attn(x_norm)

        if hasattr(self, "proj"):
            x = self.proj(x_norm)
        if hasattr(self, "pool_skip"):
            x = attention_pool(x, self.pool_skip)

        x = x + self.drop_path(x_block)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


def round_width(width, multiplier, min_width=1, divisor=1, verbose=False):
    if not multiplier:
        return width
    width *= multiplier
    min_width = min_width or divisor
    if verbose:
        pass
    width_out = max(min_width, int(width + divisor / 2) // divisor * divisor)
    if width_out < 0.9 * width:
        width_out += divisor
    return int(width_out)


# @BACKBONES.register_module()
class MViT(nn.Module):
    def __init__(self,
                 img_size=512, in_channels=3, patch_kernel=(7, 7), patch_stride=(4, 4), patch_padding=(3, 3),
                 num_heads=1, qkv_pool_kernel=(3, 3), stride_kv=4, window_size=56, mlp_ratio=4.0, qkv_bias=True,
                 embed_dim=96, depth=10, drop_path_rate=0.2, last_block_indexes=(0, 2, 7, 9),
                 norm_layer=nn.LayerNorm, residual_pooling=True, use_rel_pos=True, rel_pos_zero_init=True,
                 out_features=("scale2", "scale3", "scale4", "scale5"), use_abs_pos=False, out_levels=None, **kwargs):
        super(MViT, self).__init__()
        self.patch_embed = PatchEmbed(
            kernel_size=patch_kernel,
            stride=patch_stride,
            padding=patch_padding,
            in_chans=in_channels,
            embed_dim=embed_dim,
        )

        if use_abs_pos:
            # Initialize absoluate positional embedding with pretrain image size.
            pass
        else:
            self.pos_embed = None

        dim_out = embed_dim

        stage = 2
        stride = patch_stride[0]
        self._out_feature_strides = {}
        self._out_feature_channels = {}

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        input_size = (img_size // patch_stride[0], img_size // patch_stride[1])
        self.blocks = nn.ModuleList()
        for i in range(depth):
            if i == last_block_indexes[1] or i == last_block_indexes[2]:
                stride_kv_ = stride_kv * 2
            else:
                stride_kv_ = stride_kv
            window_size_ = 0 if i in last_block_indexes[1:] else window_size
            block = MultiScaleBlock(
                dim=embed_dim,
                dim_out=dim_out,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                qkv_pool_kernel=qkv_pool_kernel,
                stride_q=2 if i - 1 in last_block_indexes else 1,
                stride_kv=stride_kv_,
                residual_pooling=residual_pooling,
                window_size=window_size_,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                input_size=input_size,
            )
            self.blocks.append(block)

            embed_dim = dim_out
            if i in last_block_indexes:
                name = f'scale{stage}'
                if name in out_features:
                    self._out_feature_channels[name] = dim_out
                    self._out_feature_strides[name] = stride
                    self.add_module(f"{name}_norm", norm_layer(dim_out))

                dim_out *= 2
                num_heads *= 2
                stride_kv = max(stride_kv // 2, 1)
                stride *= 2
                stage += 1
            if i - 1 in last_block_indexes:
                window_size = window_size // 2
                input_size = [s // 2 for s in input_size]

        self._out_features = out_features
        self._last_block_indexes = last_block_indexes

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=0.02)

        self.out_levels = out_levels

    def init_weights(self, pretrained=None):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        if pretrained is not None:
            state_dict = torch.load(pretrained)['model_state']
            model_dict = self.state_dict()
            incorrect_shapes = []
            for k in list(state_dict.keys()):
                if k in model_dict:
                    model_param = model_dict[k]
                    shape_model = tuple(model_param.shape)
                    shape_checkpoint = tuple(state_dict[k].shape)
                    if shape_model != shape_checkpoint:
                        incorrect_shapes.append((k, shape_checkpoint, shape_model))
                        state_dict.pop(k)
            self.load_state_dict(state_dict, strict=False)
            pass

    def forward(self, x):
        x = self.patch_embed(x)

        if self.pos_embed is not None:
            x = x + get_abs_pos(self.pos_embed, self.pretrain_use_cls_token, x.shape[1:3])

        outputs = []
        stage = 2
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in self._last_block_indexes:
                name = f"scale{stage}"
                if name in self._out_features:
                    x_out = getattr(self, f"{name}_norm")(x)
                    outputs.append(x_out.permute(0, 3, 1, 2))
                stage += 1
        if self.out_levels is not None:
            outputs = [outputs[level] for level in self.out_levels]
        return outputs


if __name__ == "__main__":
    _root = "./datasets"
    model = MViT()
    a = torch.randn(1, 3, 512, 512)
    outputs = model(a)
    pass
