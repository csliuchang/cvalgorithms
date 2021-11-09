# Copyright (c) OpenMMLab. All rights reserved.
import torch
from models.base.blocks.comm_blocks import ConvModule
from torch import nn as nn
from torch.nn import functional as F
from ...utils.weight_init import constant_init


class SelfAttentionBlock(nn.Module):
    """General self-attention block/non-local block.

    Please refer to https://arxiv.org/abs/1706.03762 for details about key,
    query and value.

    Args:
        key_in_channels (int): Input channels of key feature.
        query_in_channels (int): Input channels of query feature.
        channels (int): Output channels of key/query transform.
        out_channels (int): Output channels.
        share_key_query (bool): Whether share projection weight between key
            and query projection.
        query_downsample (nn.Module): Query downsample module.
        key_downsample (nn.Module): Key downsample module.
        key_query_num_convs (int): Number of convs for key/query projection.
        value_num_convs (int): Number of convs for value projection.
        matmul_norm (bool): Whether normalize attention map with sqrt of
            channels
        with_out (bool): Whether use out projection.
        conv_cfg (dict|None): Config of conv layers.
        norm_cfg (dict|None): Config of norm layers.
        act_cfg (dict|None): Config of activation layers.
    """

    def __init__(self, key_in_channels, query_in_channels, channels,
                 out_channels, share_key_query, query_downsample,
                 key_downsample, key_query_num_convs, value_out_num_convs,
                 key_query_norm, value_out_norm, matmul_norm, with_out,
                 conv_cfg, norm_cfg, act_cfg):
        super(SelfAttentionBlock, self).__init__()
        if share_key_query:
            assert key_in_channels == query_in_channels
        self.key_in_channels = key_in_channels
        self.query_in_channels = query_in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.share_key_query = share_key_query
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.key_project = self.build_project(
            key_in_channels,
            channels,
            num_convs=key_query_num_convs,
            use_conv_module=key_query_norm,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        if share_key_query:
            self.query_project = self.key_project
        else:
            self.query_project = self.build_project(
                query_in_channels,
                channels,
                num_convs=key_query_num_convs,
                use_conv_module=key_query_norm,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
        self.value_project = self.build_project(
            key_in_channels,
            channels if with_out else out_channels,
            num_convs=value_out_num_convs,
            use_conv_module=value_out_norm,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        if with_out:
            self.out_project = self.build_project(
                channels,
                out_channels,
                num_convs=value_out_num_convs,
                use_conv_module=value_out_norm,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
        else:
            self.out_project = None

        self.query_downsample = query_downsample
        self.key_downsample = key_downsample
        self.matmul_norm = matmul_norm

        self.init_weights()

    def init_weights(self):
        """Initialize weight of later layer."""
        if self.out_project is not None:
            if not isinstance(self.out_project, ConvModule):
                constant_init(self.out_project, 0)

    def build_project(self, in_channels, channels, num_convs, use_conv_module,
                      conv_cfg, norm_cfg, act_cfg):
        """Build projection layer for key/query/value/out."""
        if use_conv_module:
            convs = [
                ConvModule(
                    in_channels,
                    channels,
                    1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg)
            ]
            for _ in range(num_convs - 1):
                convs.append(
                    ConvModule(
                        channels,
                        channels,
                        1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg))
        else:
            convs = [nn.Conv2d(in_channels, channels, 1)]
            for _ in range(num_convs - 1):
                convs.append(nn.Conv2d(channels, channels, 1))
        if len(convs) > 1:
            convs = nn.Sequential(*convs)
        else:
            convs = convs[0]
        return convs

    def forward(self, query_feats, key_feats):
        """Forward function."""
        batch_size = query_feats.size(0)
        query = self.query_project(query_feats)
        if self.query_downsample is not None:
            query = self.query_downsample(query)
        query = query.reshape(*query.shape[:2], -1)
        query = query.permute(0, 2, 1).contiguous()

        key = self.key_project(key_feats)
        value = self.value_project(key_feats)
        if self.key_downsample is not None:
            key = self.key_downsample(key)
            value = self.key_downsample(value)
        key = key.reshape(*key.shape[:2], -1)
        value = value.reshape(*value.shape[:2], -1)
        value = value.permute(0, 2, 1).contiguous()

        sim_map = torch.matmul(query, key)
        if self.matmul_norm:
            sim_map = (self.channels ** -.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)

        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.reshape(batch_size, -1, *query_feats.shape[2:])
        if self.out_project is not None:
            context = self.out_project(context)
        return context


class BasicAttentionModule(nn.Module):
    """
    Basic Attention Module for Siamese Network fusion
    """

    def __init__(self, in_chn, stride=8, activation=nn.ReLU):
        super(BasicAttentionModule, self).__init__()
        self.in_chn = in_chn
        self.key_channel = self.in_chn // 8
        self.act = activation
        self.stride = stride
        self.pool = nn.AvgPool2d(self.stride)
        # q, k, v
        self.q_conv = nn.Conv2d(in_channels=in_chn, out_channels=in_chn // 8, kernel_size=1)
        self.k_conv = nn.Conv2d(in_channels=in_chn, out_channels=in_chn // 8, kernel_size=1)
        self.v_conv = nn.Conv2d(in_channels=in_chn, out_channels=in_chn, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, features):
        x = self.pool(features)
        B, C, W, H = x.size()
        proj_query = self.q_conv(x).view(B, -1, W * H).permute(0, 2, 1)  # B * (W*H) * N
        proj_key = self.k_conv(x).view(B, -1, W * H)  #
        proj_value = self.v_conv(x).view(B, -1, W * H)
        energy = torch.matmul(proj_query, proj_key)
        energy = (self.key_channel ** -.5) * energy
        attention = self.softmax(energy)
        out = torch.matmul(proj_value, attention.permute(0, 2, 1))
        out = out.view(B, C, W, H)
        out = F.interpolate(out, [W * self.stride, H * self.stride])
        out = out + features
        return out


class PyramidAttentionModule(nn.Module):
    def __init__(self, in_channels, out_channels, sizes=None, stride=1):
        super(PyramidAttentionModule, self).__init__()
        group = len(sizes)
        self.stages = nn.ModuleList(_PAMBlock(in_channels, out_channels // 8, out_channels, scale=size,
                                              stride=stride) for size in sizes)
        self.conv_bn = ConvModule(in_channels * group, out_channels, 3, 1, 1)

    def forward(self, feats):
        priors = [stage(feats) for stage in self.stages]

        # concat
        context = []
        for i in range(0, len(priors)):
            context += [priors[i]]
        outputs = self.conv_bn(torch.cat(context, 1))

        return outputs


class _PAMBlock(nn.Module):
    """
    The basic implementation for self-attention block/non-local block
    Input/Output:
        N * C  *  H  *  (2*W)
    Parameters:
        in_channels       : the dimension of the input feature map
        key_channels      : the dimension after the key/query transform
        value_channels    : the dimension after the value transform
        scale             : choose the scale to partition the input feature maps
        ds                : downsampling scale
    """

    def __init__(self, in_channels, key_channels, value_channels, scale=1, stride=1):
        super(_PAMBlock, self).__init__()
        self.scale = scale
        self.stride = stride
        self.pool = nn.AvgPool2d(self.stride)
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.value_channels = value_channels

        self.f_key = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.key_channels)
        )
        self.f_query = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.key_channels)
        )
        self.f_value = nn.Conv2d(in_channels=self.in_channels, out_channels=self.value_channels,
                                 kernel_size=1, stride=1, padding=0)

    def forward(self, inputs):
        x = inputs
        if self.stride != 1:
            x = self.pool(inputs)

        # input_shape: b, c, h, 2w
        b, c, h, w = x.size(0), x.size(1), x.size(2), x.size(3) // 2
        local_y, local_x = [], []
        step_h, step_w = h // self.scale, w // self.scale
        for i in range(0, self.scale):
            for j in range(0, self.scale):
                start_x, start_y = i * step_h, j * step_w
                end_x, end_y = min(start_x + step_h, h), min(start_y + step_w, w)
                if i == (self.scale - 1):
                    end_x = h
                if j == (self.scale - 1):
                    end_y = w
                local_x += [start_x, end_x]
                local_y += [start_y, end_y]

        value, query, key = self.f_value(x), self.f_query(x), self.f_key(x)
        value = torch.stack([value[:, :, :, :w], value[:, :, :, w:]], 4)
        query = torch.stack([query[:, :, :, :w], query[:, :, :, w:]], 4)
        key = torch.stack([key[:, :, :, :w], key[:, :, :, w:]], 4)

        local_block_cnt = 2 * self.scale * self.scale

        v_list = [value[:, :, local_x[i]:local_x[i + 1], local_y[i]:local_y[i + 1]] for i in
                  range(0, local_block_cnt, 2)]
        v_locals = torch.cat(v_list, dim=0)
        q_list = [query[:, :, local_x[i]:local_x[i + 1], local_y[i]:local_y[i + 1]] for i in
                  range(0, local_block_cnt, 2)]
        q_locals = torch.cat(q_list, dim=0)
        k_list = [key[:, :, local_x[i]:local_x[i + 1], local_y[i]:local_y[i + 1]] for i in range(0, local_block_cnt, 2)]
        k_locals = torch.cat(k_list, dim=0)
        # print(v_locals.shape)
        context_locals = self._self_att(v_locals, q_locals, k_locals)

        context_list = []
        for i in range(0, self.scale):
            row_tmp = []
            for j in range(0, self.scale):
                left = b * (j + i * self.scale)
                right = b * (j + i * self.scale) + b
                tmp = context_locals[left:right]
                row_tmp.append(tmp)
            context_list.append(torch.cat(row_tmp, 3))

        context = torch.cat(context_list, 2)
        context = torch.cat([context[:, :, :, :, 0], context[:, :, :, :, 1]], 3)

        if self.stride != 1:
            context = F.interpolate(context, [h * self.stride, 2 * w * self.stride])

        return context

    def _self_att(self, value_local, query_local, key_local):
        batch_size_new = value_local.size(0)
        h_local, w_local = value_local.size(2), value_local.size(3)
        value_local = value_local.contiguous().view(batch_size_new, self.value_channels, -1)

        query_local = query_local.contiguous().view(batch_size_new, self.key_channels, -1)
        query_local = query_local.permute(0, 2, 1)
        key_local = key_local.contiguous().view(batch_size_new, self.key_channels, -1)

        sim_map = torch.matmul(query_local, key_local)  # batch matrix multiplication
        sim_map = (self.key_channels ** -.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)

        context_local = torch.matmul(value_local, sim_map.permute(0, 2, 1))
        # context_local = context_local.permute(0, 2, 1).contiguous()
        context_local = context_local.view(batch_size_new, self.value_channels, h_local, w_local, 2)
        return context_local
