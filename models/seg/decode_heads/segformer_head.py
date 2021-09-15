# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from models.base.blocks.conv_module import ConvModule

from ...builder import HEADS
from .decode_head import BaseDecodeHead

from ...utils import resize
from models.base.blocks.conv_module import CondConv2D


@HEADS.register_module()
class SegformerHead(BaseDecodeHead):
    """The all mlp Head of segformer.

    This head is the implementation of
    `Segformer <https://arxiv.org/abs/2105.15203>` _.

    Args:
        interpolate_mode: The interpolate mode of MLP head upsample operation.
            Default: 'bilinear'.
    """

    def __init__(self, interpolate_mode='bilinear', **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)

        self.interpolate_mode = interpolate_mode
        num_inputs = len(self.in_channels)

        assert num_inputs == len(self.in_index)

        self.convs = nn.ModuleList()
        for i in range(num_inputs):
            self.convs.append(
                CondConv2D(
                    in_channels=self.in_channels[i],
                    out_channels=self.head_width,
                    kernel_size=3,
                    stride=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))

        self.fusion_conv = CondConv2D(
            in_channels=self.head_width * num_inputs,
            out_channels=self.head_width,
            kernel_size=3,
            norm_cfg=self.norm_cfg)

        self.conv_seg = CondConv2D(self.head_width, self.num_classes, kernel_size=1, act_cfg=None)

    def forward(self, inputs):
        # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        inputs = self._transform_inputs(inputs)
        outs = []
        for idx in range(len(inputs)):
            x = inputs[idx]
            conv = self.convs[idx]
            outs.append(
                resize(
                    input=conv(x),
                    size=inputs[0].shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners))

        out = self.fusion_conv(torch.cat(outs, dim=1))

        out = self.cls_seg(out)

        return out

    def losses(self, seg_logit, seg_label):
        loss = dict()
        seg_logit = resize(
            input=seg_logit,
            size=seg_label.shape[1:],
            mode='bilinear',
            align_corners=self.align_corners)
        loss_1 = self.loss(seg_logit, seg_label)
        loss["loss"] = loss_1
        return loss
