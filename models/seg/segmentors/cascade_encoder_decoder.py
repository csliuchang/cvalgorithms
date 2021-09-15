# Copyright (c) OpenMMLab. All rights reserved.
from torch import nn

from ...utils import add_prefix, resize
from ... import builder
from ...builder import SEGMENTORS
from .encoder_decoder import EncoderDecoder


@SEGMENTORS.register_module()
class CascadeEncoderDecoder(EncoderDecoder):
    """Cascade Encoder Decoder segmentors.

    CascadeEncoderDecoder almost the same as EncoderDecoder, while decoders of
    CascadeEncoderDecoder are cascaded. The output of previous decoder_head
    will be the input of next decoder_head.
    """

    def __init__(self,
                 num_stages,
                 backbone,
                 decode_head,
                 neck=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        self.num_stages = num_stages
        super(CascadeEncoderDecoder, self).__init__(
            backbone=backbone,
            decode_head=decode_head,
            neck=neck,
            auxiliary_head=auxiliary_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)

    def _init_decode_head(self, decode_head):
        """Initialize ``decode_head``"""
        assert isinstance(decode_head, list)
        assert len(decode_head) == self.num_stages
        self.decode_head = nn.ModuleList()
        for i in range(self.num_stages):
            self.decode_head.append(builder.build_head(decode_head[i]))
        self.align_corners = self.decode_head[-1].align_corners
        self.num_classes = self.decode_head[-1].num_classes

    def encode_decode(self, img):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        x = self.extract_feat(img)
        out = self.decode_head[0].forward_infer(x)
        for i in range(1, self.num_stages):
            out = self.decode_head[i].forward_test(x, out)
        out = resize(
            input=out,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        return out

    def _decode_head_forward_train(self, x, ground_truth):
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()

        loss_decode, prev_outputs = self.decode_head[0].forward_train(
            x, ground_truth, return_val=True)

        losses.update(add_prefix(loss_decode, 'decode_0'))

        for i in range(1, self.num_stages):
            # forward test again, maybe unnecessary for most methods.
            loss_decode = self.decode_head[i].forward_train(
                x, prev_outputs, ground_truth)
            losses.update(add_prefix(loss_decode, f'decode_{i}'))

        return losses

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone and heads.

        Parameters
        ----------
        pretrained : str, optional
            Path to pre-trained weights.
            Defaults to None.
        """

        super(EncoderDecoder, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        for i in range(len(self.decode_head)):
            self.decode_head[i].init_weights()
        if self.with_auxiliary_head:
            if isinstance(self.auxiliary_head, nn.ModuleList):
                for aux_head in self.auxiliary_head:
                    aux_head.init_weights()
            else:
                self.auxiliary_head.init_weights()
