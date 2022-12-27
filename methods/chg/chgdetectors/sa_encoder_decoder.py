import torch
import torch.nn as nn
from .base import BaseChanger
from deepcv2.ds_network import builder
from deepcv2.ds_network.builder import CHGDETECTORS
from deepcv2.common.ops.warppers import resize
from deepcv2.common.common_func import add_prefix


@CHGDETECTORS.register_module()
class SiamEncoderDecoder(BaseChanger):
    """
    Siam Encoder Decoder change detectors
        SiamEncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """
    def __init__(self,
                 backbone,
                 decode_head,
                 neck=None,
                 sep_channel=2,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 num_classes=2,
                 pretrained=None,
                 ):
        self.num_classes = num_classes
        self.sep_channel = sep_channel
        super(SiamEncoderDecoder, self).__init__(
        )
        if pretrained is not None:
            assert backbone.get('pretrained') is None, \
                'both backbone and segmentor set pretrained weight'
            backbone.pretrained = pretrained
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)
        self._init_decode_head(decode_head)
        self._init_auxiliary_head(auxiliary_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        assert self.with_decode_head

    def encode_decode(self, inputs):
        x = self.extract_feat(inputs)
        out = self._decode_head_forward_infer(x)
        out = resize(
            input=out,
            size=inputs.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        return out
        pass

    def _decode_head_forward_infer(self, x):
        """Run forward function and calculate loss for decode head in
        inference."""
        seg_logits = self.decode_head.forward_infer(x)
        return seg_logits

    def _init_decode_head(self, decode_head):
        self.decode_head = builder.build_head(decode_head,
                                              update_args=dict(num_classes=self.num_classes))
        self.align_corners = self.decode_head.align_corners

    def _init_auxiliary_head(self, auxiliary_head):
        """Initialize ``auxiliary_head``"""
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                self.auxiliary_head = nn.ModuleList()
                for head_cfg in auxiliary_head:
                    self.auxiliary_head.append(builder.build_head(head_cfg,
                                                                  update_args=dict(num_classes=self.num_classes)))
            else:
                self.auxiliary_head = builder.build_head(auxiliary_head)

    def forward_train(self, inputs, ground_truth):
        """Forward function for training.

        Parameters
        ----------
        inputs : Tensor
            Input images.
        ground_truth : Tensor
            Semantic segmentation masks
            used if the architecture supports semantic segmentation task.

        Returns
        -------
        dict[str, Tensor]
            a dictionary of loss components
        """
        x = self.extract_feat(inputs)
        losses = dict()
        gt_masks = ground_truth['mask'].to(inputs.device)
        loss_decode = self._decode_head_forward_train(x, gt_masks)
        losses.update(loss_decode)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(
                x, gt_masks)
            losses.update(loss_aux)

        return losses

    def forward_infer(self, inputs, **kwargs):
        seg_logit = self.encode_decode(inputs)
        if self.num_classes > 1:
            seg_probs = torch.softmax(seg_logit, dim=1)
        else:
            seg_probs = torch.sigmoid(seg_logit)

        return seg_probs

    def extract_feat(self, inputs):
        img1, img2 = torch.split(inputs, self.sep_channel, dim=1)
        x1, x2 = self.backbone(img1), self.backbone(img2)
        if self.with_neck:
            x = self.neck(x1, x2)
        else:
            raise ValueError("NECK is needed for siamese network type")
        return x

    def _decode_head_forward_train(self, x, ground_truth):
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        loss_decode = self.decode_head.forward_train(x, ground_truth)

        losses.update(add_prefix(loss_decode, 'decode'))
        return losses

    def _auxiliary_head_forward_train(self, x, ground_truth):
        """Run forward function and calculate loss for auxiliary head in
        training."""
        losses = dict()
        if isinstance(self.auxiliary_head, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head):
                loss_aux = aux_head.forward_train(x, ground_truth)
                losses.update(add_prefix(loss_aux, f'aux_{idx}'))
        else:
            loss_aux = self.auxiliary_head.forward_train(x, ground_truth)
            losses.update(add_prefix(loss_aux, 'aux'))

        return losses

