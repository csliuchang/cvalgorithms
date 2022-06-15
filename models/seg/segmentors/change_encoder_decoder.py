from torch import nn
import torch
from ...utils import add_prefix, resize
from ... import builder
from ...builder import SEGMENTORS
from specific.siamese.builder import build_siamese_layer
from .encoder_decoder import EncoderDecoder


@SEGMENTORS.register_module()
class ChangeEncoderDecoder(EncoderDecoder):
    def __init__(self,
                 backbone,
                 decode_head,
                 neck=None,
                 auxiliary_head=None,
                 siamese_layer=None,
                 train_cfg=None,
                 test_cfg=None,
                 use_operation=None,
                 pretrained=None):
        super(ChangeEncoderDecoder, self).__init__(
            backbone=backbone,
            decode_head=decode_head,
            neck=neck,
            auxiliary_head=auxiliary_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained
        )
        self.use_operation = use_operation
        self.siamese_layer = build_siamese_layer(siamese_layer)

    def extract_feat(self, inputs):
        """Use Siamese Network Extract Features"""
        inputs_n, inputs_g = torch.chunk(inputs, 2, dim=1)
        features_n, features_g = self.backbone(inputs_n), self.backbone(inputs_g)
        if self.use_operation:
            return [torch.tanh(feature_n-feature_g) for feature_n, feature_g in zip(features_n, features_g)]
        else:
            return [features_n, features_g]

    def _init_decode_head(self, decode_head):
        """Initialize ``decode_head``"""
        self.decode_head = builder.build_head(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes

    def _init_auxiliary_head(self, auxiliary_head):
        """Initialize ``auxiliary_head``"""
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                self.auxiliary_head = nn.ModuleList()
                for head_cfg in auxiliary_head:
                    self.auxiliary_head.append(builder.build_head(head_cfg))
            else:
                self.auxiliary_head = builder.build_head(auxiliary_head)

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone and heads.

        Parameters
        ----------
        pretrained : str, optional
            Path to pre-trained weights.
            Defaults to None.
        """

        super(EncoderDecoder, self).init_weights(pretrained)
        if pretrained:
            self.backbone.init_weights(pretrained=pretrained)
        self.decode_head.init_weights()
        if self.with_auxiliary_head:
            if isinstance(self.auxiliary_head, nn.ModuleList):
                for aux_head in self.auxiliary_head:
                    aux_head.init_weights()
            else:
                self.auxiliary_head.init_weights()

    def encode_decode(self, inputs):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        x = self.extract_feat(inputs)
        out = self._decode_head_forward_infer(x)
        out = resize(
            input=out,
            size=inputs.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        return out

    def _decode_head_forward_train(self, x, ground_truth):
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        x_n, x_g = x
        # share weight decode
        features_n = self.decode_head(x_n, return_feat=True)
        features_g = self.decode_head(x_g, return_feat=True)
        # siamese fuse
        changes = self.siamese_layer(features_n, features_g)
        loss_decode = self.decode_head.losses(changes, ground_truth)
        losses.update(add_prefix(loss_decode, 'decode'))
        return losses

    def _decode_head_forward_infer(self, x):
        """Run forward function and calculate loss for decode head in
        inference."""
        x_n, x_g = x
        features_n = self.decode_head(x_n, return_feat=True)
        features_g = self.decode_head(x_g, return_feat=True)
        changes = self.siamese_layer(features_n, features_g)
        # print(changes[changes!=0])
        return changes

    def _auxiliary_head_forward_train(self, x, ground_truth):
        """Run forward function and calculate loss for auxiliary head in
        training."""
        losses = dict()
        if isinstance(self.auxiliary_head, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head):
                loss_aux = aux_head.forward_train(x, ground_truth)
                losses.update(add_prefix(loss_aux, f'aux_{idx}'))
        else:
            loss_aux = self.auxiliary_head.forward_train(
                x, ground_truth)
            losses.update(add_prefix(loss_aux, 'aux'))

        return losses

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
        ground_truth = ground_truth['gt_masks']
        x = self.extract_feat(inputs)

        losses = dict()
        loss_decode = self._decode_head_forward_train(x, ground_truth)
        losses.update(loss_decode)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(
                x, ground_truth)
            losses.update(loss_aux)

        return losses

    def forward_infer(self, inputs):
        changes = self.encode_decode(inputs)
        return changes









