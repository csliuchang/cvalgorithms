from torch import nn
import torch
import torch.nn.functional as F
from ...utils import add_prefix, resize
from ... import builder
from ...builder import SEGMENTORS
from .encoder_decoder import EncoderDecoder


@SEGMENTORS.register_module()
class SiameseEncoderDecoder(EncoderDecoder):
    def __init__(self,
                 backbone,
                 decode_head,
                 neck=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 use_operation=None,
                 pretrained=None):
        self.use_operation = use_operation
        super(SiameseEncoderDecoder, self).__init__(
            backbone=backbone,
            decode_head=decode_head,
            neck=neck,
            auxiliary_head=auxiliary_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained
        )

    def extract_feat(self, inputs):
        """Use Siamese Network Extract Features"""
        inputs_n, inputs_g = torch.chunk(inputs, 2, dim=1)
        features_n, features_g = self.backbone(inputs_n), self.backbone(inputs_g)
        if self.use_operation:
            return [torch .tanh(feature_n-feature_g) for feature_n, feature_g in zip(features_n, features_g)]
        else:
            return [features_n, features_g]









