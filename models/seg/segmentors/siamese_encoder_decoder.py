from torch import nn
import torch
from ...utils import add_prefix, resize
from ... import builder
from ...builder import SEGMENTORS, build_siamese_layer
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
                 pretrained=None):
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
        inputs_g, inputs_n = torch.chunk(inputs, 2, dim=1)
        features_n, features_g = self.backbone(inputs_n), self.backbone(inputs_g)
        return [features_n, features_g]





