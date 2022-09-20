from torch import nn
import torch
from ...utils import add_prefix, resize
from ... import builder
from ...builder import SEGMENTORS
from models.specific.siamese.builder import build_siamese_layer
from .siamese_encoder_decoder import SiameseEncoderDecoder
from ..losses import BatchContrastiveLoss
import torch.nn.functional as F
import numpy as np


class TinyChangeClassifier(SiameseEncoderDecoder):
    def __init__(self,
                 backbone,
                 decode_head,
                 neck=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 export_feature=None,
                 with_constant_head=True,
                 **kwargs):
        super(TinyChangeClassifier, self).__init__(
            backbone=backbone,
            decode_head=decode_head,
            neck=neck,
            auxiliary_head=auxiliary_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained
        )

    def extract_feat(self, inputs):
        pass

    def forward_train(self, inputs, ground_truth):
        assert type(ground_truth) is dict, "Wrong ground truth type ! "
        ground_truth = ground_truth.get('gt_mask', None)
        pass
