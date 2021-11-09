import torch.nn as nn
from .decode_head import BaseDecodeHead
from ...builder import HEADS, build_loss


class ESCHead(BaseDecodeHead):
    def __init__(self):
        super(ESCHead, self).__init__()

    def forward(self, inputs):
        pass
