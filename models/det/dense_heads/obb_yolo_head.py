import torch
import torch.nn as nn

from models.builder import HEADS, build_loss
from specific import build_anchor_generator
from .base_dense_head import BaseDenseHead
from models.utils import normal_init, multi_apply, images_to_levels
from specific.bbox import build_assigner, build_sampler, build_bbox_coder


@HEADS.register_module()
class RYOLO(BaseDenseHead):
    def __init__(self):
        super(RYOLO, self).__init__()





