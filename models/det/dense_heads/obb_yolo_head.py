import torch

from models.builder import HEADS, build_loss
from .base_dense_head import BaseDenseHead


@HEADS.register_module()
class RYOLO(BaseDenseHead):
    def __init__(self):
        super(RYOLO, self).__init__()





