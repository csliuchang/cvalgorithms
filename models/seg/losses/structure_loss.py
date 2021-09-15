import numpy as np

import torch
import torch.nn.functional as F
from ...builder import LOSSES
import torch.nn as nn


@LOSSES.register_module()
class StructureLoss(nn.Module):
    def __init__(self):
        super(StructureLoss, self).__init__()

    def __call__(self, pred, mask):
        mask = torch.tensor(mask, dtype=torch.float32)
        weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
        wbce = F.cross_entropy_with_logits(pred, mask, reduce='none')
        wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

        pred = torch.sigmoid(pred)
        inter = ((pred * mask) * weit).sum(dim=(2, 3))
        union = ((pred + mask) * weit).sum(dim=(2, 3))
        wiou = 1 - (inter + 1) / (union - inter + 1)

        return (wbce + wiou).mean()
