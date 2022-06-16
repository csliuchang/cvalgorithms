import torch.nn as nn
import torch
from ...builder import LOSSES


@LOSSES.register_module()
class BatchContrastiveLoss(nn.Module):
    """
    pass
    """

    def __init__(self, margin=2.0):
        super(BatchContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, distance, label):
        losses = dict()
        label_change = torch.zeros_like(label).to(label.device)
        label_change[label == 1] = -1  # not similar
        label_change[label == 0] = 1
        mask = (label != 255)
        distance = distance * mask
        pos_num = torch.sum((label_change == 1).float()) + 0.0001
        neg_num = torch.sum((label_change == -1).float()) + 0.0001

        loss_1 = torch.sum((1 + label_change) / 2 * torch.pow(distance, 2)) / pos_num
        loss_2 = torch.sum((1 - label_change) / 2 * mask *
                           torch.pow(torch.clamp(self.margin - distance, min=0.0), 2)
                           ) / neg_num
        losses['cnt_loss'] = loss_1 + loss_2
        return losses

