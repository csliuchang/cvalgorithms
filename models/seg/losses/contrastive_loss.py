import torch.nn as nn
import torch
from ...builder import LOSSES


@LOSSES.register_module()
class BBContrastiveLoss(nn.Module):
    """
    batch-balanced contrastive loss
    """

    def __init__(self, margin=2.0):
        super(BBContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, distance, label):
        label[label == 255] = 1
        mask = (label != 255).float()
        distance = distance * mask
        pos_sum = torch.sum((label == 1).float()) + 0.0001
        neg_sum = torch.sum((label == -1).float()) + 0.0001
        # pos loss
        loss_1 = torch.sum(
            (1 + label) / 2 * torch.pow(distance, 2)
        ) / pos_sum
        # neg loss
        loss_2 = torch.sum(
            (1 - label) / 2 * mask *
            torch.pow(torch.clamp(self.margin - distance, min=0.0), 2)
        ) / neg_sum
        loss = loss_1 + loss_2
        return loss
