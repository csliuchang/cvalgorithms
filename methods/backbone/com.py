from typing import List, Optional

import torchvision
from torch import Tensor, reshape, stack

from torch.nn import (
    Conv2d,
    InstanceNorm2d,
    Module,
    ModuleList,
    PReLU,
    Sequential,
    Upsample,
)


def _get_backbone(
        bkbn_name, pretrained, output_layer_bkbn, freeze_backbone,
) -> ModuleList:
    """
    Get version model from torchvision
    """
    model = getattr(torchvision.models, bkbn_name)(
        pretrained=pretrained
    ).features

    derived_model = ModuleList([])

    for name, layer in model.named_children():
        derived_model.append(layer)
        if name == output_layer_bkbn:
            break

    # freeze
    if freeze_backbone:
        for param in derived_model.parameters():
            param.requires_grad = False

    return derived_model
