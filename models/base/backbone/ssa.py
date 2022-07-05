import torch.nn as nn


class Mlp(nn.Module):
    def __init__(self):
        super(Mlp, self).__init__()
        pass

    def _init_weights(self, m):
        pass

    def forward(self):
        pass


class ShuntedTransformer(nn.Module):
    def __init__(self, num_classes, depths, **kwargs):
        super(ShuntedTransformer, self).__init__()
        self.num_classes = num_classes
        self.depths = depths

    def forward(self, x):
        pass