from abc import ABCMeta, abstractmethod
import torch.nn as nn


class BaseDenseHead(nn.Module, metaclass=ABCMeta):
    def __init__(self):
        super(BaseDenseHead, self).__init__()


    @abstractmethod
    def loss(self, **kwargs):
        pass

    @abstractmethod
    def get_bboxes(self, **kwargs):
        pass

    def forward_train(self, inputs, ground_truth, **kwargs):
        outs = self(inputs)
        loss_inputs = outs + ground_truth + inputs
        losses = self.loss(*loss_inputs)
        return losses

