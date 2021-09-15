# Modified by Chang Liu
# Contact: liuchang@deepsight.ai
import torch
import torch.nn.functional as F
from torch import nn
from models.utils import weight_reduce_loss
from models.builder import LOSSES


def _sigmoid_focal_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        alpha: float = -1,
        gamma: float = 2,
        reduction: str = "none",
) -> torch.Tensor:
    """
    Loss used in RRetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        Loss tensor with the reduction option applied.
    """
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


def sigmoid_focal_loss(pred,
                       target,
                       weight=None,
                       gamma=2.0,
                       alpha=0.25,
                       reduction='mean',
                       avg_factor=None):
    """A warpper _sigmoid_focal_loss

    Parameters
    ----------
    pred : torch.Tensor
        The prediction with shape (N, C), C is the number
        of classes.
    target : torch.Tensor
        The learning label of the prediction.
    weight : torch.Tensor, optional
        Sample-wise loss weight.
    gamma : float, optional
        The gamma for calculating the modulating
        factor. Defaults to 2.0.
    alpha : float, optional
        A balanced form for Focal Loss.
        Defaults to 0.25.
    reduction : str, optional
        The method used to reduce the loss into
        a scalar. Defaults to 'mean'. Options are "none", "mean" and "sum".
    avg_factor : int, optional
        Average factor that is used to average
        the loss. Defaults to None.
    """
    loss = _sigmoid_focal_loss(pred, target, alpha, gamma, 'none')
    # print(f"-----loss is --------------{sum(loss)}--------------------------------")

    if weight is not None:
        if weight.shape != loss.shape:
            if weight.size(0) == loss.size(0):
                weight = weight.view(-1, 1)
            else:
                assert weight.numel() == loss.numel()
                weight = weight.view(loss.size(0), -1)
        assert weight.ndim == loss.ndim
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


@LOSSES.register_module()
class FocalLoss(nn.Module):

    def __init__(self,
                 use_sigmoid=True,
                 gamma=2.0,
                 alpha=0.25,
                 reduction='mean',
                 loss_weight=1.0):
        """`Focal Loss <https://arxiv.org/abs/1708.02002>`_

        Parameters
        ----------
        use_sigmoid : bool, optional
            Whether to the prediction is
            used for sigmoid or softmax. Defaults to True.
        gamma : float, optional
            The gamma for calculating the modulating
            factor. Defaults to 2.0.
        alpha : float, optional
            A balanced form for Focal Loss.
            Defaults to 0.25.
        reduction : str, optional
            The method used to reduce the loss into
            a scalar. Defaults to 'mean'. Options are "none", "mean" and
            "sum".
        loss_weight : float, optional
            Weight of loss. Defaults to 1.0.
        """
        super(FocalLoss, self).__init__()
        assert use_sigmoid is True, 'Only sigmoid focal loss supported now.'
        self.use_sigmoid = use_sigmoid
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Parameters
        ----------
        pred : torch.Tensor
            The prediction.
        target : torch.Tensor
            The learning label of the prediction.
        weight : torch.Tensor, optional
            The weight of loss for each
            prediction. Defaults to None.
        avg_factor : int, optional
            Average factor that is used to average
            the loss. Defaults to None.
        reduction_override : str, optional
            The reduction method used to
            override the original reduction method of the loss.
            Options are "none", "mean" and "sum".

        Returns
        -------
        torch.Tensor
            The calculated loss
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.use_sigmoid:
            loss_cls = self.loss_weight * sigmoid_focal_loss(
                pred,
                target,
                weight,
                gamma=self.gamma,
                alpha=self.alpha,
                reduction=reduction,
                avg_factor=avg_factor)
        else:
            raise NotImplementedError
        return loss_cls


sigmoid_focal_loss_jit = torch.jit.script(
    _sigmoid_focal_loss
)  # type: torch.jit.ScriptModule
