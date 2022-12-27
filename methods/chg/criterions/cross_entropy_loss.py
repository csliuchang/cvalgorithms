import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import cv2
from deepcv2.ds_network.builder import LOSSES

from deepcv2.ds_network.utils.loss_ops import weight_reduce_loss


def cross_entropy(pred,
                  label,
                  weight=None,
                  class_weight=None,
                  reduction='mean',
                  avg_factor=None,
                  ignore_index=-100):
    """The wrapper function for :func:`F.cross_entropy`"""
    if ignore_index is None:
        ignore_index = -100
    loss = F.cross_entropy(
        pred,
        label,
        weight=class_weight,
        reduction='none',
        ignore_index=ignore_index)

    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

    return loss


def _expand_onehot_labels(labels, label_weights, label_channels):
    """Expand onehot labels to match the size of prediction."""
    bin_labels = labels.new_full((labels.size(0), label_channels), 0)
    inds = torch.nonzero(labels >= 1, as_tuple=False).squeeze()
    if inds.numel() > 0:
        bin_labels[inds, labels[inds] - 1] = 1
    if label_weights is None:
        bin_label_weights = None
    else:
        bin_label_weights = label_weights.view(-1, 1).expand(
            label_weights.size(0), label_channels)
    return bin_labels, bin_label_weights


def binary_cross_entropy(pred,
                         label,
                         weight=None,
                         reduction='mean',
                         avg_factor=None,
                         class_weight=None):
    """Calculate the binary CrossEntropy loss.

    Parameters
    ----------
    pred : torch.Tensor
        The prediction with shape (N, 1).
    label : torch.Tensor
        The learning label of the prediction.
    weight : torch.Tensor, optional
        Sample-wise loss weight.
    reduction : str, optional
        The method used to reduce the loss.
        Options are "none", "mean" and "sum".
    avg_factor : int, optional
        Average factor that is used to average
        the loss. Defaults to None.
    class_weight : list[float], optional
        The weight for each class.

    Returns
    -------
    torch.Tensor
        The calculated loss
    """
    if pred.dim() != label.dim():
        label, weight = _expand_onehot_labels(label, weight, pred.size(-1))

    if weight is not None:
        weight = weight.float()
    loss = F.binary_cross_entropy_with_logits(
        pred, label.float(), weight=class_weight, reduction='none')
    loss = weight_reduce_loss(
        loss, weight, reduction=reduction, avg_factor=avg_factor)

    return loss


def mask_cross_entropy(pred,
                       target,
                       label,
                       reduction='mean',
                       avg_factor=None,
                       class_weight=None):
    """Calculate the CrossEntropy loss for masks.

    Parameters
    ----------
    pred : torch.Tensor
        The prediction with shape (N, C), C is the number
        of classes.
    target : torch.Tensor
        The learning label of the prediction.
    label : torch.Tensor
        ``label`` indicates the class label of the mask'
        corresponding object. This will be used to select the mask in the
        of the class which the object belongs to when the mask prediction
        if not class-agnostic.
    reduction : str, optional
        The method used to reduce the loss.
        Options are "none", "mean" and "sum".
    avg_factor : int, optional
        Average factor that is used to average
        the loss. Defaults to None.
    class_weight : list[float], optional
        The weight for each class.

    Returns
    -------
    torch.Tensor
        The calculated loss
    """
    assert reduction == 'mean' and avg_factor is None
    num_rois = pred.size()[0]
    inds = torch.arange(0, num_rois, dtype=torch.long, device=pred.device)
    pred_slice = pred[inds, label].squeeze(1)
    return F.binary_cross_entropy_with_logits(
        pred_slice, target, weight=class_weight, reduction='mean')[None]


def flatten_binary_scores(scores, labels, ignore=None):
    scores = scores.view(-1)
    labels = labels.view(-1)
    if ignore is None:
        return scores, labels
    valid = (labels != ignore)
    vscores = scores[valid]
    vlabels = labels[valid]
    return vscores, vlabels


class StableBCELoss(nn.Module):
    def __init__(self, **kwargs):
        super(StableBCELoss, self).__init__()

    def forward(self, pred, label, ignore=None, **kwargs):
        pred, label = flatten_binary_scores(pred, label, ignore)
        neg_abs = - pred.abs()
        loss = pred.clamp(min=0) - pred * Variable(label.float()) \
               + (1 + neg_abs.exp()).log()
        return loss.mean()


@LOSSES.register_module()
class CrossEntropyLoss(nn.Module):
    """CrossEntropyLoss.

    Parameters
    ----------
    use_sigmoid : bool, optional
        Whether the prediction uses sigmoid
        of softmax. Defaults to False.
    use_mask : bool, optional
        Whether to use mask cross entropy loss.
        Defaults to False.
    reduction : str, optional
        . Defaults to 'mean'.
        Options are "none", "mean" and "sum".
    class_weight : list[float], optional
        Weight of each class.
        Defaults to None.
    loss_weight : float, optional
        Weight of the loss. Defaults to 1.0.
    """

    def __init__(self,
                 use_sigmoid=False,
                 use_mask=False,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0,
                 ignore_label=None,
                 loss_name='loss_ce'):
        super(CrossEntropyLoss, self).__init__()
        assert (use_sigmoid is False) or (use_mask is False)
        self.use_sigmoid = use_sigmoid
        self.use_mask = use_mask
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        self.ignore_label = ignore_label
        self.loss_name = loss_name
        if self.use_sigmoid:
            self.cls_criterion = binary_cross_entropy
        elif self.use_mask:
            self.cls_criterion = mask_cross_entropy
        else:
            self.cls_criterion = cross_entropy

    def forward(self,
                pred,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function."""
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.class_weight is not None:
            class_weight = pred.new_tensor(self.class_weight)
        else:
            class_weight = None

        loss_cls = self.loss_weight * self.cls_criterion(
            pred,
            label,
            weight,
            class_weight=class_weight,
            reduction=reduction,
            avg_factor=avg_factor,
            ignore_index=self.ignore_label)

        return loss_cls


class GaussianCrossEntropyLoss(nn.Module):
    """CrossEntropyLoss.

    Parameters
    ----------
    use_sigmoid : bool, optional
        Whether the prediction uses sigmoid
        of softmax. Defaults to False.
    use_mask : bool, optional
        Whether to use mask cross entropy loss.
        Defaults to False.
    reduction : str, optional
        . Defaults to 'mean'.
        Options are "none", "mean" and "sum".
    class_weight : list[float], optional
        Weight of each class.
        Defaults to None.
    loss_weight : float, optional
        Weight of the loss. Defaults to 1.0.
    """

    def __init__(self,
                 use_sigmoid=False,
                 use_mask=False,
                 gamma=1,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0,
                 ignore_label=None,
                 loss_name='loss_gce'):
        super(GaussianCrossEntropyLoss, self).__init__()
        assert (use_sigmoid is False) or (use_mask is False)
        self.use_sigmoid = use_sigmoid
        self.use_mask = use_mask
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        self.ignore_label = ignore_label
        self.loss_name = loss_name
        if self.use_sigmoid:
            self.cls_criterion = binary_cross_entropy
        elif self.use_mask:
            self.cls_criterion = mask_cross_entropy
        else:
            self.cls_criterion = cross_entropy

        self.gamma = gamma

    def forward(self,
                pred,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function."""
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.class_weight is not None:
            class_weight = pred.new_tensor(self.class_weight)
        else:
            class_weight = None

        loss_cls = self.loss_weight * self.cls_criterion(
            pred,
            label,
            gaussian_transform(label, self.gamma),
            class_weight=class_weight,
            reduction=reduction,
            avg_factor=avg_factor,
            ignore_index=self.ignore_label)

        return loss_cls


class TopKLoss(CrossEntropyLoss):
    """
    Network has to have NO LINEARITY!
    """
    def __init__(self,
                 use_sigmoid=False,
                 use_mask=False,
                 class_weight=None,
                 loss_weight=1.0,
                 k=10):
        self.k = k
        super(TopKLoss, self).__init__(
            use_sigmoid=use_sigmoid,
            use_mask=use_mask,
            class_weight=class_weight,
            loss_weight=loss_weight,
            reduction='none')

    def forward(self,
                pred,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        ce_loss = super(TopKLoss, self).forward(pred, label)
        num_voxels = np.prod(ce_loss.shape, dtype=np.int64)
        loss, _ = torch.topk(ce_loss.view((-1,)), int(num_voxels * self.k / 100), sorted=False)
        return loss.mean()


class OhemCELoss(nn.Module):
    def __init__(self, thresh, n_min, ignore_lb=255, *args, **kwargs):
        super(OhemCELoss, self).__init__()
        self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float))
        self.n_min = n_min
        self.ignore_lb = ignore_lb
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_lb, reduction='none')

    def forward(self, logits, labels):
        N, C, H, W = logits.size()
        self.thresh = self.thresh.to(logits.device)
        loss = self.criteria(logits, labels).view(-1)
        loss, _ = torch.sort(loss, descending=True)
        if len(loss) <= self.n_min:
            loss = loss
        elif loss[self.n_min] > self.thresh:
            loss = loss[loss > self.thresh]
        else:
            loss = loss[:self.n_min]
        return torch.mean(loss)


class BCEDiceLoss(nn.Module):
    """
    bce + loss for change detection
    """

    def __init__(self,
                 loss_weight=1.0,
                 ignore_label=-1):
        super(BCEDiceLoss, self).__init__()
        self.loss_weight = loss_weight
        self.ignore_label = ignore_label
        self.bce_criterion = binary_cross_entropy_with_logits
        self.dice_criterion = dice_loss_with_logits

    def forward(self,
                pred,
                label):
        """Forward function."""
        # print(len(pred[pred != 0]))
        loss_bce = self.loss_weight * self.bce_criterion(pred, label, ignore_index=self.ignore_label)
        loss_dice = self.loss_weight * self.dice_criterion(pred, label, ignore_index=self.ignore_label)
        loss_cls = loss_bce + loss_dice
        return loss_cls


def binary_cross_entropy_with_logits(output: torch.Tensor, target: torch.Tensor, reduction: str = 'none',
                                     ignore_index: int = 255):
    output, target = _masked_ignore(output, target, ignore_index)
    output = torch.sigmoid(output)
    return F.binary_cross_entropy(output, target, reduction=reduction)


def dice_loss_with_logits(y_pred: torch.Tensor, y_true: torch.Tensor,
                          smooth_value: float = 1.0,
                          ignore_index: int = 255,
                          ignore_channel: int = -1):
    c = y_pred.size(1)
    y_pred, y_true = select(y_pred, y_true, ignore_index)
    weight = torch.as_tensor([True] * c, device=y_pred.device)
    if c == 1:
        y_prob = y_pred.sigmoid()
        return 1. - dice_coeff(y_prob, y_true.reshape(-1, 1), weight, smooth_value)
    else:
        y_prob = y_pred.softmax(dim=1)
        y_true = F.one_hot(y_true, num_classes=c)
        if ignore_channel != -1:
            weight[ignore_channel] = False

        return 1. - dice_coeff(y_prob, y_true, weight, smooth_value)


def dice_coeff(y_pred, y_true, weights: torch.Tensor, smooth_value: float = 1.0, ):
    y_pred = y_pred[:, weights]
    y_true = y_true[:, weights]
    inter = torch.sum(y_pred * y_true, dim=0)
    z = y_pred.sum(dim=0) + y_true.sum(dim=0) + smooth_value

    return ((2 * inter + smooth_value) / z).mean()


def select(y_pred: torch.Tensor, y_true: torch.Tensor, ignore_index: int):
    assert y_pred.ndim == 4 and y_true.ndim == 3
    c = y_pred.size(1)
    y_pred = y_pred.permute(0, 2, 3, 1).reshape(-1, c)
    y_true = y_true.reshape(-1)

    valid = y_true != ignore_index

    y_pred = y_pred[valid, :]
    y_true = y_true[valid]
    return y_pred, y_true


def _masked_ignore(y_pred: torch.Tensor, y_true: torch.Tensor, ignore_index: int):
    # usually used for BCE-like loss
    y_pred = y_pred.reshape((-1,))
    y_true = y_true.reshape((-1,))
    valid = y_true != ignore_index
    y_true = y_true.masked_select(valid).float()
    y_pred = y_pred.masked_select(valid).float()
    return y_pred, y_true

def gaussian_transform(batch_mask, gamma=1):

    c, h, w = batch_mask.shape
    dst_trf = torch.zeros_like(batch_mask, dtype=torch.float32)
    np_mask = batch_mask.cpu().numpy()

    for b, mask in enumerate(np_mask):
        num_labels, labels = cv2.connectedComponents((mask * 255.0).astype(np.uint8), connectivity=8)
        for idx in range(1, num_labels):
            mask_roi = np.zeros((h, w))
            k = labels == idx
            mask_roi[k] = 1
            dst_trf_roi = cv2.GaussianBlur(mask_roi, (3, 3), gamma) + 1
            dst_trf[b] += torch.tensor(dst_trf_roi,dtype=torch.float32, device=batch_mask.device)

    return dst_trf

def cross_entropy_trans(input, target, weight=None, reduction='mean',ignore_index=255):
    """
    logSoftmax_with_loss
    :param input: torch.Tensor, N*C*H*W
    :param target: torch.Tensor, N*1*H*W,/ N*H*W
    :param weight: torch.Tensor, C
    :return: torch.Tensor [0]
    """
    target = target.long()
    if target.dim() == 4:
        target = torch.squeeze(target, dim=1)
    if input.shape[-1] != target.shape[-1]:
        input = F.interpolate(input, size=target.shape[1:], mode='bilinear',align_corners=True)

    return F.cross_entropy(input=input, target=target, weight=weight,
                           ignore_index=ignore_index, reduction=reduction)