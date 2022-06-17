import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from ...utils import weight_reduce_loss
from ...builder import LOSSES
from itertools import filterfalse


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
                 ignore_label=None):
        super(CrossEntropyLoss, self).__init__()
        assert (use_sigmoid is False) or (use_mask is False)
        self.use_sigmoid = use_sigmoid
        self.use_mask = use_mask
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        self.ignore_label = ignore_label

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


@LOSSES.register_module()
class DiveCELoss(nn.Module):
    def __init__(self, div_factor=0.7, ignore_lb=255, *args, **kwargs):
        super(DiveCELoss, self).__init__()
        self.div_factor = div_factor
        self.ignore_lb = ignore_lb
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_lb, reduction='none')

    def forward(self, logits, labels):
        loss_div_value = []
        N, C, H, W = logits.size()
        loss = self.criteria(logits, labels).view(-1)
        loss_div = torch.mul(torch.max(loss), self.div_factor)
        index = torch.nonzero(loss > loss_div, as_tuple=False)
        return torch.mean(loss[index])


@LOSSES.register_module()
class OhemCELoss(nn.Module):
    def __init__(self, thresh, n_min, ignore_lb=255, *args, **kwargs):
        super(OhemCELoss, self).__init__()
        self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float)).cuda()
        self.n_min = n_min
        self.ignore_lb = ignore_lb
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_lb, reduction='none')

    def forward(self, logits, labels):
        N, C, H, W = logits.size()
        loss = self.criteria(logits, labels).view(-1)
        loss, _ = torch.sort(loss, descending=True)
        if len(loss) <= self.n_min:
            loss = loss
        elif loss[self.n_min] > self.thresh:
            loss = loss[loss > self.thresh]
        else:
            loss = loss[:self.n_min]
        return torch.mean(loss)


@LOSSES.register_module()
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


def _masked_ignore(y_pred: torch.Tensor, y_true: torch.Tensor, ignore_index: int):
    # usually used for BCE-like loss
    y_pred = y_pred.reshape((-1,))
    y_true = y_true.reshape((-1,))
    valid = y_true != ignore_index
    y_true = y_true.masked_select(valid).float()
    y_pred = y_pred.masked_select(valid).float()
    return y_pred, y_true


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


def select(y_pred: torch.Tensor, y_true: torch.Tensor, ignore_index: int):
    assert y_pred.ndim == 4 and y_true.ndim == 3
    c = y_pred.size(1)
    y_pred = y_pred.permute(0, 2, 3, 1).reshape(-1, c)
    y_true = y_true.reshape(-1)

    valid = y_true != ignore_index

    y_pred = y_pred[valid, :]
    y_true = y_true[valid]
    return y_pred, y_true


def dice_coeff(y_pred, y_true, weights: torch.Tensor, smooth_value: float = 1.0, ):
    y_pred = y_pred[:, weights]
    y_true = y_true[:, weights]
    inter = torch.sum(y_pred * y_true, dim=0)
    z = y_pred.sum(dim=0) + y_true.sum(dim=0) + smooth_value

    return ((2 * inter + smooth_value) / z).mean()


@LOSSES.register_module()
class CriterionOhemDSN(nn.Module):
    '''
    DSN : We need to consider two supervision for the model.
    '''

    def __init__(self, ignore_index=255, thresh=0.7, min_kept=100000, use_weight=True, reduction='mean'):
        super(CriterionOhemDSN, self).__init__()
        self.ignore_index = ignore_index
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, reduction=reduction)

    def forward(self, preds, target):
        h, w = target.size(1), target.size(2)

        if len(preds) >= 2:
            scale_pred = F.interpolate(input=preds, size=(h, w), mode='bilinear', align_corners=True)
            loss1 = self.criterion(scale_pred, target)
            loss2 = lovasz_softmax(F.softmax(scale_pred, dim=1), target, ignore=self.ignore_index)

            scale_pred = F.interpolate(input=preds, size=(h, w), mode='bilinear', align_corners=True)
            loss3 = self.criterion(scale_pred, target)
            return loss1 + loss2 * 0.8 + loss3 * 0.1
        else:
            scale_pred = F.interpolate(input=preds[0], size=(h, w), mode='bilinear', align_corners=True)
            loss1 = self.criterion(scale_pred, target)
            loss2 = lovasz_softmax(F.softmax(scale_pred, dim=1), target, ignore=self.ignore_index)

            return loss1 + loss2 * 0.8


def lovasz_softmax(probas, labels, classes='present', per_image=False, ignore=None):
    """
    Multi-class Lovasz-Softmax loss
      probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1).
              Interpreted as binary (sigmoid) output with outputs of size [B, H, W].
      labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
      per_image: compute the loss per image instead of per batch
      ignore: void class labels
    """
    if per_image:
        loss = mean(lovasz_softmax_flat(*flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), ignore), classes=classes)
                    for prob, lab in zip(probas, labels))
    else:
        loss = lovasz_softmax_flat(*flatten_probas(probas, labels, ignore), classes=classes)
    return loss


def lovasz_softmax_flat(probas, labels, classes='present'):
    """
    Multi-class Lovasz-Softmax loss
      probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [P] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
    """
    if probas.numel() == 0:
        # only void pixels, the gradients should be 0
        return probas * 0.
    C = probas.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
    for c in class_to_sum:
        fg = (labels == c).float()  # foreground for class c
        if (classes is 'present' and fg.sum() == 0):
            continue
        if C == 1:
            if len(classes) > 1:
                raise ValueError('Sigmoid output possible only with 1 class')
            class_pred = probas[:, 0]
        else:
            class_pred = probas[:, c]
        errors = (Variable(fg) - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, Variable(lovasz_grad(fg_sorted))))
    return mean(losses)


def isnan(x):
    return x != x


def mean(l, ignore_nan=False, empty=0):
    """
    nanmean compatible with generators.
    """
    l = iter(l)
    if ignore_nan:
        l = filterfalse(isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def flatten_probas(probas, labels, ignore=None):
    """
    Flattens predictions in the batch
    """
    if probas.dim() == 3:
        # assumes output of a sigmoid layer
        B, H, W = probas.size()
        probas = probas.view(B, 1, H, W)
    B, C, H, W = probas.size()
    probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C
    labels = labels.view(-1)
    if ignore is None:
        return probas, labels
    valid = (labels != ignore)
    vprobas = probas[valid.nonzero().squeeze()]
    vlabels = labels[valid]
    return vprobas, vlabels
