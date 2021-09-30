from typing import List
import torch
from torchvision.ops import boxes as box_ops
from torchvision.ops import nms
from opts.rnms.rnms_wrapper import batched_rnms


def batched_nms(
        boxes: torch.Tensor, scores: torch.Tensor, idxs: torch.Tensor, iou_threshold: float
):
    """
    Same as torchvision.ops.boxes.batched_nms, but safer.
    """
    assert boxes.shape[-1] == 4
    # TODO may need better strategy.
    # Investigate after having a fully-cuda NMS op.
    if len(boxes) < 40000:
        # fp16 does not have enough range for batched NMS
        return box_ops.batched_nms(boxes.float(), scores, idxs, iou_threshold)

    result_mask = scores.new_zeros(scores.size(), dtype=torch.bool)
    for id in torch.jit.annotate(List[int], torch.unique(idxs).cpu().tolist()):
        mask = (idxs == id).nonzero().view(-1)
        keep = nms(boxes[mask], scores[mask], iou_threshold)
        result_mask[mask[keep]] = True
    keep = result_mask.nonzero().view(-1)
    keep = keep[scores[keep].argsort(descending=True)]
    return keep


def multiclass_rnms(multi_bboxes,
                    multi_scores,
                    score_thr,
                    nms_cfg,
                    max_num=-1,
                    score_factors=None):
    """NMS for multi-class bboxes.

    Args:
        multi_bboxes (Tensor): shape (n, #class*5) or (n, 5)
        multi_scores (Tensor): shape (n, #class), where the last column
            contains scores of the background class, but this will be ignored.
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms_thr (float): NMS IoU threshold
        max_num (int): if there are more than max_num bboxes after NMS,
            only top max_num will be kept.
        score_factors (Tensor): The factors multiplied to scores before
            applying NMS

    Returns:
        tuple: (bboxes, labels), tensors of shape (k, 6) and (k, 1). Labels
            are 0-based.
    """
    num_classes = multi_scores.size(1) - 1
    # exclude background category
    if multi_bboxes.shape[1] > 5:
        bboxes = multi_bboxes.view(multi_scores.size(0), -1, 5)
    else:
        bboxes = multi_bboxes[:, None].expand(-1, num_classes, 5)
    scores = multi_scores[:, :-1]

    # filter out boxes with low scores
    valid_mask = scores > score_thr
    bboxes = bboxes[valid_mask]
    if score_factors is not None:
        scores = scores * score_factors[:, None]
    scores = scores[valid_mask]
    # labels = valid_mask.nonzero()[:, 1]
    labels = torch.nonzero(valid_mask, as_tuple=True)[1]
    if bboxes.numel() == 0:
        bboxes = multi_bboxes.new_zeros((0, 6))
        labels = multi_bboxes.new_zeros((0,), dtype=torch.long)
        return bboxes, labels

    dets, keep = batched_rnms(bboxes, scores, labels, nms_cfg)

    if max_num > 0:
        dets = dets[:max_num]
        keep = keep[:max_num]

    return dets, labels[keep]


def multiclass_nms(multi_bboxes,
                   multi_scores,
                   score_thr,
                   nms_cfg,
                   max_num=-1,
                   score_factors=None):
    """NMS for multi-class bboxes.

    Args:
        multi_bboxes (Tensor): shape (n, #class*5) or (n, 5)
        multi_scores (Tensor): shape (n, #class), where the last column
            contains scores of the background class, but this will be ignored.
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms_thr (float): NMS IoU threshold
        max_num (int): if there are more than max_num bboxes after NMS,
            only top max_num will be kept.
        score_factors (Tensor): The factors multiplied to scores before
            applying NMS

    Returns:
        tuple: (bboxes, labels), tensors of shape (k, 6) and (k, 1). Labels
            are 0-based.
    """
    num_classes = multi_scores.size(1) - 1
    # exclude background category
    if multi_bboxes.shape[1] > 4:
        bboxes = multi_bboxes.view(multi_scores.size(0), -1, 4)
    else:
        bboxes = multi_bboxes[:, None].expand(-1, num_classes, 4)
    scores = multi_scores[:, :-1]

    # filter out boxes with low scores
    valid_mask = scores > score_thr
    bboxes = bboxes[valid_mask]
    if score_factors is not None:
        scores = scores * score_factors[:, None]
    scores = scores[valid_mask]
    # labels = valid_mask.nonzero()[:, 1]
    # topk_idxs = torch.nonzero(valid_mask, as_tuple=True)[0]
    #
    # scores, idxs = scores.sort(descending=True)
    # topk_idxs = topk_idxs[idxs]
    # labels = topk_idxs % num_classes
    labels = torch.nonzero(valid_mask, as_tuple=True)[1]
    if bboxes.numel() == 0:
        bboxes = multi_bboxes.new_zeros((0, 5))
        labels = multi_bboxes.new_zeros((0,), dtype=torch.long)
        return bboxes, labels
    keep = box_ops.batched_nms(bboxes, scores, labels.float(), nms_cfg.iou_thr)

    if max_num > 0:
        keep = keep[:max_num]

        bboxes = bboxes[keep]
        scores = scores[keep].reshape(-1, 1)
        labels = labels[keep]

    return torch.cat([bboxes, scores], dim=1), labels


# def multiclass_nms(multi_bboxes,
#                    multi_scores,
#                    score_thr,
#                    nms_cfg,
#                    max_num=-1,
#                    score_factors=None,
#                    return_inds=False):
#     """NMS for multi-class bboxes.
#
#     Args:
#         multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
#         multi_scores (Tensor): shape (n, #class), where the last column
#             contains scores of the background class, but this will be ignored.
#         score_thr (float): bbox threshold, bboxes with scores lower than it
#             will not be considered.
#         nms_thr (float): NMS IoU threshold
#         max_num (int, optional): if there are more than max_num bboxes after
#             NMS, only top max_num will be kept. Default to -1.
#         score_factors (Tensor, optional): The factors multiplied to scores
#             before applying NMS. Default to None.
#         return_inds (bool, optional): Whether return the indices of kept
#             bboxes. Default to False.
#
#     Returns:
#         tuple: (dets, labels, indices (optional)), tensors of shape (k, 5),
#             (k), and (k). Dets are boxes with scores. Labels are 0-based.
#     """
#     num_classes = multi_scores.size(1) - 1
#     # exclude background category
#     if multi_bboxes.shape[1] > 4:
#         bboxes = multi_bboxes.view(multi_scores.size(0), -1, 4)
#     else:
#         bboxes = multi_bboxes[:, None].expand(
#             multi_scores.size(0), num_classes, 4)
#
#     scores = multi_scores[:, :-1]
#
#     labels = torch.arange(num_classes, dtype=torch.long)
#     labels = labels.view(1, -1).expand_as(scores)
#
#     bboxes = bboxes.reshape(-1, 4)
#     scores = scores.reshape(-1)
#     labels = labels.reshape(-1)
#
#     if not torch.onnx.is_in_onnx_export():
#         # NonZero not supported  in TensorRT
#         # remove low scoring boxes
#         valid_mask = scores > score_thr
#     # multiply score_factor after threshold to preserve more bboxes, improve
#     # mAP by 1% for YOLOv3
#     if score_factors is not None:
#         # expand the shape to match original shape of score
#         score_factors = score_factors.view(-1, 1).expand(
#             multi_scores.size(0), num_classes)
#         score_factors = score_factors.reshape(-1)
#         scores = scores * score_factors
#
#     if not torch.onnx.is_in_onnx_export():
#         # NonZero not supported  in TensorRT
#         inds = valid_mask.nonzero(as_tuple=False).squeeze(1)
#         bboxes, scores, labels = bboxes[inds], scores[inds], labels[inds]
#     else:
#         # TensorRT NMS plugin has invalid output filled with -1
#         # add dummy data to make detection output correct.
#         bboxes = torch.cat([bboxes, bboxes.new_zeros(1, 4)], dim=0)
#         scores = torch.cat([scores, scores.new_zeros(1)], dim=0)
#         labels = torch.cat([labels, labels.new_zeros(1)], dim=0)
#
#     if bboxes.numel() == 0:
#         if torch.onnx.is_in_onnx_export():
#             raise RuntimeError('[ONNX Error] Can not record NMS '
#                                'as it has not been executed this time')
#         dets = torch.cat([bboxes, scores[:, None]], -1)
#         if return_inds:
#             return dets, labels, inds
#         else:
#             return dets, labels
#
#     dets, keep = batched_nms(bboxes, scores, labels, nms_cfg)
#
#     if max_num > 0:
#         dets = dets[:max_num]
#         keep = keep[:max_num]
#
#     if return_inds:
#         return dets, labels[keep], inds[keep]
#     else:
#         return dets, labels[keep]
