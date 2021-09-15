import torch

from ..builder import BBOX_ASSIGNERS
from ..iou_calculators import build_iou_calculator
from models.utils import box_xyxy_to_cxcywh
from .assign_result import AssignResult
from .base_assigner import BaseAssigner

__all__ = ['UniformAssigner']


@BBOX_ASSIGNERS.register_module()
class UniformAssigner(BaseAssigner):
    """Uniform Matching between the anchors and gt boxes, which can achieve
    balance in positive anchors, and gt_bboxes_ignore was not considered for
    now.

    Args:
        pos_ignore_thr (float): the threshold to ignore positive anchors
        neg_ignore_thr (float): the threshold to ignore negative anchors
        match_times(int): Number of positive anchors for each gt box.
           Default 4.
        iou_calculator (dict): iou_calculator config
    """

    def __init__(self,
                 pos_ignore_thr,
                 neg_ignore_thr,
                 match_times=4,
                 iou_calculator=dict(type='BboxOverlaps2D')):
        self.match_times = match_times
        self.pos_ignore_thr = pos_ignore_thr
        self.neg_ignore_thr = neg_ignore_thr
        self.iou_calculator = build_iou_calculator(iou_calculator)

    def assign(self, bboxed_pred, anchors, gt_bboxes, gt_bboxes_ignore=None, gt_labels=None):
        num_gts, num_bboxes = gt_bboxes.size(0), bboxed_pred.size(0)

        # default assign
        assigned_gt_inds = bboxed_pred.new_full((num_bboxes,),
                                                0,
                                                dtype=torch.long)
        assigned_labels = bboxed_pred.new_full((num_bboxes,),
                                               -1,
                                               dtype=torch.long)

        # process none data
        if num_gts == 0 or num_bboxes == 0:
            # No ground truth or bboxes, return empty assigment
            if num_gts == 0:
                # No ground truth, assign all to background
                assigned_gt_inds[:] = 0
            assigned_result = AssignResult(
                num_gts, assigned_gt_inds, None, labels=assigned_labels)
            assigned_result.set_extra_property(
                'pos_idx', bboxed_pred.new_empty(0, dtype=torch.bool)
            )
            assigned_result.set_extra_property('pos_predicted_boxes',
                                               bboxed_pred.new_empty((0, 4)))
            assigned_result.set_extra_property('target_boxes',
                                               bboxed_pred.new_empty((0, 4)))
            return assigned_result

        # Compute the l1 cost between boxes
        # Note that we use anchors and predict boxes both
        cost_bbox = torch.cdist(
            box_xyxy_to_cxcywh(bboxed_pred),
            box_xyxy_to_cxcywh(gt_bboxes),
            p=1
        )
        cost_bbox_anchors = torch.cdist(
            box_xyxy_to_cxcywh(anchors), box_xyxy_to_cxcywh(gt_bboxes),
            p=1
        )

        # TODO:  mmdetection think CPU is diff from GPU, for commpute speed, we use GPU
        assert cost_bbox.device == cost_bbox_anchors.device
        C = cost_bbox
        C1 = cost_bbox_anchors

        index = torch.topk(
            C,
            k=self.match_times,
            dim=0,
            largest=False
        )[1]

        # self.match_times x n
        index1 = torch.topk(C1, k=self.match_times, dim=0, largest=False)[1]
        # (self.match_times*2) x n
        indexes = torch.cat((index, index1), dim=1).reshape(-1).to(bboxed_pred.device)

        pred_overlaps = self.iou_calculator(bboxed_pred, gt_bboxes)
        anchor_overlaps = self.iou_calculator(anchors, gt_bboxes)
        pred_max_overlaps, _ = pred_overlaps.max(dim=1)
        anchor_max_overlaps, _ = anchor_overlaps.max(dim=0)

        # 3.Compute the ignore indexes of positive sample use anchors
        # and predict boxes
        ignore_idx = pred_max_overlaps > self.neg_ignore_thr
        assigned_gt_inds[ignore_idx] = -1

        # 4. Compute the ignore indexes of positive sample use anchors
        # and predict boxes
        pos_gt_index = torch.arange(
            0, C1.size(1), device=bboxed_pred.device).repeat(self.match_times * 2
                                                             )
        pos_ious = anchor_overlaps[indexes, pos_gt_index]
        pos_ignore_idx = pos_ious < self.pos_ignore_thr

        pos_gt_index_with_ignore = pos_gt_index + 1
        pos_gt_index_with_ignore[pos_ignore_idx] = -1
        assigned_gt_inds[indexes] = pos_gt_index_with_ignore

        if gt_labels is not None:
            assigned_labels = assigned_gt_inds.new_full((num_bboxes,), -1)
            pos_inds = torch.nonzero(assigned_gt_inds > 0, as_tuple=False).squeeze()
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[
                    assigned_gt_inds[pos_inds] - 1
                    ]
        else:
            assigned_labels = None

        assign_result = AssignResult(
            num_gts,
            assigned_gt_inds,
            anchor_max_overlaps,
            labels=assigned_labels
        )
        assign_result.set_extra_property('pos_idx', ~pos_ignore_idx)
        assign_result.set_extra_property('pos_predicted_boxes',
                                         bboxed_pred[indexes])
        assign_result.set_extra_property('target_boxes',
                                         gt_bboxes[pos_gt_index])
        return assign_result
