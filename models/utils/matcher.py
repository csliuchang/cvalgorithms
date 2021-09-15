import math
from typing import Tuple

import torch
import torch.nn as nn
from .comm import box_xyxy_to_cxcywh
import numpy as np
from specific.anchor.anchor_generator import DefaultAnchorGenerator
from ..utils import nonzero_tuple, batched_nms




def permute_to_N_HWA_K(tensor, K: int):
    """
    Transpose/reshape a tensor from (N, (Ai x K), H, W) to (N, (HxWxAi), K)
    """
    assert tensor.dim() == 4, tensor.shape
    N, _, H, W = tensor.shape
    tensor = tensor.view(N, -1, K, H, W).permute(0, 3, 4, 1, 2)
    tensor = tensor.reshape(N, -1, K)
    return tensor


class UniformMatcher(nn.Module):
    """
    Uniform Matching between the anchors and gt boxes, which can achieve
    balance in positive anchors.

    Args:
        match_times(int): Number of positive anchors for each gt box.
    """

    def __init__(self, match_times: int = 4):
        super(UniformMatcher, self).__init__()
        self.match_times = match_times

    @torch.no_grad()
    def forward(self, pred_boxes, anchors, all_targets, sizes):
        bs, num_queries = pred_boxes.shape[:2]

        # We flatten to compute the cost matrices in a batch
        # [batch_size * num_anchors, 4]
        out_bbox = pred_boxes.flatten(0, 1)
        anchors = anchors.flatten(0, 1)

        # Also concat the target boxes
        cost_bbox = torch.cdist(
            box_xyxy_to_cxcywh(out_bbox), box_xyxy_to_cxcywh(all_targets), p=1)
        cost_bbox_anchors = torch.cdist(
            box_xyxy_to_cxcywh(anchors), box_xyxy_to_cxcywh(all_targets), p=1)

        # Final cost matrix
        C = cost_bbox.view(bs, num_queries, -1).cpu()
        C1 = cost_bbox_anchors.view(bs, num_queries, -1).cpu()

        all_indices_list = [[] for _ in range(bs)]

        # positive indices when matching predict boxes and gt boxes
        indices = [tuple(torch.topk(c[i], k=self.match_times, dim=0, largest=False)[1].numpy().tolist())
                   for i, c in enumerate(C.split(sizes, -1))]

        # positive indices when matching anchor boxes and gt boxes
        indices1 = [tuple(torch.topk(c[i], k=self.match_times, dim=0, largest=False)[1].numpy().tolist())
                    for i, c in enumerate(C1.split(sizes, -1))]

        # concat the indices according to image ids
        for img_id, (idx, idx1) in enumerate(zip(indices, indices1)):
            img_id_i = [np.array(idx_ + idx1_) for (idx_, idx1_) in zip(idx, idx1)]
            img_id_j = [np.array(list(range(len(idx_))) + list(range(len(idx1_))))
                        for (idx_, idx1_) in zip(idx, idx1)]
            all_indices_list[img_id] = [*zip(img_id_i, img_id_j)]

        # re-organize the positive indices
        all_indices = []
        for img_id in range(bs):
            all_idx_i = []
            all_idx_j = []
            for idx_list in all_indices_list[img_id]:
                idx_i, idx_j = idx_list
                all_idx_i.append(idx_i)
                all_idx_j.append(idx_j)
            all_idx_i = np.hstack(all_idx_i)
            all_idx_j = np.hstack(all_idx_j)
            all_indices.append((all_idx_i, all_idx_j))
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in all_indices]