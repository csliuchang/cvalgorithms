import torch
import torch.nn as nn
import math
from models.builder import HEADS, build_loss
from specific import build_anchor_generator
from .base_dense_head import BaseDenseHead
from models.utils import normal_init
from ..dense_heads import AnchorHead
from models.utils import get_norm, rbbox2circumhbbox, unmap, ranchor_inside_flags, \
    padding_results, rdets2points, rdets2points_tensor, get_activation, c2_xavier_fill
import numpy as np
from models.utils import multiclass_rnms


@HEADS.register_module()
class RRetinaHead(AnchorHead):
    def __init__(self,
                 num_classes,
                 in_channels,
                 stacked_convs=4,
                 norm_cfg="BN",
                 act_cfg="ReLU",
                 use_h_gt=True,
                 anchor_generator=None,
                 bbox_coder=None,
                 **kwargs
                 ):

        self.stacked_convs = stacked_convs
        self.norm = norm_cfg
        self.act = act_cfg
        self.use_h_gt = use_h_gt
        self.prior_prob = 0.01
        super(RRetinaHead, self).__init__(num_classes, in_channels, anchor_generator, bbox_coder, **kwargs)

    def _init_layers(self):
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(nn.Sequential(
                nn.Conv2d(chn,
                          self.feat_channels,
                          3,
                          stride=1,
                          padding=1),
                get_norm(self.norm, self.feat_channels),
                get_activation(self.act)

            )),
            self.reg_convs.append(nn.Sequential(
                nn.Conv2d(chn,
                          self.feat_channels,
                          3,
                          stride=1,
                          padding=1),
                get_norm(self.norm, self.feat_channels),
                get_activation(self.act)
            ))
        self.retina_cls = nn.Conv2d(
            self.feat_channels,
            self.num_anchors * self.cls_out_channels,
            3,
            padding=1)
        self.retina_reg = nn.Conv2d(
            self.feat_channels, self.num_anchors * 5, 3, padding=1)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, (nn.GroupNorm, nn.BatchNorm2d, nn.SyncBatchNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        # Use prior in model initialization to improve stability
        bias_value = -math.log((1 - self.prior_prob) / self.prior_prob)
        normal_init(self.retina_cls, std=0.01, bias=bias_value)
        normal_init(self.retina_reg, std=0.01)

    def forward_single(self, x):
        cls_feat = x
        reg_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        cls_score = self.retina_cls(cls_feat)
        bbox_pred = self.retina_reg(reg_feat)
        return cls_score, bbox_pred

    def _get_bboxes_single(self,
                           cls_score_list,
                           bbox_pred_list,
                           anchors,
                           img_shape,
                           cfg,):
        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_score_list) == len(bbox_pred_list) == len(anchors)
        # pre nms, to do reduce
        mlvl_bboxes = []
        mlvl_scores = []
        for cls_score, bbox_pred, anchors in zip(cls_score_list,
                                                 bbox_pred_list, anchors):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            cls_score = cls_score.permute(1, 2, 0).reshape(-1, self.cls_out_channels)
            scores = cls_score.sigmoid()
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 5)
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                max_scores, _ = scores.max(dim=1)

                _, topk_inds = max_scores.topk(nms_pre)
                anchors = anchors[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
            bboxes = self.bbox_coder.decode(
                anchors, bbox_pred, max_shape=img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)

        mlvl_bboxes = torch.cat(mlvl_bboxes)
        mlvl_scores = torch.cat(mlvl_scores)

        # add nms
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)

        det_bboxes, det_labels = multiclass_rnms(mlvl_bboxes, mlvl_scores,
                                                 cfg.score_thr, cfg.nms,
                                                 cfg.max_per_img)

        return torch.cat([det_bboxes[None, :, :], det_labels[None, :, None]], dim=2)

    def get_bboxes(self, cls_scores, bbox_preds, img_metas, cfg=None, **kwargs):
        num_levels = len(cls_scores)
        assert len(cls_scores) == len(bbox_preds)

        device = cls_scores[0].device

        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_anchors = self.anchor_generator.grid_anchors(
            featmap_sizes, device=device)

        result_list = []

        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_preds_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            img_shape = img_metas[img_id].shape[-2:]
            proposals = self._get_bboxes_single(cls_score_list, bbox_preds_list,
                                                mlvl_anchors, img_shape,
                                                cfg)
            result_list.append(proposals)
        nms_pre = cfg.get('nms_pre', -1) * num_levels
        final_results = padding_results(result_list, nms_pre, nums_tensor=7)
        final_results = torch.stack([rdets2points_tensor(final_result) for final_result in final_results], dim=0)
        return final_results

    def loss_single(self, cls_score, bbox_pred, anchors, labels, label_weights,
                    bbox_targets, bbox_weights, num_total_samples):
        # labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        # cls loss
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
        labels = labels.reshape(-1)
        # print(f'------num total samples is ---------------{num_total_samples}----------------------')
        loss_cls = self.loss_cls(cls_score, labels.float(), label_weights, avg_factor=num_total_samples)

        # reg loss
        bbox_targets = bbox_targets.reshape(-1, 5)
        bbox_weights = bbox_weights.reshape(-1, 5)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 5)
        if self.reg_decoded_bbox:
            anchors = anchors.reshape(-1, 5)
            bbox_pred = self.bbox_coder.decode(anchors, bbox_pred)
        loss_bbox = self.loss_bbox(
            bbox_pred,
            bbox_targets,
            bbox_weights,
            avg_factor=num_total_samples)
        # print(f'the cls loss is {loss_cls}, and the bbox loss is {loss_bbox}')
        return loss_cls, loss_bbox

    def _get_targets_single(self,
                            flat_anchors,
                            valid_flags,
                            gt_bboxes,
                            gt_bboxes_ignore,
                            gt_labels,
                            inputs,
                            label_channels=1,
                            unmap_outputs=True):
        """Compute regression and classification targets for anchors in
            a single image.

        Args:
            flat_anchors (Tensor): Multi-level anchors of the image, which are
                concatenated into a single tensor of shape (num_anchors, 5)
            valid_flags (Tensor): Multi level valid flags of the image,
                which are concatenated into a single tensor of
                    shape (num_anchors,).
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 5).
            img_meta (dict): Meta info of the image.
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 5).
            img_meta (dict): Meta info of the image.
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            label_channels (int): Channel of label.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.

        Returns:
            tuple:
                labels_list (list[Tensor]): Labels of each level
                label_weights_list (list[Tensor]): Label weights of each level
                bbox_targets_list (list[Tensor]): BBox targets of each level
                bbox_weights_list (list[Tensor]): BBox weights of each level
                num_total_pos (int): Number of positive samples in all images
                num_total_neg (int): Number of negative samples in all images
        """
        inside_flags = ranchor_inside_flags(flat_anchors, valid_flags,
                                            inputs.shape[1:],
                                            self.train_cfg.allowed_border)
        if not inside_flags.any():
            return (None,) * 6
        # assign gt and sample anchors
        anchors = flat_anchors[inside_flags, :]

        if self.use_h_gt:
            anchors_assign = rbbox2circumhbbox(anchors)
            gt_bboxes_assign = rbbox2circumhbbox(gt_bboxes)
            if gt_bboxes_ignore is not None and gt_bboxes_ignore.numel() > 0:
                gt_bboxes_ignore_assign = rbbox2circumhbbox(gt_bboxes_ignore)
            else:
                gt_bboxes_ignore_assign = None
        else:
            anchors_assign = anchors
            gt_bboxes_assign = gt_bboxes
            gt_bboxes_ignore_assign = gt_bboxes_ignore

        assign_result = self.assigner.assign(
            anchors_assign, gt_bboxes_assign, gt_bboxes_ignore_assign,
            None if self.sampling else gt_labels)
        sampling_result = self.sampler.sample(assign_result, anchors,
                                              gt_bboxes)

        num_valid_anchors = anchors.shape[0]
        bbox_targets = torch.zeros_like(anchors)
        bbox_weights = torch.zeros_like(anchors)
        labels = anchors.new_full((num_valid_anchors,),
                                  self.background_label,
                                  dtype=torch.long)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            if not self.reg_decoded_bbox:
                pos_bbox_targets = self.bbox_coder.encode(
                    sampling_result.pos_bboxes, sampling_result.pos_gt_bboxes)
            else:
                pos_bbox_targets = sampling_result.pos_gt_bboxes
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0
            if gt_labels is None:
                # only rpn gives gt_labels as None, this time FG is 1
                labels[pos_inds] = 1
            else:
                # gt labels begin 1
                labels[pos_inds] = gt_labels[
                    sampling_result.pos_assigned_gt_inds]
            if self.train_cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg.pos_weight
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # map up to original set of anchors
        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            labels = unmap(
                labels, num_total_anchors, inside_flags,
                fill=self.num_classes)  # fill bg label
            label_weights = unmap(label_weights, num_total_anchors,
                                  inside_flags)
            bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
            bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)

        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
                neg_inds, sampling_result)
