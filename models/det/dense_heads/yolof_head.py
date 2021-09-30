import torch
import torch.nn as nn
import math
from ...builder import HEADS, build_loss
from .anchor_head import AnchorHead
from ...utils import get_norm, rbbox2circumhbbox, unmap, ranchor_inside_flags, \
    padding_results, rdets2points_tensor, get_activation, c2_xavier_fill, normal_init
from ...utils import multiclass_nms, multi_apply
from models.base.plugins.gcnet import ContextBlock

INF = 1e-8


@HEADS.register_module()
class YOLOFeatureHead(AnchorHead):
    def __init__(self,
                 num_classes,
                 in_channels,
                 block_dilations,
                 stacked_residual_blocks,
                 reg_num_convs,
                 cls_num_convs,
                 norm_cfg="BN",
                 act_cfg="ReLU",
                 use_h_gt=True,
                 anchor_generator=None,
                 bbox_coder=None,
                 **kwargs
                 ):
        self.in_channels, self.out_channels, self.mid_channels = in_channels, in_channels // 4, in_channels // 8
        self.decoder_in_channels = in_channels // 4
        self.stacked_residual_blocks = stacked_residual_blocks
        self.block_dilations = block_dilations
        self.reg_num_convs = reg_num_convs
        self.cls_num_convs = cls_num_convs
        self.norm = norm_cfg
        self.act = act_cfg
        self.use_h_gt = use_h_gt

        self.prior_prob = 0.01
        super(YOLOFeatureHead, self).__init__(num_classes, in_channels, anchor_generator, bbox_coder, **kwargs)

    def _init_layers(self):
        # build encoder
        self.lateral_conv = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            kernel_size=1
        )
        self.lateral_norm = get_norm(self.norm, self.out_channels)

        self.fpn_conv = nn.Conv2d(self.out_channels,
                                  self.out_channels,
                                  kernel_size=3,
                                  padding=1
                                  )
        self.fpn_norm = get_norm(self.norm, self.out_channels)

        encoder_blocks = []
        for i in range(self.stacked_residual_blocks):
            dilation = self.block_dilations[i]
            encoder_blocks.append(
                Bottleneck(self.out_channels,
                           self.mid_channels,
                           dilation,
                           self.norm,
                           self.act
                           )
            )
        self.dilated_encoder_blocks = nn.Sequential(*encoder_blocks)

        # build dense gcnet blocks
        self.gc1 = ContextBlock(self.out_channels, ratio=8)
        self.gc2 = ContextBlock(self.out_channels, ratio=8)
        self.gc3 = ContextBlock(self.out_channels, ratio=8)

        # build decoder
        cls_subnet = []
        bbox_subnet = []
        for i in range(self.cls_num_convs):
            cls_subnet.append(
                nn.Conv2d(self.decoder_in_channels,
                          self.decoder_in_channels,
                          kernel_size=3,
                          stride=1,
                          padding=1)
            )
            cls_subnet.append(get_norm(self.norm, self.decoder_in_channels))
            cls_subnet.append(get_activation(self.act))
        for i in range(self.reg_num_convs):
            bbox_subnet.append(
                nn.Conv2d(self.decoder_in_channels,
                          self.decoder_in_channels,
                          kernel_size=3,
                          stride=1,
                          padding=1))
            bbox_subnet.append(get_norm(self.norm, self.decoder_in_channels))
            bbox_subnet.append(get_activation(self.act))
            self.cls_subnet = nn.Sequential(*cls_subnet)
            self.bbox_subnet = nn.Sequential(*bbox_subnet)
            self.cls_score = nn.Conv2d(self.decoder_in_channels,
                                       self.num_anchors * self.num_classes,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)
            self.bbox_pred = nn.Conv2d(self.decoder_in_channels,
                                       self.num_anchors * 4,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)
            self.object_pred = nn.Conv2d(self.decoder_in_channels,
                                         self.num_anchors,
                                         kernel_size=3,
                                         stride=1,
                                         padding=1)

    def init_weights(self):
        c2_xavier_fill(self.lateral_conv)
        c2_xavier_fill(self.fpn_conv)
        for m in [self.lateral_norm, self.fpn_norm]:
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
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
        torch.nn.init.constant_(self.cls_score.bias, bias_value)

    def loss(self, cls_scores, bbox_preds, gt_labels, gt_bboxes, gt_masks, inputs):
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (batch, num_anchors * num_classes, h, w)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (batch, num_anchors * 4, h, w)
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss. Default: None

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert len(cls_scores) == 1
        assert self.anchor_generator.num_levels == 1
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        device = cls_scores[0].device
        img_batch, img_size = inputs.shape[0], inputs.shape[2:]
        anchor_list, valid_flag_list = self.get_anchors(featmap_sizes, img_batch, img_size, device=device)

        anchor_list = [anchors[0] for anchors in anchor_list]
        valid_flag_list = [valid_flags[0] for valid_flags in valid_flag_list]

        cls_scores_list = levels_to_images(cls_scores)
        bbox_preds_list = levels_to_images(bbox_preds)

        label_channels = self.cls_out_channels

        cls_reg_targets = self.get_targets(
            cls_scores_list,
            bbox_preds_list,
            anchor_list,
            valid_flag_list,
            inputs,
            gt_bboxes_list=gt_bboxes,
            gt_labels_list=gt_labels,
            label_channels=label_channels)
        if cls_reg_targets is None:
            return None
        (batch_labels, batch_label_weights, num_total_pos, num_total_neg,
         batch_bbox_weights, batch_pos_predicted_boxes,
         batch_target_boxes) = cls_reg_targets

        flatten_labels = batch_labels.reshape(-1)
        batch_label_weights = batch_label_weights.reshape(-1)
        cls_score = cls_scores[0].permute(0, 2, 3,
                                          1).reshape(-1, self.cls_out_channels)

        num_total_samples = (num_total_pos +
                             num_total_neg) if self.sampling else num_total_pos

        # classification loss
        loss_cls = self.loss_cls(
            cls_score,
            flatten_labels.float(),
            batch_label_weights,
            avg_factor=num_total_samples)

        # regression loss
        if batch_pos_predicted_boxes.shape[0] == 0:
            # no pos sample
            loss_bbox = batch_pos_predicted_boxes.sum() * 0
        else:
            loss_bbox = self.loss_bbox(
                batch_pos_predicted_boxes,
                batch_target_boxes,
                batch_bbox_weights.float(),
                avg_factor=num_total_samples)

        return dict(loss_cls=loss_cls, loss_bbox=loss_bbox)

    def get_targets(self, cls_scores_list, bbox_preds_list, anchor_list, valid_flag_list, inputs, gt_bboxes_list,
                    gt_bboxes_ignore_list=None, gt_labels_list=None, label_channels=None):
        num_imgs = inputs.shape[0]
        # compute targets for each image
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]
        results = multi_apply(
            self._get_targets_single,
            bbox_preds_list,
            anchor_list,
            valid_flag_list,
            gt_bboxes_list,
            gt_bboxes_ignore_list,
            gt_labels_list,
            inputs,
            label_channels=label_channels,

        )
        (all_labels, all_label_weights, pos_inds_list, neg_inds_list,
         sampling_results_list) = results[:5]
        rest_results = list(results[5:])  # user-added return values
        # no valid anchors
        if any([labels is None for labels in all_labels]):
            return None
        # sampled anchors of all images
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])

        batch_labels = torch.stack(all_labels, 0)
        batch_label_weights = torch.stack(all_label_weights, 0)

        res = (batch_labels, batch_label_weights, num_total_pos, num_total_neg)
        for i, rests in enumerate(rest_results):  # user-added return values
            rest_results[i] = torch.cat(rests, 0)

        return res + tuple(rest_results)

    def _get_targets_single(self,
                            bbox_preds,
                            flat_anchors,
                            valid_flags,
                            gt_bboxes,
                            gt_bboxes_ignore,
                            gt_labels,
                            inputs,
                            label_channels=1,
                            unmap_outputs=True):
        inside_flags = ranchor_inside_flags(flat_anchors, valid_flags,
                                            inputs.shape[-2:],
                                            self.train_cfg.allowed_border)
        if not inside_flags.any():
            return (None,) * 8
        # assign gt and sample anchors
        anchors = flat_anchors[inside_flags, :]
        bbox_preds = bbox_preds.reshape(-1, 4)
        bbox_preds = bbox_preds[inside_flags, :]

        # decoder bbox
        decoder_bbox_preds = self.bbox_coder.decode(anchors, bbox_preds)
        assign_result = self.assigner.assign(decoder_bbox_preds, anchors, gt_bboxes, gt_bboxes_ignore,
                                             None if self.sampling else gt_labels)
        pos_bbox_weights = assign_result.get_extra_property('pos_idx')
        pos_predicted_boxes = assign_result.get_extra_property(
            'pos_predicted_boxes')
        pos_target_boxes = assign_result.get_extra_property('target_boxes')

        sampling_result = self.sampler.sample(assign_result, anchors,
                                              gt_bboxes)
        num_valid_anchors = anchors.shape[0]
        labels = anchors.new_full((num_valid_anchors,),
                                  self.num_classes,
                                  dtype=torch.long)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            if gt_labels is None:
                # Only rpn gives gt_labels as None
                # Foreground is the first class since v2.5.0
                labels[pos_inds] = 0
            else:
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

        return (labels, label_weights, pos_inds, neg_inds, sampling_result,
                pos_bbox_weights, pos_predicted_boxes, pos_target_boxes)

    def forward_single(self, feature):
        # encoder
        out = self.lateral_norm(self.lateral_conv(feature))
        out = self.fpn_norm(self.fpn_conv(out))

        out1 = self.dilated_encoder_blocks[0](out)
        out2 = self.dilated_encoder_blocks[1](out1) + self.gc1(out1)
        out3 = self.dilated_encoder_blocks[2](out2) + self.gc2(out2 + out1)
        out = self.dilated_encoder_blocks[3](out3) + self.gc3(out3 + out2 + out1)

        cls_score = self.cls_score(self.cls_subnet(out))
        N, _, H, W = cls_score.shape
        cls_score = cls_score.view(N, -1, self.num_classes, H, W)

        reg_feat = self.bbox_subnet(out)
        bbox_reg = self.bbox_pred(reg_feat)
        objectness = self.object_pred(reg_feat)

        # implicit objectness
        objectness = objectness.view(N, -1, 1, H, W)
        normalized_cls_score = cls_score + objectness - torch.log(
            1. + torch.clamp(cls_score.exp(), max=INF) + torch.clamp(
                objectness.exp(), max=INF))
        normalized_cls_score = normalized_cls_score.view(N, -1, H, W)
        return normalized_cls_score, bbox_reg

    def refine_bboxes(self):
        pass

    def filter_bboxes(self, cls_scores, bbox_preds):
        num_levels = len(cls_scores)
        assert num_levels == len(bbox_preds)

        num_imgs = cls_scores[0].size(0)

        for i in range(num_levels):
            assert num_imgs == cls_scores[i].size(0) == bbox_preds[i].size(0)

        device = cls_scores[0].device
        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_anchors = self.anchor_generator.grid_anchors(
            featmap_sizes, device=device)

        bboxes_list = [[] for _ in range(num_imgs)]

        for lvl in range(num_levels):
            cls_score = cls_scores[lvl]
            bbox_pred = bbox_preds[lvl]

            anchors = mlvl_anchors[lvl]

            cls_score = cls_score.permute(0, 2, 3, 1)  # (N, H, W, A*C)
            cls_score = cls_score.reshape(num_imgs, -1, self.num_anchors, self.cls_out_channels)  # (N, H*W, A, C)

            cls_score, _ = cls_score.max(dim=-1, keepdim=True)  # (N, H*W, A, 1)
            best_ind = cls_score.argmax(dim=-2, keepdim=True)  # (N, H*W, 1, 1)
            best_ind = best_ind.expand(-1, -1, -1, 5)  # (N, H*W, 1, 5)

            bbox_pred = bbox_pred.permute(0, 2, 3, 1)  # (N, H, W, A*5)
            bbox_pred = bbox_pred.reshape(num_imgs, -1, self.num_anchors, 5)  # (N, H*W, A, 5)

            best_pred = bbox_pred.gather(dim=-2, index=best_ind).squeeze(dim=-2)  # (N, H*W, 5)

            # anchors shape (H*W*A, 5)
            anchors = anchors.reshape(-1, self.num_anchors, 5)  # (H*W, A, 5)

            for img_id in range(num_imgs):
                best_ind_i = best_ind[img_id]  # (H*W, 1, 5)
                best_pred_i = best_pred[img_id]  # (H*W, 5)
                best_anchor_i = anchors.gather(dim=-2, index=best_ind_i).squeeze(dim=-2)  # (H*W, 5)
                best_bbox_i = self.bbox_coder.decode(best_anchor_i, best_pred_i)
                bboxes_list[img_id].append(best_bbox_i.detach())

        return bboxes_list

    def _get_bboxes_single(self,
                           cls_score_list,
                           bbox_pred_list,
                           anchors,
                           img_shape,
                           cfg, ):
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
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
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

        det_bboxes, det_labels = multiclass_nms(mlvl_bboxes, mlvl_scores,
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
        final_results = padding_results(result_list, nms_pre, 6)
        final_results = torch.stack([final_result for final_result in final_results], dim=0)

        return final_results

    def loss_refine(self, ):
        pass

    def loss_single(self, cls_score, bbox_pred, anchors, labels, label_weights,
                    bbox_targets, bbox_weights, num_total_samples):
        # labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        # cls loss
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
        labels = labels.reshape(-1, self.num_classes)
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


class Bottleneck(nn.Module):

    def __init__(self,
                 in_channels: int = 512,
                 mid_channels: int = 128,
                 dilation: int = 1,
                 norm_type: str = 'BN',
                 act_type: str = 'ReLU'):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, padding=0),
            get_norm(norm_type, mid_channels),
            get_activation(act_type)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=dilation, dilation=dilation),
            get_norm(norm_type, mid_channels),
            get_activation(act_type)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(mid_channels, in_channels, kernel_size=1, padding=0),
            get_norm(norm_type, in_channels),
            get_activation(act_type)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = out + identity
        return out


def levels_to_images(mlvl_tensor):
    """Concat multi-level feature maps by image.

    [feature_level0, feature_level1...] -> [feature_image0, feature_image1...]
    Convert the shape of each element in mlvl_tensor from (N, C, H, W) to
    (N, H*W , C), then split the element to N elements with shape (H*W, C), and
    concat elements in same image of all level along first dimension.

    Args:
        mlvl_tensor (list[torch.Tensor]): list of Tensor which collect from
            corresponding level. Each element is of shape (N, C, H, W)

    Returns:
        list[torch.Tensor]: A list that contains N tensors and each tensor is
            of shape (num_elements, C)
    """
    batch_size = mlvl_tensor[0].size(0)
    batch_list = [[] for _ in range(batch_size)]
    channels = mlvl_tensor[0].size(1)
    for t in mlvl_tensor:
        t = t.permute(0, 2, 3, 1)
        t = t.view(batch_size, -1, channels).contiguous()
        for img in range(batch_size):
            batch_list[img].append(t[img])
    return [torch.cat(item, 0) for item in batch_list]
