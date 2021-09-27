import torch
import torch.nn as nn

from models.builder import HEADS, build_loss
from specific import build_anchor_generator
from .base_dense_head import BaseDenseHead
from models.utils import normal_init, multi_apply, images_to_levels
from specific.bbox import build_assigner, build_sampler, build_bbox_coder

@HEADS.register_module()
class AnchorHead(BaseDenseHead):
    """
    Anchor _base head (RetinaNet RRetinaNet YOLO F)
    """

    def __init__(self, num_classes,
                 in_channels,
                 anchor_generator,
                 bbox_coder,
                 loss_cls,
                 loss_bbox,
                 background_label=None,
                 reg_decoded_bbox=True,
                 feat_channels=256,
                 train_cfg=None,
                 test_cfg=None):
        super(AnchorHead, self).__init__()
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        if train_cfg.auto_anchors is True:
            assert len(anchor_generator.strides) > 0, 'len of features must > 0'
            per_anchor_nums = int(train_cfg.num_anchors/len(anchor_generator.strides))
            anchor_generator.base_sizes = []
            for i in range(0, train_cfg.num_anchors, per_anchor_nums):
                anchor_generator.base_sizes.append(train_cfg.anchors[i:per_anchor_nums+i:])
                anchor_generator.auto_anchors = train_cfg.auto_anchors
        self.anchor_generator = build_anchor_generator(anchor_generator)
        self.assigner = build_assigner(self.train_cfg.assigner)
        self.sampling = self.train_cfg.sampler_cfg.sampling
        self.sampler = build_sampler(self.train_cfg.sampler_cfg, context=self)
        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.cls_out_channels = num_classes
        self.num_anchors = self.anchor_generator.num_base_anchors[0]
        self.background_label = (
            num_classes if background_label is None else background_label)
        self.reg_decoded_bbox = reg_decoded_bbox
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self._init_layers()

    def _init_layers(self):
        self.conv_cls = nn.Conv2d(self.in_channels, self.num_anchors * self.cls_out_channels, 1)
        self.conv_reg = nn.Conv2d(self.in_channels, self.num_anchors * 4, 1)

    def init_weights(self):
        normal_init(self.conv_cls, std=0.01)
        normal_init(self.conv_reg, std=0.01)

    def forward_single(self, x):
        """
        rewrite this function for your own det
        """
        cls_score = self.conv_cls(x)
        bbox_pred = self.conv_reg(x)
        return cls_score, bbox_pred

    def forward(self, feats):
        return multi_apply(self.forward_single, feats)

    def get_anchors(self, featmap_sizes, img_batch, img_size, device='cuda'):
        """
        get anchors according to feature map size.
        Be attention this function will be rewrite for auto anchor
        """
        multi_level_anchors = self.anchor_generator.grid_anchors(
            featmap_sizes, device
        )
        valid_flag_list = []
        anchor_list = [multi_level_anchors for _ in range(img_batch)]
        for img_id in range(img_batch):
            multi_level_flags = self.anchor_generator.valid_flags(
                featmap_sizes, img_size, device)
            valid_flag_list.append(multi_level_flags)
        return anchor_list, valid_flag_list

    def get_targets(self,
                    anchor_list,
                    valid_flag_list,
                    inputs,
                    gt_bboxes_list,
                    gt_bboxes_ignore_list=None,
                    gt_labels_list=None,
                    label_channels=1,
                    unmap_outputs=True,
                    return_sampling_results=False):
        """Compute regression and classification targets for anchors in
            multiple images.

        Args:
            anchor_list (list[list[Tensor]]): Multi level anchors of each
                image. The outer list indicates images, and the inner list
                corresponds to feature levels of the image. Each element of
                the inner list is a tensor of shape (num_anchors, 4).
            valid_flag_list (list[list[Tensor]]): Multi level valid flags of
                each image. The outer list indicates images, and the inner list
                corresponds to feature levels of the image. Each element of
                the inner list is a tensor of shape (num_anchors, )
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image.
            img_metas (list[dict]): Meta info of each image.
            gt_bboxes_ignore_list (list[Tensor]): Ground truth bboxes to be
                ignored.
            gt_labels_list (list[Tensor]): Ground truth labels of each box.
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
            additional_returns: This function enables user-defined returns from
                `self._get_targets_single`. These returns are currently refined
                to properties at each feature map (i.e. having HxW dimension).
                The results will be concatenated after the end

        """
        num_imgs = inputs.shape[0]
        assert len(anchor_list) == len(valid_flag_list) == num_imgs

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        # concat all level anchors to a single tensor
        concat_anchor_list = []
        concat_valid_flag_list = []
        for i in range(num_imgs):
            assert len(anchor_list[i]) == len(valid_flag_list[i])
            concat_anchor_list.append(torch.cat(anchor_list[i]))
            concat_valid_flag_list.append(torch.cat(valid_flag_list[i]))

        # compute targets for each image
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]
        results = multi_apply(
            self._get_targets_single,
            concat_anchor_list,
            concat_valid_flag_list,
            gt_bboxes_list,
            gt_bboxes_ignore_list,
            gt_labels_list,
            inputs,
            label_channels=label_channels,
            unmap_outputs=unmap_outputs)
        (all_labels, all_label_weights, all_bbox_targets, all_bbox_weights,
         pos_inds_list, neg_inds_list, sampling_results_list) = results[:7]
        rest_results = list(results[7:])  # user-added return values
        # no valid anchors
        if any([labels is None for labels in all_labels]):
            return None
        # sampled anchors of all images
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
        # split targets to a list w.r.t. multiple levels
        labels_list = images_to_levels(all_labels, num_level_anchors)
        label_weights_list = images_to_levels(all_label_weights,
                                              num_level_anchors)
        bbox_targets_list = images_to_levels(all_bbox_targets,
                                             num_level_anchors)
        bbox_weights_list = images_to_levels(all_bbox_weights,
                                             num_level_anchors)
        res = (labels_list, label_weights_list, bbox_targets_list,
               bbox_weights_list, num_total_pos, num_total_neg)
        if return_sampling_results:
            res = res + (sampling_results_list, )
        for i, r in enumerate(rest_results):  # user-added return values
            rest_results[i] = images_to_levels(r, num_level_anchors)

        return res + tuple(rest_results)


    def _get_targets_single(self,
                            flat_anchors,
                            valid_flags,
                            gt_bboxes,
                            gt_bboxes_ignore,
                            gt_labels,
                            inputs,
                            label_channels=1,
                            unmap_outputs=True):
        raise NotImplementedError

    def get_bboxes(self, cls_scores, bbox_preds, img_metas, cfg=None, rois=None, **kwargs):
        pass

    def _get_bboxes_single(self,
                           cls_score_list,
                           bbox_pred_list,
                           anchors,
                           img_shape,
                           cfg,):
        """
        Transform outputs for a single batch item into labeled boxes
        """
        pass

    def loss_single(self, cls_score, bbox_pred, anchors, labels, label_weights,
                    bbox_targets, bbox_weights, num_total_samples):
        raise NotImplementedError

    def loss(self, cls_scores, pred_bbox, gt_labels, gt_bboxes, gt_masks, inputs):
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.anchor_generator.num_levels
        device = cls_scores[0].device
        img_batch, img_size = inputs.shape[0], inputs.shape[2:]
        anchor_list, valid_flag_list = self.get_anchors(featmap_sizes, img_batch, img_size, device=device)
        label_channels = self.cls_out_channels

        # assign anchor
        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            inputs,
            gt_bboxes_list=gt_bboxes,
            gt_labels_list=gt_labels,
            label_channels=label_channels
        )
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        num_total_samples = (
            num_total_pos + num_total_neg if self.sampling else num_total_pos)

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        concat_anchor_list = []
        for i in range(len(anchor_list)):
            concat_anchor_list.append(torch.cat(anchor_list[i]))
        all_anchor_list = images_to_levels(concat_anchor_list,
                                           num_level_anchors)

        losses_cls, losses_bbox = multi_apply(
            self.loss_single,
            cls_scores,
            pred_bbox,
            all_anchor_list,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_samples=num_total_samples)
        return dict(loss_cls=losses_cls, loss_bbox=losses_bbox)
