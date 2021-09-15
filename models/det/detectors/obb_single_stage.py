import torch
import numpy as np
from models.utils import points2rdets, rdets2points_tensor
from models.builder import DETECTORS, build_backbone, build_head, build_neck
from .base import BaseDetector


@DETECTORS.register_module()
class OBBSingleStageDetector(BaseDetector):
    """Base class for single-stage detectors.

    Single-stage detectors directly and densely predict bounding boxes on the
    output features of the backbone+neck.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(OBBSingleStageDetector, self).__init__()
        self.device = None
        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = build_head(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        """Initialize the weights in detector

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        super(OBBSingleStageDetector, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        self.bbox_head.init_weights()

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck
        """
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward(self, inputs, return_metrics=False, **kwargs):
        """Calls either :func:`forward_train` or :func:`forward_infer` depending
        on whether ``return_metrics`` is ``True``.
        """
        if self.device != inputs.device:
            self.device = inputs.device
        if return_metrics:
            metrics = self.forward_train(inputs, **kwargs)
            return self._parse_metrics(metrics)
        else:
            return self.forward_infer(inputs, **kwargs)

    def forward_train(self, img, ground_truth, **kwargs):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
            :param img:
            :param ground_truth:
        """
        losses = dict()
        x = self.extract_feat(img)

        outs = self.bbox_head(x)
        input_base = self.concate_tuple_dict(outs, ground_truth, img)
        loss_base = self.bbox_head.loss(*input_base)
        for name, value in loss_base.items():
            losses['s0.{}'.format(name)] = value

        return losses

    def forward_infer(self, img, rescale=False, **kwargs):
        """Test function without test time augmentation

        Args:
            imgs (list[torch.Tensor]): List of multiple images
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            np.ndarray: proposals
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        outs += img, self.test_cfg
        bbox_list = self.bbox_head.get_bboxes(
            *outs, rescale=rescale)
        return bbox_list

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test function with test time augmentation"""
        raise NotImplementedError

    def concate_tuple_dict(self, outs, ground_truth, inputs):
        """
        concate tuple and dict and output a tuple
        """
        gt_labels = [torch.as_tensor(gt_label, device=self.device) for gt_label in ground_truth['gt_labels']]
        gt_bboxes = [torch.as_tensor(points2rdets(gt_bbox), dtype=torch.float32, device=self.device) for gt_bbox in ground_truth['gt_bboxes']]
        gt_masks = [torch.as_tensor(gt_mask, device=self.device) for gt_mask in ground_truth['gt_masks']]
        return outs + (gt_labels, gt_bboxes, gt_masks, inputs)
