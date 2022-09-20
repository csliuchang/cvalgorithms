import torch

from ...builder import DETECTORS, build_backbone, build_head, build_neck
from .base import BaseDetector
from models.specific.bbox.coder.delta_xywh_bbox_coder import delta2bbox

__all__ = ["YOLOFeature"]


@DETECTORS.register_module()
class YOLOFeature(BaseDetector):
    """
    A YOLOF deepcv version implementment
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(YOLOFeature, self).__init__()
        self.device = None
        self.export = False
        self.backbone = build_backbone(backbone)
        self.input_stride = bbox_head.anchor_generator.strides

        self.fm_size = [backbone.input_size[0] / self.input_stride[i]
                        for i in range(len(self.input_stride))]

        if train_cfg is not None:
            bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        if neck is not None:
            self.neck = build_neck(neck)
        self.bbox_head = build_head(bbox_head)
        self.num_classes = self.bbox_head.num_classes
        # squeeze list for inference onnx
        self.fm_sizes = [[i, i] for i in self.fm_size]
        self.anchors_image = self.bbox_head.anchor_generator.grid_anchors(self.fm_sizes)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights(pretrained=pretrained)

    def extract_feat(self, inputs):
        """
        comm feature extract
        """
        features = self.backbone(inputs)
        if self.with_neck:
            features = self.neck(features)
        return features

    def init_weights(self, pretrained=None):
        """
        Pretrained backbone
        """
        super(YOLOFeature, self).init_weights(pretrained)
        if hasattr(self.backbone, 'init_weights'):
            self.backbone.init_weights(pretrained)
        self.bbox_head.init_weights()

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

    def forward_train(self, inputs, ground_truth, **kwargs):
        losses = dict()
        x = self.extract_feat(inputs)
        x = [x] if x is not list else x
        # first stage
        outs = self.bbox_head(x)
        input_base = self.concate_tuple_dict(outs, ground_truth, inputs)
        loss_base = self.bbox_head.loss(*input_base)
        for name, value in loss_base.items():
            losses['s0.{}'.format(name)] = value
        return losses

    def forward_infer(self, inputs, **kwargs):

        x = self.extract_feat(inputs)

        img_batch, image_shape = inputs.shape[0], inputs.shape[-2:]
        bbox_cls = []
        x = [x] if x is not list else x
        outs = self.bbox_head(x)
        if not self.export:
            bbox_inputs = outs + (inputs, self.test_cfg)
            bbox_cls = self.bbox_head.get_bboxes(*bbox_inputs)
            return bbox_cls

        cls_score_list, bbox_pred_list = outs
        anchors_image = [anchor_image.to(self.device) for anchor_image in self.anchors_image]

        num_levels = len(cls_score_list)
        for img_id in range(img_batch):
            cls_scores = [
                cls_score_list[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_preds = [
                bbox_pred_list[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_cls.append(self.inference(cls_scores, bbox_preds, anchors_image, image_shape))
        return torch.cat(bbox_cls, dim=0)

    def inference(self, cls_score_list, bbox_pred_list, anchors_image, image_shape):
        """
        Generate box params[x1,y1,x2,y2] for eval
        """
        num_levels = len(cls_score_list)
        mlvl_bboxes, mlvl_cls = [], []
        cls_score_list = [cls_score_list[i].detach() for i in range(num_levels)]
        for cls_score, bbox_pred, anchors in zip(cls_score_list, bbox_pred_list, anchors_image):
            cls_score = cls_score.permute(1, 2, 0).reshape(-1, self.num_classes).sigmoid()
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            bboxes = delta2bbox(anchors, bbox_pred, max_shape=image_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_cls.append(cls_score)
        mlvl_bboxes = torch.cat(mlvl_bboxes, dim=0)
        mlvl_cls = torch.cat(mlvl_cls, dim=0)
        mlvl_label = torch.zeros_like(mlvl_cls) + 1
        return torch.cat([mlvl_bboxes, mlvl_cls, mlvl_label], dim=1)

    def concate_tuple_dict(self, outs, ground_truth, inputs):
        """
        concate tuple and dict and output a tuple
        """
        gt_labels = [torch.as_tensor(gt_label, device=self.device) for gt_label in ground_truth['gt_labels']]
        gt_bboxes = [torch.as_tensor((gt_bbox), dtype=torch.float32, device=self.device) for gt_bbox in
                     ground_truth['gt_bboxes']]
        gt_masks = [torch.as_tensor(gt_mask, device=self.device) for gt_mask in ground_truth['gt_masks']]
        return outs + (gt_labels, gt_bboxes, gt_masks, inputs)
