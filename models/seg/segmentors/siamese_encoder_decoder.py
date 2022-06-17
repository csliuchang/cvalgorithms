from torch import nn
import torch
from ...utils import add_prefix, resize
from ... import builder
from ...builder import SEGMENTORS
from specific.siamese.builder import build_siamese_layer
from .encoder_decoder import EncoderDecoder
from ..losses import BatchContrastiveLoss
import torch.nn.functional as F
import numpy as np
import random


@SEGMENTORS.register_module()
class SiameseEncoderDecoder(EncoderDecoder):
    def __init__(self,
                 backbone,
                 decode_head,
                 neck=None,
                 fusion_mode='mid',
                 auxiliary_head=None,
                 siamese_layer=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 export_feature=None,
                 with_constant_head=True,
                 **kwargs):
        super(SiameseEncoderDecoder, self).__init__(
            backbone=backbone,
            decode_head=decode_head,
            neck=neck,
            auxiliary_head=auxiliary_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained
        )
        self.fusion_mode = fusion_mode
        self.export_feature = export_feature
        self.with_constant_head = with_constant_head
        self.constant_loss = BatchContrastiveLoss()
        self.sig = nn.Sigmoid()
        self.sft = nn.Softmax(dim=1)
        if self.fusion_mode == "mid":
            siamese_blocks = []
            for chn in siamese_layer.fea_list:
                siamese_layer.in_c, siamese_layer.ou_c = chn, chn
                siamese_blocks.append(build_siamese_layer(siamese_layer))
            self.siamese_layer = nn.ModuleList(siamese_blocks)
        else:
            self.siamese_layer = build_siamese_layer(siamese_layer)

    def extract_feat(self, inputs):
        """Use Siamese Network Extract Features"""
        inputs_n, inputs_g = torch.chunk(inputs, 2, dim=1)
        inputs_switch = self._switch_channel(inputs_n)
        features_n, features_g = self.backbone(inputs_switch), self.backbone(inputs_g)
        # mid fusion could reduce siamese head
        return [features_n, features_g]

    def _init_decode_head(self, decode_head):
        """Initialize ``decode_head``"""
        self.decode_head = builder.build_head(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone and heads.

        Parameters
        ----------
        pretrained : str, optional
            Path to pre-trained weights.
            Defaults to None.
        """

        super(EncoderDecoder, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        self.decode_head.init_weights()
        if self.with_auxiliary_head:
            if isinstance(self.auxiliary_head, nn.ModuleList):
                for aux_head in self.auxiliary_head:
                    aux_head.init_weights()
            else:
                self.auxiliary_head.init_weights()

    def encode_decode(self, inputs):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        x = self.extract_feat(inputs)
        out = self._decode_head_forward_infer(x)
        out = resize(
            input=out,
            size=inputs.shape[2:],

            mode='bilinear',
            align_corners=self.align_corners)
        return out

    def _decode_head_forward_train(self, x, ground_truth):
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        if self.fusion_mode == "mid":
            x = [self.siamese_layer[idx](x[0][idx], x[1][idx]) for idx in range(len(x[0]))]
            loss_decode = self.decode_head.forward_train(x, ground_truth)
        elif self.fusion_mode == "later":
            x_n, x_g = x
            # share weight decode
            _, features_n = self.decode_head(x_n, return_feat=True)
            _, features_g = self.decode_head(x_g, return_feat=True)
            # siamese fuse
            changes = self.siamese_layer(features_n, features_g)
            try:
                loss_decode = self.decode_head.losses(changes, ground_truth)
            except Exception as e:
                print("Head class loss is not suit for change detection", e)
                loss_decode = self.decode_head.losses_change(changes, ground_truth)
        else:
            raise Exception("not support this siamese type ")
        losses.update(add_prefix(loss_decode, 'decode'))
        return losses

    def _constant_distance_forward_train(self, pair, ground_truth):
        # calculate point-wise correlation at each spatial location
        image_pair, anchor_pair = pair
        # filter channels
        image_feat, anchor_feat = image_pair[-1], anchor_pair[-1]
        # sparse distribution
        image_sparse_feat, anchor_sparse_feat = self._sparse_feats(image_feat), self._sparse_feats(anchor_feat)
        # w distant
        dist = torch.pairwise_distance(image_sparse_feat, anchor_sparse_feat)
        return dist
        pass

    @staticmethod
    def _sparse_feats(feature, distance_type='pixel'):
        h, w = feature.shape[2], feature.shape[3]
        feature = torch.nn.functional.upsample(feature, size=(h * 2, w * 2), mode='bilinear', align_corners=True)
        feature = torch.nn.functional.adaptive_avg_pool2d(feature, output_size=(int(np.ceil(h / 2)), int(np.ceil(w / 2))))
        return feature

    def _decode_head_forward_infer(self, x):
        """Run forward function and calculate loss for decode head in
        inference."""
        # trans tuple to list
        if self.fusion_mode == "mid":
            x = [self.siamese_layer[idx](x[0][idx], x[1][idx]) for idx in range(len(x[0]))]
            changes = self.decode_head.forward_infer(x)
        elif self.fusion_mode == "later":
            x_n, x_g = x
            _, features_n = self.decode_head(x_n, return_feat=True)
            _, features_g = self.decode_head(x_g, return_feat=True)
            changes = self.siamese_layer(features_n, features_g)
        else:
            raise Exception("not support this siamese type ")
        return changes

    def _auxiliary_head_forward_train(self, x, ground_truth):
        """Run forward function and calculate loss for auxiliary head in
        training."""
        losses = dict()
        if isinstance(self.auxiliary_head, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head):
                loss_aux = aux_head.forward_train(x, ground_truth)
                losses.update(add_prefix(loss_aux, f'aux_{idx}'))
        else:
            loss_aux = self.auxiliary_head.forward_train(
                x, ground_truth)
            losses.update(add_prefix(loss_aux, 'aux'))

        return losses

    def _constant_head_forward_train(self, distance, ground_truth):
        losses = dict()
        r_distance = resize(
            input=distance,
            size=ground_truth.shape[1:],
            mode='bilinear',
            align_corners=True)
        seg_label = ground_truth.unsqueeze(1)
        loss_cnt = self.constant_loss(r_distance, seg_label)
        losses.update(add_prefix(loss_cnt, 'cnt'))
        return losses

    @staticmethod
    def _switch_channel(inputs):
        index = list(range(int(inputs.shape[1])))
        random.shuffle(index)
        exchange_inputs = inputs[:, index, :, :]
        return exchange_inputs

    def forward_train(self, inputs, ground_truth):
        """Forward function for training.

        Parameters
        ----------
        inputs : Tensor
            Input images.
        ground_truth : Tensor
            Semantic segmentation masks
            used if the architecture supports semantic segmentation task.

        Returns
        -------
        dict[str, Tensor]
            a dictionary of loss components
        """
        if type(ground_truth) is dict:
            ground_truth = ground_truth['gt_masks']
        x = self.extract_feat(inputs)
        losses = dict()
        loss_decode = self._decode_head_forward_train(x, ground_truth)
        losses.update(loss_decode)

        if self.with_auxiliary_head:
            # change the location with 0, 1
            aux = [self.siamese_layer[idx](x[1][idx], x[0][idx]) for idx in range(len(x[0]))]
            loss_aux = self._auxiliary_head_forward_train(
                aux, ground_truth)
            losses.update(loss_aux)

        if self.with_constant_head:
            distance = F.pairwise_distance(x[0][-1], x[1][-1], keepdim=True)  # pixel-wise distance
            # pixel distance
            loss_cos = self._constant_head_forward_train(distance, ground_truth)
            losses.update(loss_cos)
            pass

        return losses

    def forward_infer(self, inputs, features=None):
        """
        forward infer
        Args:
            inputs:
            features: use for onnx trans

        Returns:
        """

        if torch.onnx.is_in_onnx_export():
            if self.fus_type == 'later':
                return self.encode_decode(inputs)
            else:
                if self.export_feature:
                    return self._get_template_feat(inputs)
                else:
                    return self._get_inference_feat(inputs, features)
        return self.encode_decode(inputs)

    def _get_template_feat(self, template):
        feats = self.backbone(template)
        if type(feats) is list:
            feats_que = []
            for idx in range(len(feats)):
                # squeeze the feats as N x C
                assert len(feats[idx].size()) == 4
                n, c, w, h = feats[idx].shape
                _feats = feats[idx].reshape(n, c * w * h)
                feats_que.append(_feats)
            feats_total = torch.cat(feats_que, dim=1)
            return feats_total
        elif type(feats) is torch.Tensor:
            n, c, w, h = feats.shape
            return feats.reshape(n, c * w * h)
        else:
            raise RuntimeError

    def _get_inference_feat(self, images, features):
        features_img = self.backbone(images)
        feats_cur = []
        # recover feature map
        for idx in range(len(features_img)):
            assert features is not None
            n, c, w, h = features_img[idx].shape
            if features.shape[1] == c * w * h:
                feats_left = features
            else:
                feats_left, feats_right = torch.split(features, c * w * h, dim=1)
                features = feats_right
            feats_single = feats_left.reshape(n, c, w, h)
            feats_cur.append(feats_single)

        x = [self.siamese_layer[idx](features_img[idx], feats_cur[idx])
             for idx in range(len(features_img))]
        changes = self.decode_head.forward_infer(x)
        changes = resize(
            input=changes,
            size=images.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        if self.fusion_mode == 'mid':
            changes = self.sft(changes)
        else:
            changes = self.sig(changes)
        changes = changes.permute(0, 2, 3, 1)
        return changes

    def forward(self, inputs, features=None, return_metrics=False, **kwargs):
        """Calls either :func:`forward_train` or :func:`forward_infer` depending
        on whether ``return_metrics`` is ``True``.
        """
        if return_metrics:
            metrics = self.forward_train(inputs, **kwargs)
            return self._parse_metrics(metrics)
        else:
            return self.forward_infer(inputs, features)









