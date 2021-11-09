import torch
import torch.nn as nn
from .decode_head import BaseDecodeHead
from ...builder import HEADS, build_loss
import torch.nn.functional as F
from ...utils import resize, normal_init
from models.base.blocks.comm_blocks import BasicConv2d


@HEADS.register_module()
class AlignHead(BaseDecodeHead):
    def __init__(self, in_channels: list, loss, head_width, **kwargs):
        super(AlignHead, self).__init__(in_channels,  head_width, **kwargs)
        res_layers_a, res_layers_b, align_blocks, head_list = [], [], [], []
        for i in range(len(in_channels)):
            res_layer_a = ResBlock(in_channels[::-1][i], head_width)
            res_layer_b = ResBlock(head_width, head_width)
            align_block = AlignFeatAgg(head_width)
            head = nn.Conv2d(head_width, self.num_classes, kernel_size=3, stride=1, padding=1)
            self.add_module(f"ResBlock_{i+1}a", res_layer_a)
            res_layers_a.append(res_layer_a)
            if i >= 1:
                self.add_module(f"ResBlock_{i+1}b", res_layer_b)
                res_layers_b.append(res_layer_b)
                self.add_module(f"AlignFeatAgg_{i}", align_block)
                align_blocks.append(align_block)
                self.add_module(f"head{i}", head)
                head_list.append(head)
        self.AlignFeatAgg_high = AlignFeatAgg(head_width, in_channels[0])
        self.head = nn.Conv2d(head_width, self.num_classes, kernel_size=3, stride=1, padding=1)
        self.res_layers_a = res_layers_a
        self.res_layers_b = res_layers_b
        self.align_blocks = align_blocks
        self.head_list = head_list
        self.loss = build_loss(loss)

    def forward(self, inputs, return_feat=False):
        x1, x2, x3, x4 = inputs
        x5 = self.AlignFeatAgg_high(x4)
        feature_list = [x1, x2, x3, x4, x5]
        per_features = self.res_layers_a[0](x1)
        supervision_list = []
        for idx, (res_layers_a, res_layers_b, align_block) in enumerate(
                zip(self.res_layers_a[1:], self.res_layers_b, self.align_blocks)):
            per_features = res_layers_b(align_block(per_features, res_layers_a(feature_list[idx])))
            supervision_list.append(per_features)
        if return_feat:
            return per_features
        else:
            if self.training:
                final_list = []
                for idx, supervision in enumerate(supervision_list):
                    out = self.head_list[idx](supervision)
                    final_list.append(out)
                return final_list
            else:
                return self.head_list[-1](per_features)

    def losses(self, seg_logit, seg_label):
        loss = dict()
        loss_value = 0
        for seg in seg_logit:
            seg = resize(
                input=seg,
                size=seg_label.shape[1:],
                mode='bilinear',
                align_corners=self.align_corners)
            loss_v = self.loss(seg, seg_label)
            loss_value += loss_v
        loss['loss'] = loss_value
        return loss


class ResBlock(nn.Module):
    """
    A residual block with three convs
    """

    def __init__(self, in_feat, out_feat):
        super(ResBlock, self).__init__()

        self.unify = nn.Conv2d(in_feat, out_feat, 1)
        self.residual = nn.Sequential(
            BasicConv2d(out_feat, out_feat // 4, 3, 1, 1),
            nn.Conv2d(out_feat // 4, out_feat, 3, 1, 1, bias=False)
        )
        self.norm = nn.BatchNorm2d(out_feat)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, feats):
        feats = self.unify(feats)
        residual = self.residual(feats)
        return self.relu(self.norm(feats + residual))


class AlignFeatAgg(nn.Module):
    def __init__(self, in_feat, feat=None):
        super(AlignFeatAgg, self).__init__()
        self.delta_gen_1 = nn.Sequential(
            BasicConv2d(in_feat * 2, in_feat, kernel_size=1),
            nn.Conv2d(in_feat, 2, kernel_size=3, padding=1, stride=1, bias=False))
        self.delta_gen_2 = nn.Sequential(
            BasicConv2d(in_feat * 2, in_feat, kernel_size=1),
            nn.Conv2d(in_feat, 2, kernel_size=3, padding=1, stride=1, bias=False))

        if feat:
            self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(output_size=(3, 3)),
                                      BasicConv2d(feat, feat, 1),
                                      BasicConv2d(feat, in_feat, kernel_size=3, padding=1, stride=1))
            self.adapt = nn.Sequential(
                BasicConv2d(feat, feat, kernel_size=1),
                BasicConv2d(feat, in_feat, kernel_size=3, padding=1, stride=1)
            )

        # init weights
        self.delta_gen_1[1].weight.data.zero_()
        self.delta_gen_2[1].weight.data.zero_()

    @staticmethod
    def _bilinear_interpolate_torch_grid_sample(in_feats, size, delta):
        out_h, out_w = size
        n, c, h, w = in_feats.shape
        s = 1.0
        norm = torch.tensor([[[[h / s, w / s]]]]).type_as(in_feats).to(in_feats.device)
        w_list = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        h_list = torch.linspace(-1.0, -1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((h_list.unsqueeze(2), w_list.unsqueeze(2)), dim=2)
        grid = grid.repeat(n, 1, 1, 1).type_as(in_feats).to(in_feats.device)
        grid = grid + delta.permute(0, 2, 3, 1) / norm

        output = F.grid_sample(in_feats, grid, align_corners=True)
        return output

    def forward(self, low, high=None):
        h, w = low.size(2), low.size(3)
        if high is None:
            high = self._high_feature_exa(low, h, w)
        else:
            high = self._multi_feature_exa(low, high, h, w)
        return high

    def _high_feature_exa(self, low, h, w):
        high = self.pool(low)
        low = self.adapt(low)
        high_stage_up = F.interpolate(input=high, size=(h, w), mode='bilinear', align_corners=True)
        concat = torch.cat((low, high_stage_up), 1)
        delta = self.delta_gen_1(concat)
        high = self._bilinear_interpolate_torch_grid_sample(high, (h, w), delta)
        high += low
        return high

    def _multi_feature_exa(self, low, high, h, w):
        high = F.interpolate(input=high, size=(h, w), mode='bilinear', align_corners=True)
        concat = torch.cat((low, high), 1)
        delta_1 = self.delta_gen_1(concat)
        delta_2 = self.delta_gen_2(concat)
        high = self._bilinear_interpolate_torch_grid_sample(high, (h, w), delta_1)
        low = self._bilinear_interpolate_torch_grid_sample(low, (h, w), delta_2)
        high += low
        return high

