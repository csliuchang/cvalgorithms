import torch
import torch.nn as nn
from .decode_head import BaseDecodeHead
from ...builder import HEADS, build_loss
import torch.nn.functional as F
import math
from ...utils import resize
from models.seg.decode_heads.fcn_head import FCNHead

BatchNorm2d = nn.BatchNorm2d


@HEADS.register_module()
class STDCHead(BaseDecodeHead):
    def __init__(self, in_channels, mid_channels, conv_out_channels, sp16_in_channels, sp8_in_channels,
                 sp4_in_channels, sp2_in_channels, loss, bound_loss, stride=[2, 4, 8, 16, 32], use_boundary_2=True,
                 use_boundary_4=True,
                 use_boundary_8=True, **kwargs):
        super(STDCHead, self).__init__(in_channels, **kwargs)
        self.arm16 = AttentionRefinementModule(mid_channels, 128)
        self.arm32 = AttentionRefinementModule(in_channels, 128)
        self.avg_pool32 = nn.AdaptiveAvgPool2d(1)
        self.conv_head32 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
        self.conv_head16 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
        self.conv_avg = ConvBNReLU(in_channels, 128, ks=1, stride=1, padding=0)

        self.use_boundary_2 = use_boundary_2
        self.use_boundary_4 = use_boundary_4
        self.use_boundary_8 = use_boundary_8
        self.stride = stride

        inplane = sp8_in_channels + conv_out_channels

        self.ffm = FeatureFusionModule(inplane, 256)
        self.conv_out = BiSeNetOutput(256, 256, self.num_classes)
        self.conv_out16 = BiSeNetOutput(conv_out_channels, 64, self.num_classes)
        self.conv_out32 = BiSeNetOutput(conv_out_channels, 64, self.num_classes)

        self.conv_out_sp16 = BiSeNetOutput(sp16_in_channels, 64, 1)

        self.conv_out_sp8 = BiSeNetOutput(sp8_in_channels, 64, 1)
        self.conv_out_sp4 = BiSeNetOutput(sp4_in_channels, 64, 1)
        self.conv_out_sp2 = BiSeNetOutput(sp2_in_channels, 64, 1)

        self.loss_p = build_loss(loss)
        self.loss_2 = build_loss(loss)
        self.loss_3 = build_loss(loss)
        self.loss_bound_1 = build_loss(bound_loss)
        self.loss_bound_2 = build_loss(bound_loss)
        self.loss_bound_3 = build_loss(bound_loss)
        self.init_weights()

    def init_weights(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def forward(self, inputs):
        feat2, feat4, feat8, feat16, feat32 = inputs
        avg = self.avg_pool32(feat32)
        _, _, feat32_h, feat32_w = feat32.shape
        _, _, feat16_h, feat16_h = feat16.shape
        _, _, feat8_h, feat8_h = feat8.shape
        ori_h, ori_w = feat32_h * self.stride[-1], feat32_w * self.stride[-1]
        avg = self.conv_avg(avg)
        avg_up = F.interpolate(avg, size=(feat32_h, feat32_w), mode='nearest')

        feat32_arm = self.arm32(feat32)
        feat32_sum = feat32_arm + avg_up
        feat32_up = F.interpolate(feat32_sum, size=(feat16_h, feat16_h), mode='nearest')
        feat32_up = self.conv_head32(feat32_up)

        feat16_arm = self.arm16(feat16)
        feat16_sum = feat16_arm + feat32_up
        feat16_up = F.interpolate(feat16_sum, size=(feat8_h, feat8_h), mode='nearest')
        feat16_up = self.conv_head16(feat16_up)

        feat_out_sp2 = self.conv_out_sp2(feat2)

        feat_out_sp4 = self.conv_out_sp4(feat4)

        feat_out_sp8 = self.conv_out_sp8(feat8)

        # feat_out_sp16 = self.conv_out_sp16(feat16)

        feat_fuse = self.ffm(feat8, feat16_up)

        feat_out = self.conv_out(feat_fuse)
        feat_out16 = self.conv_out16(feat16_up)
        feat_out32 = self.conv_out32(feat32_up)

        feat_out = F.interpolate(feat_out, size=(ori_h, ori_w), mode='bilinear', align_corners=True)
        feat_out16 = F.interpolate(feat_out16, size=(ori_h, ori_w), mode='bilinear', align_corners=True)
        feat_out32 = F.interpolate(feat_out32, size=(ori_h, ori_w), mode='bilinear', align_corners=True)

        if self.use_boundary_2 and self.use_boundary_4 and self.use_boundary_8 and self.training:
            return feat_out, feat_out16, feat_out32, feat_out_sp2, feat_out_sp4, feat_out_sp8  # x8, x16
        else:
            return feat_out

    def losses(self, seg_logit, seg_label):

        loss = dict()
        p1 = self.loss_p(seg_logit[0], seg_label)
        p2 = self.loss_2(seg_logit[1], seg_label)
        p3 = self.loss_3(seg_logit[2], seg_label)
        b1 = self.loss_bound_1(seg_logit[3], seg_label)
        b2 = self.loss_bound_2(seg_logit[4], seg_label)
        b3 = self.loss_bound_3(seg_logit[5], seg_label)
        loss['loss'] = p1 + p2 + p3 + b1 + b2 + b3
        return loss


class AttentionRefinementModule(nn.Module):
    def __init__(self, in_chan, out_chan, *args, **kwargs):
        super(AttentionRefinementModule, self).__init__()
        self.conv = ConvBNReLU(in_chan, out_chan, ks=3, stride=1, padding=1)
        self.ave_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv_atten = nn.Conv2d(out_chan, out_chan, kernel_size=1, bias=False)
        self.bn_atten = BatchNorm2d(out_chan)
        # self.bn_atten = BatchNorm2d(out_chan, activation='none')
        self.sigmoid_atten = nn.Sigmoid()
        self.init_weight()

    def forward(self, x):
        feat = self.conv(x)
        atten = self.ave_pool(feat)
        atten = self.conv_atten(atten)
        atten = self.bn_atten(atten)
        atten = self.sigmoid_atten(atten)
        out = torch.mul(feat, atten)
        return out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, *args, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan,
                              out_chan,
                              kernel_size=ks,
                              stride=stride,
                              padding=padding,
                              bias=False)
        self.bn = BatchNorm2d(out_chan)
        # self.bn = BatchNorm2d(out_chan, activation='none')
        self.relu = nn.ReLU()
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


class FeatureFusionModule(nn.Module):
    def __init__(self, in_chan, out_chan, *args, **kwargs):
        super(FeatureFusionModule, self).__init__()
        self.convblk = ConvBNReLU(in_chan, out_chan, ks=1, stride=1, padding=0)
        self.conv1 = nn.Conv2d(out_chan,
                               out_chan // 4,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=False)
        self.conv2 = nn.Conv2d(out_chan // 4,
                               out_chan,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.init_weight()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, fsp, fcp):
        fcat = torch.cat([fsp, fcp], dim=1)
        feat = self.convblk(fcat)
        # atten = F.avg_pool2d(feat, (56, 56))
        atten = self.avg_pool(feat)
        atten = self.conv1(atten)
        atten = self.relu(atten)
        atten = self.conv2(atten)
        atten = self.sigmoid(atten)
        feat_atten = torch.mul(feat, atten)
        feat_out = feat_atten + feat
        return feat_out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


class BiSeNetOutput(nn.Module):
    def __init__(self, in_chan, mid_chan, n_classes, *args, **kwargs):
        super(BiSeNetOutput, self).__init__()
        self.conv = ConvBNReLU(in_chan, mid_chan, ks=3, stride=1, padding=1)
        self.conv_out = nn.Conv2d(mid_chan, n_classes, kernel_size=1, bias=False)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.conv_out(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


@HEADS.register_module()
class STDCSimple(FCNHead):
    def __init__(self, boundary_threshold=0.1, **kwargs):
        super(STDCSimple, self).__init__(**kwargs)
        self.boundary_threshold = boundary_threshold
        # Using register buffer to make laplacian kernel on the same
        # device of `seg_label`.
        self.register_buffer(
            'laplacian_kernel',
            torch.tensor([-1, -1, -1, -1, 8, -1, -1, -1, -1],
                         dtype=torch.float32,
                         requires_grad=False).reshape((1, 1, 3, 3)))
        self.fusion_kernel = torch.nn.Parameter(
            torch.tensor([[6. / 10], [3. / 10], [1. / 10]],
                         dtype=torch.float32).reshape(1, 3, 1, 1),
            requires_grad=False)
        self.stride_list = [1, 2, 4]

    def losses(self, seg_logit, seg_label):
        seg_label = seg_label.unsqueeze(1).float()

        boundary_out = []
        for idx in range(len(self.stride_list)):
            boundary_targets = F.conv2d(
                seg_label, self.laplacian_kernel, stride=self.stride_list[idx], padding=1)
            boundary_targets = boundary_targets.clamp(min=0)
            boundary_targets_up = resize(
                boundary_targets,
                size=seg_label.size()[2:],
                mode='nearest')
            boundary_targets_up[
                boundary_targets_up > self.boundary_threshold] = 1
            boundary_targets_up[
                boundary_targets_up <= self.boundary_threshold] = 0
            boundary_out.append(boundary_targets_up)
        boudary_targets_pyramids = torch.stack(boundary_out, dim=1)

        boundary_targets_pyramids = boudary_targets_pyramids.squeeze(2)
        boundary_targets_pyramid = F.conv2d(boundary_targets_pyramids,
                                            self.fusion_kernel)
        seg_logit = F.interpolate(
            seg_logit,
            seg_label.shape[2:],
            mode='bilinear',
            align_corners=True)
        loss = super(STDCSimple, self).losses(seg_logit,
                                              boundary_targets_pyramid.squeeze(1).long())
        return loss
