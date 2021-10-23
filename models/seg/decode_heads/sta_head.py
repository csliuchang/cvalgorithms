import torch
import torch.nn as nn
from .decode_head import BaseDecodeHead
from ...builder import HEADS, build_loss
import torch.nn.functional as F
from models.base.blocks.self_attention_block import BasicAttentionModule
from models.base.blocks.conv_module import ConvModule


@HEADS.register_module()
class STAHead(BaseDecodeHead):
    def __init__(self, in_channels, loss, stride, dr_channels=[64, 128, 256, 512], out_channel=96, **kwargs):
        super(STAHead, self).__init__(in_channels, **kwargs)
        self.loss = build_loss(loss)
        self.decoder_n = Decoder(dr_channels, out_channel, f_chn=in_channels)
        self.decoder_g = Decoder(dr_channels, out_channel, f_chn=in_channels)
        self.SABlock = SAModule(in_channels, stride=stride)

    def init_weights(self):
        pass

    def forward(self, inputs):
        inputs_n, inputs_g = inputs
        inputs_n, inputs_g = self.decoder_n(inputs_n), self.decoder_g(inputs_g)
        feature_n, feature_g = self.SABlock(inputs_n, inputs_g)
        # feature distance
        distance = F.pairwise_distance(feature_n, feature_g, keepdim=True)
        distance = F.interpolate(distance, scale_factor=4, mode='bilinear', align_corners=True)
        return distance

    def losses(self, seg_logit, seg_label):
        loss = dict()
        seg_label[seg_label == 255] = -1
        seg_label[seg_label == 0] = 1
        loss['loss'] = self.loss(seg_logit, seg_label)
        return loss


class SAModule(nn.Module):
    """
    self attention module for change detection
    """
    def __init__(self, in_channel, stride=1, mode='BAM'):
        super(SAModule, self).__init__()
        self.in_chn = in_channel
        self.mode = mode
        if self.mode == 'BAM':
            self.Self_Att = BasicAttentionModule(self.in_chn, stride=stride)
        elif self.mode == 'PAM':
            pass
        else:
            print("Not support this mode !")

    def forward(self, n, g):
        height = n.shape[3]
        x = torch.cat((n, g), 3)
        x = self.Self_Att(x)
        return x[:, :, :, 0:height], x[:, :, :, height:]


class Decoder(nn.Module):
    def __init__(self, dr_channels=[64, 128, 256, 512], out_channel=96, f_chn=32):
        super(Decoder, self).__init__()

        for i in range(len(dr_channels)):
            self.add_module(name=f'dr{i+2}', module=ConvModule(dr_channels[i], out_channel, 3, 1, 1,
                                                               norm_cfg=dict(type='BN'),
                                                               act_cfg=dict(type='ReLU')), )

        self.last_conv = nn.Sequential(nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Conv2d(256, f_chn, kernel_size=1, stride=1, padding=0, bias=False),
                                       nn.BatchNorm2d(f_chn),
                                       nn.ReLU(),
                                       )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, feature_lists: list):
        f2, f3, f4, x = feature_lists
        x2, x3, x4, x = self.dr2(f2), self.dr3(f3), self.dr4(f4), self.dr5(x)

        x = F.interpolate(x, size=x2.size()[2:], mode='bilinear', align_corners=True)
        x3 = F.interpolate(x3, size=x2.size()[2:], mode='bilinear', align_corners=True)
        x4 = F.interpolate(x4, size=x2.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat((x, x2, x3, x4), dim=1)
        x = self.last_conv(x)
        return x






