import torch
import torch.nn as nn
from .decode_head import BaseDecodeHead
from ...builder import HEADS, build_loss
import torch.nn.functional as F
from ...utils import resize
from models.base.blocks.self_attention_block import BasicAttentionModule
from models.base.blocks.conv_module import ConvModule


@HEADS.register_module()
class STAHead(BaseDecodeHead):
    def __init__(self, in_channels, loss, dr_channels=[64, 128, 256, 512], out_channel=96, **kwargs):
        super(STAHead, self).__init__(in_channels, **kwargs)
        self.loss = build_loss(loss)
        self.decoder = Decoder(dr_channels, out_channel, f_chn=in_channels)

    def init_weights(self):
        pass

    def forward(self, inputs):
        final = self.decoder(inputs)
        return final

    def losses(self, seg_logit, seg_label):
        loss = dict()
        seg_logit = resize(
            input=seg_logit,
            size=seg_label.shape[1:],
            mode='bilinear',
            align_corners=self.align_corners)
        loss['loss'] = self.loss(seg_logit, seg_label)
        return loss

    def forward_train(self, inputs, **kwargs):
        """Forward function for training.

        Parameters
        ----------
        inputs : list[Tensor]
            List of multi-level img features.

        Returns
        -------
        dict[str, Tensor]
            a dictionary of loss components
        """
        return self.forward(inputs)


class Decoder(nn.Module):
    def __init__(self, dr_channels=[64, 128, 256, 512], out_channel=96, f_chn=32):
        super(Decoder, self).__init__()

        for i in range(len(dr_channels)):
            self.add_module(name=f'dr{i + 2}', module=ConvModule(dr_channels[i], out_channel, 3, 1, 1,
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
