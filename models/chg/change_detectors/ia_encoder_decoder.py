from .base import BaseSegmentor


class IAEncoderDecoder(BaseSegmentor):
    def __init__(self):
        super(IAEncoderDecoder, self).__init__()
        pass

    def _decode_head_forward_train(self, x, gt_semantic_seg):
        pass

    def forward_train(self, img, ground_truth):
        x = self.extract_feat(img)

        losses = dict()
        loss_decode = self._decode_head_forward_train(
            x, ground_truth
        )

        losses.update(loss_decode)