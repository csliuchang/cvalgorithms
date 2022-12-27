from deepcv2.ds_network.builder import CHGDETECTORS
from .base import BaseChanger


@CHGDETECTORS.register_module()
class SiamDetEncoderDecoder(BaseChanger):

    def __init__(self):
        super(SiamDetEncoderDecoder, self).__init__()

    def extract_feat(self, inputs):
        pass

    def encode_decode(self, inputs):
        pass

    def forward_train(self, inputs, **kwargs):
        pass

    def forward_infer(self, inputs, **kwargs):
        pass

