from utils import Registry, build_from_cfg

SIAMESE_LAYER = Registry('siamese_layer')


def build_siamese_layer(cfg):
    return build_from_cfg(cfg, SIAMESE_LAYER, )
