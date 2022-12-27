from deepcv2.common import Registry, build_from_cfg

CHGDETECTORS = Registry('chgdetector')
BACKBONES = Registry('backbone')
HEADS = Registry('head')
NECKS = Registry('neck')
LOSSES = Registry('loss')
METRICS = Registry('metric')
PIXEL_SAMPLERS = Registry('pixel sampler')


def build_neck(cfg, update_args=None):
    return build_from_cfg(cfg, NECKS, update_args=update_args)


def build_head(cfg, update_args=None):
    return build_from_cfg(cfg, HEADS, update_args=update_args)


def build_backbone(cfg, update_args=None):
    return build_from_cfg(cfg, BACKBONES, update_args=update_args)


def build_loss(cfg, update_args=None):
    return build_from_cfg(cfg, LOSSES, update_args=update_args)


def build_metric(cfg, update_args=None):
    return build_from_cfg(cfg, METRICS, update_args=update_args)


def build_chgdetector(cfg, update_args=None):
    return build_from_cfg(cfg, CHGDETECTORS, update_args=update_args)


def build_pixel_sampler(cfg, **default_args):
    """Build pixel sampler for segmentation map."""
    return build_from_cfg(cfg, PIXEL_SAMPLERS, default_args)
