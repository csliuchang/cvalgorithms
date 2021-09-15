import inspect
import platform

from .blocks import *
from .plugin import abbreviation
from .registry import BLOCK_LAYERS


def build_block_layer(cfg, postfix='', anonymous=False, **kwargs):
    """Build block layer.

    Parameters
    ----------
    cfg : dict, optional
        cfg should contain:
            type (str): identify block layer type.
            layer args: args needed to instantiate a block layer.
    postfix : {int, str}
        appended into abbreviation to create named layer. Default: ''.

    Returns
    -------
    (name, layer) : tuple[str, nn.Module]
        name is abbreviation + postfix.
        layer is created block layer.
    """
    if not isinstance(cfg, dict):
        raise TypeError('cfg must be a dict')
    if 'type' not in cfg:
        raise KeyError('the cfg dict must contain the key "type"')
    cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')
    if layer_type not in BLOCK_LAYERS:
        raise KeyError(f'Unrecognized block type {layer_type}')

    block_layer = BLOCK_LAYERS.get(layer_type)
    layer = block_layer(**kwargs, **cfg_)

    if anonymous:
        return layer
    else:
        abbr = abbreviation(block_layer)
        assert isinstance(postfix, (int, str))
        name = abbr + str(postfix)
        return name, layer
