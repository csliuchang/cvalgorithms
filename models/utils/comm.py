import torch.distributed as dist
from torch.nn import functional as F
from functools import partial
from torch import nn
import torch


BatchNorm2d = torch.nn.BatchNorm2d


def get_activation_fn(activation):
    """
    Return an activation function given a string
    """
    if activation == 'relu':
        return F.relu
    if activation == 'gelu':
        return F.gelu
    if activation == 'glu':
        return F.glu
    else:
        raise RuntimeError(F"no activation{activation} in there")


def get_activation(activation):
    """
    Only support `ReLU` and `LeakyReLU` now.

    Args:
        activation (str or callable):

    Returns:
        nn.Module: the activation layer
    """

    act = {
        "ReLU": nn.ReLU,
        "LeakyReLU": nn.LeakyReLU,
    }[activation]
    if activation == "LeakyReLU":
        act = partial(act, negative_slope=0.1)
    return act(inplace=True)


def get_norm(norm, out_channels, **kwargs):
    """
    Args:
        norm (str or callable): either one of BN, SyncBN, FrozenBN, GN;
            or a callable that takes a channel number and returns
            the normalization layer as a nn.Module.
        kwargs: Additional parameters in normalization layers,
            such as, eps, momentum

    Returns:
        nn.Module or None: the normalization layer
    """
    if norm is None:
        return None
    if isinstance(norm, str):
        if len(norm) == 0:
            return None
        norm = {
            "BN": BatchNorm2d,
            # Fixed in https://github.com/pytorch/pytorch/pull/36382
            "SyncBN": nn.SyncBatchNorm,
            # "FrozenBN": FrozenBatchNorm2d, # TODO detectron2
            "GN": lambda channels: nn.GroupNorm(32, channels),
            # for debugging:
            "nnSyncBN": nn.SyncBatchNorm,
        }[norm]
    return norm(out_channels, **kwargs)


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def c2_xavier_fill(module: nn.Module) -> None:
    """
    Initialize `module.weight` using the "XavierFill" implemented in Caffe2.
    Also initializes `module.bias` to 0.

    Args:
        module (torch.nn.Module): module to initialize.
    """
    # Caffe2 implementation of XavierFill in fact
    # corresponds to kaiming_uniform_ in PyTorch
    nn.init.kaiming_uniform_(module.weight, a=1)  # pyre-ignore
    if module.bias is not None:  # pyre-ignore
        nn.init.constant_(module.bias, 0)


def nonzero_tuple(x):
    """
    A 'as_tuple=True' version of torch.nonzero to support torchscript.
    because of https://github.com/pytorch/pytorch/issues/38718
    """
    if torch.jit.is_scripting():
        if x.dim() == 0:
            return x.unsqueeze(0).nonzero().unbind(1)
        return x.nonzero().unbind(1)
    else:
        return x.nonzero(as_tuple=True)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


def weight_reduce_loss(loss, weight=None, reduction='mean', avg_factor=None):
    """Apply element-wise weight and reduce loss.

    Parameters
    ----------
    loss : Tensor
        Element-wise loss.
    weight : Tensor
        Element-wise weights.
    reduction : str
        Same as built-in losses of PyTorch.
    avg_factor : float
        Avarage factor when computing the mean of losses.

    Returns
    -------
    Tensor
        Processed loss values.
    """
    if weight is not None:
        assert loss.sum() >= 0., f'the before weight loss sum is {loss.sum()}'
        loss = loss * weight

    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        if reduction == 'mean':
            assert loss.sum() >= 0., f'the loss sum is {loss.sum()}'
            assert avg_factor >= 0., f'the avg_factor is {avg_factor}'
            loss = loss.sum() / avg_factor
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss


def reduce_loss(loss, reduction):
    """Reduce loss as specified.

    Parameters
    ----------
    loss : Tensor
        Elementwise loss tensor.
    reduction : str
        Options are "none", "mean" and "sum".

    Returns
    -------
    Tensor
        Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()


def images_to_levels(target, num_levels):
    target = torch.stack(target, 0)
    level_targets = []
    start = 0
    for n in num_levels:
        end = start + n
        level_targets.append(target[:, start:end])
        start = end
    return level_targets


def multi_apply(func, *args, **kwargs):
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))


def padding_results(results, nums_pre, nums_tensor):
    """
    tensor padding results
    """
    padding_results_list = []
    for result in results:
        len_results = result.shape[1]
        final_result = torch.zeros(size=(1, nums_pre, nums_tensor))
        final_result[:, :len_results, :] = result
        padding_results_list.append(final_result)
    return torch.cat(padding_results_list, dim=0)


def unmap(data, count, inds, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """
    if data.dim() == 1:
        ret = data.new_full((count, ), fill)
        ret[inds.type(torch.bool)] = data
    else:
        new_size = (count, ) + data.size()[1:]
        ret = data.new_full(new_size, fill)
        ret[inds.type(torch.bool), :] = data
    return ret


def fp16_clamp(x, min=None, max=None):
    if not x.is_cuda and x.dtype == torch.float16:
        # clamp for cpu float16, tensor fp16 has no clamp implementation
        return x.float().clamp(min, max).half()

    return x.clamp(min, max)


def get_bbox_dim(bbox_type, with_score=False):
    if bbox_type == 'hbb':
        dim = 4
    elif bbox_type == 'obb':
        dim = 5
    elif bbox_type == 'poly':
        dim = 8
    else:
        raise ValueError(f"don't know {bbox_type} bbox dim")

    if with_score:
        dim += 1
    return dim


def arb2result(bboxes, labels, num_classes, bbox_type='hbb'):
    assert bbox_type in ['hbb', 'obb', 'poly']
    bbox_dim = get_bbox_dim(bbox_type, with_score=True)

    if bboxes.shape[0] == 0:
        return [np.zeros((0, bbox_dim), dtype=np.float32) for i in range(num_classes)]
    else:
        bboxes = bboxes.cpu().numpy()
        labels = labels.cpu().numpy()
        return [bboxes[labels == i, :] for i in range(num_classes)]
