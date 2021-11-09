import cv2
import numpy as np
import torch



def to_tensor(data):
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.
    """
    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError(f'type {type(data)} cannot be converted to tensor.')


def polyline2masks(results, bg_id=255, bg_first=False, tensor=True):
    """
    default background id is 0
    """
    if bg_first:
        bg_id = 0
    image_shape = results.get('image_shape', 'ori_image_shape')
    mask = np.ones(shape=image_shape, dtype=np.uint8) * bg_id
    for label_id, polyline in zip(results['ann_info']['labels'], results['polygons']):
        # color = int(label_id + 1)
        color = int(label_id) if bg_first else int(label_id)
        cv2.fillPoly(mask, np.array([polyline], np.int32), color=color, lineType=cv2.LINE_4)
    if tensor:
        return to_tensor(np.array(mask, dtype=np.int64))
    else:
        return np.array(mask, dtype=np.uint8)