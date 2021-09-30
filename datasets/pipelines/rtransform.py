import cv2
import numpy as np
from .transform import Resize, Rotate
from ..builder import PIPELINES
from engine.parallel import DataContainer as DC
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


def normalize(img, mean=None, std=None):
    mean, std = torch.tensor(mean), torch.tensor(std)
    if mean is None or std is None:
        return img
    return (img - mean)/std


@PIPELINES.register_module()
class RResize(Resize):
    """
        Resize  images & rotated bbox
        Inherit Resize pipeline class to handle rotated bboxes
    """

    def __init__(self, *args, **kwargs):
        super(RResize, self).__init__(*args, **kwargs)

    def _resize_img(self, results):
        image = results['img_info']
        image = cv2.resize(image, [self.resize_width, self.resize_height], interpolation=cv2.INTER_LINEAR)
        results['img_info'] = image
        results['image_shape'] = [self.resize_width, self.resize_height]

    def _resize_bboxes(self, results):
        original_height, original_width = results['ori_image_shape']
        bboxes = results['ann_info']['bboxes']
        width_ratio = float(self.resize_width) / original_width
        height_ratio = float(self.resize_height) / original_height
        new_bbox = []
        for bbox in bboxes:
            bbox = [int(np.clip(bbox[i], 0, original_height)*height_ratio) if i in [1, 3, 5, 7]
                    else int(np.clip(bbox[i], 0, original_width)*width_ratio) for i in range(len(bbox))]
            new_bbox.append(bbox)
        new_bbox = np.array(new_bbox, dtype=np.float32)
        results['ann_info']['bboxes'] = new_bbox


@PIPELINES.register_module()
class RRotate(Rotate):
    """
    RRotate pipeline for poly and bbox
    """
    def __init__(self, *args, **kwargs):
        super(RRotate, self).__init__(*args, **kwargs)

    def _rotate_annotation(self, results, M):

        height, width = results["image_shape"][1], results["image_shape"][0]
        rotate_bbox = []
        bboxes = results["ann_info"]["bboxes"]
        for pts in bboxes:
            pts = pts.reshape(4, 2)
            pts = np.concatenate(
                [pts, np.ones((pts.shape[0], 1))], axis=1)
            pts = np.matmul(pts, np.transpose(M)).reshape(-1)
            # clip pts
            pts = np.array([np.clip(pts[i], 0, height) if i in [1, 3, 5, 7]
                    else np.clip(pts[i], 0, width) for i in range(len(pts))], dtype=np.float32)
            rotate_bbox.append(pts)
        results["ann_info"]["bboxes"] = np.concatenate([rotate_bbox], axis=0)


@PIPELINES.register_module()
class Collect(object):
    """
    Collect data from the loader relevant to the specific task.

    This is usually the last stage of the data loader pipeline. Typically keys
    is set to some subset of "img", "proposals", "gt_bboxes",
    "gt_bboxes_ignore", "gt_labels", and/or "gt_masks".

    The "img_meta" item is always populated.  The contents of the "img_meta"
    dictionary depends on "meta_keys". By default this includes:

        - "img_shape": shape of the image input to the network as a tuple
            (h, w, c).  Note that images may be zero padded on the bottom/right
            if the batch tensor is larger than this shape.

        - "scale_factor": a float indicating the preprocessing scale

        - "flip": a boolean indicating if image flip transform was used

        - "filename": path to the image file

        - "ori_shape": original shape of the image as a tuple (h, w, c)

        - "pad_shape": image shape after padding

        - "img_norm_cfg": a dict of normalization information:
            - mean - per channel mean subtraction
            - std - per channel std divisor
            - to_rgb - bool indicating if bgr was converted to rgb
    """

    def __init__(self,
                 keys,
                 bg_first,
                 meta_keys=('filename', 'ori_image_shape', 'image_shape'),
                 fields=(dict(key='img', stack=True), dict(key='gt_bboxes'),
                         dict(key='gt_labels'))):
        self.fields = fields
        self.keys = keys
        self.meta_keys = meta_keys
        self.bg_first = bg_first

    def __call__(self, results):
        data = {}
        img_meta = {}
        # trans to tensor
        img = results.pop('img_info')
        if len(img.shape) == 2:
            img = img[..., None].repeat(3, -1)
        img = to_tensor(img)
        img = normalize(img, mean=results['mean'], std=results['std'])
        img = img.permute(2, 0, 1)
        results['img'] = img
        if 'bboxes' in results['ann_info']:
            results['gt_bboxes'] = np.array(results['ann_info']['bboxes'], dtype=np.float32)
            if 'image_shape' in results:
                results['gt_masks'] = np.ones(shape=results['image_shape'], dtype=np.uint8) * 0.
            else:
                results['gt_masks'] = np.ones(shape=results['ori_image_shape'], dtype=np.uint8) * 0.
            results['gt_labels'] = results['ann_info']['labels']
        elif 'polygons' in results:
            if 'masks' in self.keys:
                results['masks'] = polyline2masks(results, self.bg_first)
        for key in self.meta_keys:
            img_meta[key] = results.get(key, None)
        data['img_metas'] = img_meta
        for key in self.keys:
            data[key] = results[key]
        return data

    def __repr__(self):
        return self.__class__.__name__ + \
               f'(keys={self.keys}, meta_keys={self.meta_keys})'


def polyline2masks(results, bg_id=255, bg_first=False):
    """
    default background id is 0
    """
    if bg_first:
        bg_id = 0
    image_shape = results.get('image_shape', 'ori_image_shape')
    mask = np.ones(shape=image_shape, dtype=np.uint8) * bg_id
    for label_id, polyline in zip(results['ann_info']['labels'], results['polygons']):
        # color = int(label_id + 1)
        color = int(label_id + 1) if bg_first else int(label_id)
        cv2.fillPoly(mask, np.array([polyline], np.int32), color=color, lineType=cv2.LINE_4)

    return to_tensor(np.array(mask, dtype=np.int64))
