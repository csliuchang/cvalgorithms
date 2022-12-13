import cv2
import numpy as np
from .operate import Resize, Rotate
from ..builder import PIPELINES
import torch
from .utils import to_tensor, polyline2masks


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





