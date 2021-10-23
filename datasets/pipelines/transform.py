import cv2
import numpy as np
from ..builder import PIPELINES


@PIPELINES.register_module()
class Resize(object):
    """
        Resize images & rotated bbox
        Inherit Resize pipeline class to handle rotated bboxes
    """

    def __init__(self, img_scale):
        self.scale = img_scale
        self.resize_height, self.resize_width = self.scale

    def _resize_img(self, results):
        image = results['img_info']
        image = cv2.resize(image, [self.resize_height, self.resize_width], interpolation=cv2.INTER_LINEAR)
        results['img_info'] = image
        results['image_shape'] = [self.resize_height, self.resize_width]

    def _resize_bboxes(self, results):
        original_height, original_width = results['ori_image_shape']
        width_ratio = float(self.resize_width) / original_width
        height_ratio = float(self.resize_height) / original_height
        if "bboxes" in results["ann_info"]:
            new_bbox = []
            for bbox in results["ann_info"]["bboxes"]:
                bbox[0] = int(bbox[0] * width_ratio)
                bbox[2] = int(bbox[2] * width_ratio)
                bbox[1] = int(bbox[1] * height_ratio)
                bbox[3] = int(bbox[3] * height_ratio)
                new_bbox.append(bbox)
            new_bbox = np.array(new_bbox, dtype=np.float32)
            results['ann_info']['bboxes'] = new_bbox
        elif "polylines" in results["ann_info"]:
            new_polylines = []
            for polyline in results["ann_info"]["polylines"]:
                new_polylines.append([[poly[0] * width_ratio, poly[1] * height_ratio] for poly in polyline])
            results['polygons'] = new_polylines
        elif "masks" in results["ann_info"]:
            mask = np.array(results["ann_info"]['masks'], dtype=np.uint8)
            new_mask = cv2.resize(mask, [self.resize_height, self.resize_width], interpolation=cv2.INTER_NEAREST)
            del(results["ann_info"]['masks'])
            results["masks"] = new_mask

    def __call__(self, results):
        self._resize_img(results)
        self._resize_bboxes(results)
        return results


@PIPELINES.register_module()
class Rotate(object):
    """
    Rotate
    """
    def __init__(self, angle):
        self.angle = self._get_angle(angle)

    def _get_angle(self, angle):
        angle = np.random.randint(-angle, angle)
        return angle

    def _get_matrix(self, image):
        width, height = image.shape[1], image.shape[0]
        M = cv2.getRotationMatrix2D((width, height), self.angle, 1.0)
        return M

    def _rotate_image(self, image, M):
        height, width, channel = image.shape[0], image.shape[1], image.shape[2]
        for c in range(channel):
            image[:, :, c] = cv2.warpAffine(image[:, :, c], M, (height, width),
                                            flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REFLECT)

    def _rotate_annotation(self, results, M):
        masks = results["masks"]
        results["masks"] = cv2.warpAffine(masks, M, (masks.shape[0], masks.shape[1]),
                                          flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)
        return masks

    def __call__(self, results, *args, **kwargs):
        image = results["img_info"]
        M = self._get_matrix(image)
        self._rotate_image(image, M)
        self._rotate_annotation(results, M)
        return results


@PIPELINES.register_module()
class CutPaste(object):
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        pass
