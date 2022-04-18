import copy

import cv2
import numpy as np
import inspect
from ..builder import PIPELINES
from .utils import polyline2masks
from abc import ABCMeta, abstractmethod
from typing import Optional, List, Any, Callable


__all__ = []


class Operate(metaclass=ABCMeta):
    """
    Base class for data augmentations,
    """
    def _set_attributes(self, params: Optional[List[Any]] = None) -> None:
        """
        Set attributes from the input list of parameters.

        Args:
            params (list): list of parameters.
        """
        if params:
            for k, v in params.items():
                if k != "self" and not k.startswith("_"):
                    setattr(self, k, v)

    @abstractmethod
    def apply_image(self, img: np.ndarray):
        """
        Apply the transform on an image.
                Args:
            img (ndarray): of shape NxHxWxC, or HxWxC or HxW. The array can be
                of type uint8 in range [0, 255], or floating point in range
                [0, 1] or [0, 255].
        Returns:
            ndarray: image after apply the transformation.
        """

    @abstractmethod
    def apply_coords(self, coords: np.ndarray):
        """
        Apply the transform on coordinates.

        Args:
            coords (ndarray): floating point array of shape Nx2. Each row is (x, y).

        Returns:
            ndarray: coordinates after apply the transformation.

        Note:
            The coordinates are not pixel indices. Coordinates inside an image of
            shape (H, W) are in range [0, W] or [0, H].
            This function should correctly transform coordinates outside the image as well.
        """

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        """
        Apply the transform on a full-image segmentation.
        By default will just perform "apply_image".

        Args:
            segmentation (ndarray): of shape HxW. The array should have integer
            or bool dtype.

        Returns:
            ndarray: segmentation after apply the transformation.
        """
        return self.apply_image(segmentation)

    def apply_box(self, boxes: np.ndarray) -> np.ndarray:
        """
        Apply the transform on an axis-aligned box. By default will transform
        the corner points and use their minimum/maximum to create a new
        axis-aligned box. Note that this default may change the size of your
        box, e.g. after rotations.

        Args:
            boxes (ndarray): Nx4 floating point array of XYXY format in absolute
                coordinates.
        Returns:
            ndarray: box after apply the transformation.

        Note:
            The coordinates are not pixel indices. Coordinates inside an image of
            shape (H, W) are in range [0, W] or [0, H].

            This function does not clip boxes to force them inside the image.
            It is up to the application that uses the boxes to decide.
        """
        pass

    def apply_rbox(self):
        """
        Apply the transform on an rotate box. By default will transform
        the corner points and use their minimum/maximum to create a new
        rotate box. Note that this default may change the size of your
        box, e.g. after rotations.

        Args:
            box (ndarray): Nx5 floating point array of XYXYA format in absolute
                coordinates.
        Returns:
            ndarray: box after apply the transformation.

        Note:
            The coordinates are not pixel indices. Coordinates inside an image of
            shape (H, W) are in range [0, W] or [0, H].

            This function does not clip boxes to force them inside the image.
            It is up to the application that uses the boxes to decide.
        """
        pass

    def apply_polygons(self, polygons: list) -> list:
        """
                Apply the transform on a list of polygons, each represented by a Nx2
        array. By default will just transform all the points.

        Args:
            polygons (list[ndarray]): each is a Nx2 floating point array of
                (x, y) format in absolute coordinates.
        Returns:
            list[ndarray]: polygon after apply the transformation.

        Note:
            The coordinates are not pixel indices. Coordinates on an image of
            shape (H, W) are in range [0, W] or [0, H].
        """
        return [self.apply_coords(p) for p in polygons]

    @classmethod
    def register_type(cls, data_type: str, func: Optional[Callable] = None):
        if func is None:

            def wrapper(decorated_func):
                assert decorated_func is not None
                cls.register_type(data_type, decorated_func)
                return decorated_func

            return wrapper

        assert callable(func), "func is not callable, check your func {}".format(func)
        assert len(inspect.getfullargspec(func)) == 2, "You can only register a function that takes two positional "

        setattr(cls, "apply_" + data_type, func)

    def inverse(self) -> "Operate":
        """
        Create a transform that inverts the geometric changes (i.e. change of
        coordinates) of this transform.

        Note that the inverse is meant for geometric changes only.
        The inverse of photometric transforms that do not change coordinates
        is defined to be a no-op, even if they may be invertible.

        Returns:
            Operate:
        """
        raise NotImplementedError

    def update(self):
        pass

    def __call__(self, data, **kwargs):
        assert "annotations" in data, \
            "kwargs not have annotation, check your data metas {}".format(data.keys())
        for key, value in data['annotations'].items():
            if value is not None and key in ['segmentation', 'rotate', 'detection']:
                rec = getattr(self, 'apply_' + str(key))(data)
                return rec
            else:
                raise

    def __repr__(self):
        """
        Produce something like:
        "MyTransform(field1={self.field1}, field2={self.field2})"
        """
        pass


@PIPELINES.register_module()
class Resize(Operate):
    """
    Resize the image to a target size
    """
    def __init__(self, scale: int, interp: str = None):
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self, results: dict):
        assert 'img' in result, "Key words not in result"
        img = results['img']
        img_scale = cv2.resize(img, [self.scale[0], self.scale[1]], interpolation=cv2.INTER_LINEAR)
        results['img'] = img_scale
        return results

    def apply_coords(self, coords: np.ndarray, image_size=None):
        if image_size is None:
            image_size = self.scale
        coords[0] = coords[0] * (self.scale[0] * 1.0 / image_size[0])
        coords[1] = coords[1] * (self.scale[1] * 1.0 / image_size[1])
        return coords

    def inverse(self) -> "Operate":
        pass

    def apply_segmentation(self, results: dict):
        image_size = results['image_size']
        for poly in results['annotations']['segmentation']:
            results['annotations']['segmentation'] = \
                [self.apply_coords(p, image_size) for p in poly]
        results = self.apply_image(data)
        return results

    def apply_rotate(self, boxes: np.ndarray) -> np.ndarray:
        pass

    def apply_detection(self):
        pass


@PIPELINES.register_module()
class Resize_old(object):
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
        results['image_shape'] = [self.resize_width, self.resize_height]

    def _resize_bboxes(self, results):
        original_height, original_width = results['ori_image_shape']
        width_ratio = float(self.resize_height) / original_width
        height_ratio = float(self.resize_width) / original_height
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
            # del (results["ann_info"]['masks'])
            results["masks"] = new_mask
        else:
            raise Exception('not right format in results')

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
            image[:, :, c] = cv2.warpAffine(image[:, :, c], M, (width, height),
                                            flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REFLECT)

    def _rotate_annotation(self, results, M):
        if 'masks' in results:
            masks = results["masks"]
            results["masks"] = cv2.warpAffine(masks, M, (masks.shape[0], masks.shape[1]),
                                              flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)
        else:
            masks = polyline2masks(results, bg_first=True, tensor=False)
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
class Crop(object):
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        pass




@PIPELINES.register_module()
class CutPaste(object):
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        pass
