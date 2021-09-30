import cv2

from .base_dataset import BaseDataset
import os
import numpy as np
from .builder import DATASETS
import json


CLASS = ('df',  'guaidian')
COLOR = [(204, 78, 210), [20, 220, 20]]


@DATASETS.register_module()
class DetDatasets(BaseDataset):
    def __init__(self, *args, **kwargs):
        self.category = CLASS
        self.num_classes = len(self.category)
        self.color = COLOR
        self.cls_map = {c: i for i, c in enumerate(self.category)}
        super(DetDatasets, self).__init__(*args, **kwargs)

    def load_annotations(self, ann_file):
        data_infos = []
        lines = [line.strip() for line in open(ann_file, 'r').readlines()]
        gt_bboxes, gt_labels, gt_poly = [], [], []
        for line in lines:
            data_info = dict()
            line_parts = line.split('\t')
            img_rel_path = line_parts[0]
            data_info['filename'] = img_rel_path
            label = line_parts[1]
            label_file = os.path.join(self.data_root, label)
            json_data = json.load(open(label_file))
            for shape in json_data["shapes"]:
                cls_name = shape["label"]
                points = shape["points"]
                if len(points) == 5:
                    bbox = points2bbox(points)
                else:
                    bbox = []
                label = self.cls_map[cls_name]
                gt_labels.append(label)
                gt_poly.append(gt_poly)
                gt_bboxes.append(bbox)
        return data_infos


def points2bbox(points):
    rect = cv2.minAreaRect(np.array(points, np.int32))
    bbox = np.array([rect[0][0], rect[0][1], rect[1][0], rect[1][1]], dtype=np.float32)
    return bbox


if __name__ == "__main__":
    data_root = '/home/pupa/PycharmProjects/LightRotateDet/data/msratd500'
    DetDatasets(data_root)