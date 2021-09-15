import cv2

from .base_dataset import BaseDataset
import os
import numpy as np
from .builder import DATASETS


CLASS = ('text', )
COLOR = [(204, 78, 210)]


@DATASETS.register_module()
class RDetDatasets(BaseDataset):
    def __init__(self, *args, **kwargs):
        self.category = CLASS
        self.num_classes = len(self.category)
        self.color = COLOR
        self.cls_map = {c: i for i, c in enumerate(self.category)}
        super(RDetDatasets, self).__init__(*args, **kwargs)

    def load_annotations(self, ann_file):
        data_infos = []
        lines = [line.strip() for line in open(ann_file, 'r').readlines()]
        for line in lines:
            data_info = dict()
            line_parts = line.split('\t')
            img_rel_path = line_parts[0]
            data_info['filename'] = img_rel_path
            label = line_parts[1]
            label_file = os.path.join(self.data_root, label)
            boxes = [line.strip() for line in open(label_file, 'r', encoding='utf-8-sig').readlines() if line != '']
            gt_bboxes = []
            gt_labels = []
            data_info['ann'] = dict()
            for bbox_info in boxes:
                bbox_info = bbox_info.split(',')
                bbox = bbox_info[:8]
                bbox = [*map(lambda x: float(x), bbox)]
                cls_name = bbox_info[8]
                label = self.cls_map[cls_name]
                gt_labels.append(label)
                gt_bboxes.append(bbox)
            data_info['ann']['bboxes'] = np.array(
                gt_bboxes, dtype=np.float32
            )
            data_info['ann']['labels'] = np.array(
                gt_labels, dtype=np.int64
            )
            data_infos.append(data_info)
        return data_infos


if __name__ == "__main__":
    data_root = '/home/pupa/PycharmProjects/LightRotateDet/data/msratd500'
    RDetDatasets(data_root)