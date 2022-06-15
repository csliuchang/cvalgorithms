import cv2
import json
from .base_dataset import BaseDataset
import os
import numpy as np
from .builder import DATASETS
from cvtools.chg_process import trans_wins_format

CLASS = (["bad"])
COLOR = [(204, 78, 210)]


@DATASETS.register_module()
class SegDataset(BaseDataset):
    def __init__(self, *args, **kwargs):
        self.category = CLASS
        self.num_classes = len(self.category)
        self.color = COLOR
        self.cls_map = {c: i for i, c in enumerate(self.category)}
        super(SegDataset, self).__init__(*args, **kwargs)

    def load_annotations(self, ann_file):
        data_infos = []
        ann_file = trans_wins_format(ann_file)
        lines = [line.strip() for line in open(ann_file, 'r').readlines()]
        for line in lines:
            data_info = dict()
            line_parts = line.split('\t')
            img_rel_path = line_parts[0]
            data_info['filename'] = img_rel_path
            label_part_path = line_parts[1]
            label_file = os.path.join(self.data_root, label_part_path)
            # load json
            with open(label_file, 'r') as fp:
                json_data = json.load(fp)
            polylines = []
            gt_labels = []
            for shape_data in json_data['shapes']:
                label_name = shape_data['label']
                label = self.cls_map[label_name]
                if shape_data['shape_type'] == "polygon":
                    pts = shape_data['points']
                else:
                    pts = None
                polylines.append(pts)
                gt_labels.append(label)
            labels = np.array(
                gt_labels, dtype=np.int64
            )
            data_info['annotations'] = \
                {'segmentation': polylines, 'labels': labels}
            data_infos.append(data_info)
        return data_infos
