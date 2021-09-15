import cv2
import json
from .segdataset import BaseDataset
import os
import numpy as np
from .builder import DATASETS


@DATASETS.register_module()
class CityStcapes(BaseDataset):
    def __init__(self, *args, **kwargs):
        fr = open('./data/citystcapes/cityscapes_info.json', 'r')
        labels_info = json.load(fr)
        self.cls_map = {el['name']: el['trainId'] for el in labels_info}
        super(CityStcapes, self).__init__(*args, **kwargs)

    def load_annotations(self, ann_file):
        data_infos = []
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
            for shape_data in json_data['objects']:
                label_name = shape_data['label']
                if 'group' in label_name:
                    label = 255
                else:
                    label = self.cls_map[label_name]
                pts = shape_data['polygon']
                polylines.append(pts)
                gt_labels.append(label)

            data_info['ann'] = dict()

            data_info['ann']['polylines'] = polylines
            data_info['ann']['labels'] = np.array(
                gt_labels, dtype=np.int64
            )
            data_infos.append(data_info)
        return data_infos

    def convert_labels(self, label):
        for k, v in self.lb_map.items():
            label[label == k] = v
        return label
