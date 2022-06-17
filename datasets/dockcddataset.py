import platform

import cv2
import os
import json
from .segdataset import BaseDataset
import os.path as osp
import numpy as np
from .builder import DATASETS
from cvtools.chg_process import trans_wins_format

CLASS = ["vabnorm"]
COLOR = [(255, 255, 255)]


@DATASETS.register_module()
class DOCKCDDataset(BaseDataset):
    def __init__(self, img_g_path, img_n_path, *args, **kwargs):
        self.img_g_path = img_g_path
        self.img_n_path = img_n_path
        self.category = CLASS
        self.num_classes = len(self.category)
        self.color = COLOR
        self.cls_map = {c: i for i, c in enumerate(self.category)}
        super(DOCKCDDataset, self).__init__(*args, **kwargs)

    def load_image(self, index):
        img_str = self.data_infos[index]['filename'].split('images/')[-1]
        img_g_str = osp.join(self.data_root, self.img_g_path, img_str)
        img_n_str = osp.join(self.data_root, self.img_n_path, img_str.split('.png')[0] + '_template' + '.png')
        img_g = cv2.imread(img_g_str, cv2.IMREAD_UNCHANGED)
        img_n = cv2.imread(img_n_str, cv2.IMREAD_UNCHANGED)
        img_info = np.concatenate([img_g, img_n], axis=-1)
        ori_image_shape = img_info.shape[:2]
        return img_info, img_str, ori_image_shape

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
            if platform.system() == "Windows":
                label_file = trans_wins_format(label_file)
            # load json
            with open(label_file, 'r') as fp:
                json_data = json.load(fp)
            polylines = []
            gt_labels = []
            for shape_data in json_data['shapes']:
                label_name = shape_data['label']
                if len(self.cls_map) == 1:
                    label = self.cls_map[label_name]
                else:
                    label = self.cls_map[label_name]
                if shape_data['shape_type'] == "polygon":
                    pts = shape_data['points']
                else:
                    pts = None
                polylines.append(pts)
                gt_labels.append(label)

            data_info['annotations'] = dict()

            data_info['annotations']['segmentation'] = polylines
            data_info['annotations']['labels'] = np.array(
                gt_labels, dtype=np.int64
            )
            data_infos.append(data_info)
        return data_infos

    def convert_labels(self, label):
        for k, v in self.lb_map.items():
            label[label == k] = v
        return label

    def _compute_mean_std(self, data):
        int_mean, int_std = None, None
        for data_pre in data:
            img_pre_path = data_pre['filename']
            img_all_path_g = os.path.join(self.data_root,  img_pre_path)
            img_all_path_n = os.path.join(self.data_root,  img_pre_path)
            img_info_g = cv2.imread(img_all_path_g, cv2.IMREAD_UNCHANGED)
            img_info_n = cv2.imread(img_all_path_n, cv2.IMREAD_UNCHANGED)
            img_info = np.concatenate([img_info_g, img_info_n], axis=-1)
            pre_mean = np.mean(img_info, axis=(0, 1), dtype=np.float32)
            pre_std = np.std(img_info, axis=(0, 1), dtype=np.float32)
            if int_mean is None:
                int_mean = pre_mean
            else:
                int_mean += pre_mean
            if int_std is None:
                int_std = pre_std
            else:
                int_std += pre_std
        mean = (int_mean / len(data)).tolist()
        std = (int_std / len(data)).tolist()
        return mean, std
