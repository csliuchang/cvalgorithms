import cv2
import os
import json
from .segdataset import BaseDataset
import os.path as osp
import numpy as np
from .builder import DATASETS

CLASS = ["DIF"]
COLOR = [(255, 0, 20)]


@DATASETS.register_module()
class LEVIRDataset(BaseDataset):
    def __init__(self, img_g_path, img_n_path, *args, **kwargs):
        self.img_g_path = img_g_path
        self.img_n_path = img_n_path
        self.category = CLASS
        self.num_classes = len(self.category)
        self.color = COLOR
        self.cls_map = {c: i for i, c in enumerate(self.category)}
        super(LEVIRDataset, self).__init__(*args, **kwargs)

    def load_image(self, index):
        img_str = self.data_infos[index]['filename']
        img_g_str = osp.join(self.ann_file, self.img_g_path, img_str)
        img_n_str = osp.join(self.ann_file, self.img_n_path, img_str)
        img_g = cv2.imread(img_g_str, cv2.IMREAD_UNCHANGED)
        img_n = cv2.imread(img_n_str, cv2.IMREAD_UNCHANGED)
        img_info = np.concatenate([img_g, img_n], axis=-1)
        ori_image_shape = img_info.shape[:2]
        return img_info, img_str, ori_image_shape

    def load_annotations(self, ann_file):
        data_infos = []
        img_g_paths = osp.join(ann_file, self.img_g_path)
        label_paths = osp.join(ann_file, 'label')
        for label_path, img_g_path in zip(os.listdir(label_paths), os.listdir(img_g_paths)):
            data_info = dict()
            all_label_path = os.path.join(label_paths, label_path)
            mask = cv2.imread(all_label_path, cv2.IMREAD_UNCHANGED)
            data_info['filename'] = img_g_path
            data_info['ann'] = dict()
            mask[mask == 255] = 1
            data_info['ann']['masks'] = np.array(
                mask, dtype=np.int64
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
            img_all_path_g = os.path.join(self.data_root, "train", "A", img_pre_path)
            img_all_path_n = os.path.join(self.data_root, "train", "B", img_pre_path)
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
