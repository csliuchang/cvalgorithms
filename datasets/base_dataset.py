from .builder import DATASETS
from torch.utils.data import Dataset
import numpy as np
import os
import bisect
import cv2
from .pipelines import Compose
from torch.utils.data.dataset import ConcatDataset as _ConcatDataset

__all__ = "BaseDataset"


class BaseDataset(Dataset):
    """
    A _base datasets for rotate detection
    """

    def __init__(self, data_root, train_pipeline, val_pipeline, auto_norm=True, mean=None, std=None, train_file=None, val_file=None, test_mode=False, stage='train'):
        self.stage = stage
        self.img_ids = None
        self.test_mode = test_mode
        self.data_root = data_root
        if self.stage == "train":
            self.ann_file = os.path.join(data_root, train_file)
        elif self.stage == 'val':
            self.ann_file = os.path.join(data_root, val_file)
        else:
            self.ann_file = os.path.join(data_root, 'txt_test.txt')
        self.data_infos = self.load_annotations(self.ann_file)
        if auto_norm:
            self.mean, self.std = self._compute_mean_std(self.data_infos)
        else:
            self.mean, self.std = mean, std
        self.load_train_pipeline = Compose(train_pipeline)
        self.load_val_pipeline = Compose(val_pipeline)
        if not self.test_mode:
            self._set_group_flag()

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, index):
        if self.stage == 'val':
            return self.prepare_val_img(index)
        while True:
            data = self.prepare_train_img(index)
            if data is None:
                index = self._rand_another(index)
                continue
            return data

    def get_ann_info(self, index):
        return self.data_infos[index]['ann']

    def load_image(self, index):
        img_pre_path = self.data_infos[index]['filename']
        img_all_path = os.path.join(self.data_root, img_pre_path)
        img_info, filename = cv2.imread(img_all_path, cv2.IMREAD_UNCHANGED), img_pre_path.split('/')[-1]
        if len(img_info.shape) == 2:
            img_info = img_info[..., None].repeat(3, -1)
        ori_image_shape = img_info.shape[:2]
        return img_info, filename, ori_image_shape

    def load_annotations(self, ann_file):
        raise NotImplementedError

    def _set_group_flag(self):
        self.flag = np.zeros(len(self), dtype=np.uint8)

    def prepare_train_img(self, index):
        img_info, filename, ori_image_shape = self.load_image(index)
        ann_info = self.get_ann_info(index)
        results = dict(filename=filename, img_info=img_info, ann_info=ann_info, ori_image_shape=ori_image_shape,
                       mean=self.mean, std=self.std)
        return self.load_train_pipeline(results)

    def prepare_val_img(self, index):
        img_info, filename, ori_image_shape = self.load_image(index)
        ann_info = self.get_ann_info(index)
        results = dict(filename=filename, img_info=img_info, ann_info=ann_info, ori_image_shape=ori_image_shape,
                       mean=self.mean, std=self.std)
        return self.load_val_pipeline(results)

    def prepare_test_img(self, index):
        img_info = self.load_image(index)
        ann_info = self.get_ann_info(index)
        results = dict(img_info=img_info, ann_info=ann_info)
        return results

    def _rand_another(self, index):
        pool = np.where(self.flag == self.flag[index])[0]
        return np.random.choice(pool)

    def _compute_mean_std(self, data):
        int_mean, int_std = None, None
        for data_pre in data:
            img_pre_path = data_pre['filename']
            img_all_path = os.path.join(self.data_root, img_pre_path)
            img_info = cv2.imread(img_all_path, cv2.IMREAD_UNCHANGED)
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






