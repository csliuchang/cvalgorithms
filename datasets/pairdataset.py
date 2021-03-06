import cv2
import unicodedata
from .base_dataset import BaseDataset
import os
import numpy as np
from .builder import DATASETS


CLASS = ('open', 'short', 'mousebite', 'spur', 'copper', 'pin-hole')
COLOR = [(255, 180, 0), (0, 255, 255), (0, 0, 255), (255, 0, 20), (0, 206, 209), (255, 78, 210)]


@DATASETS.register_module()
class PairDatasets(BaseDataset):
    def __init__(self, *args, **kwargs):
        self.category = CLASS
        self.num_classes = len(self.category)
        self.color = COLOR
        self.cls_map = {c: i for i, c in enumerate(self.category)}
        super(PairDatasets, self).__init__(*args, **kwargs)

    def load_image(self, index):
        """
        :param index:
        :return: pair images
        """
        img_pre_path = self.data_infos[index]['filename'].split('.')[0].strip('\ufeff')
        img_pre_path_gn = img_pre_path + '_temp.jpg'
        img_pre_path_fn = img_pre_path + '_test.jpg'
        img_all_path_gn = os.path.join(self.data_root, img_pre_path_gn)
        img_info_gn = cv2.imread(img_all_path_gn, cv2.IMREAD_UNCHANGED)
        img_all_path_fn = os.path.join(self.data_root, img_pre_path_fn)
        img_info_fn = cv2.imread(img_all_path_fn, cv2.IMREAD_UNCHANGED)
        if len(img_info_fn.shape) == 3:
            img_info_fn = img_info_fn[:, :, 0][..., None].repeat(3, -1)
        if len(img_info_gn.shape) == 3:
            img_info_gn = img_info_gn[:, :, 0][..., None].repeat(3, -1)
        if len(img_info_gn.shape) == 2:
            img_info_gn = img_info_gn[..., None].repeat(3, -1)
        if len(img_info_fn.shape) == 2:
            img_info_fn = img_info_fn[..., None].repeat(3, -1)
        # add gn noise
        # img_cat = np.zeros((img_info_gn.shape[0], img_info_gn.shape[1] * 2, 3))
        # img_cat[:img_info_gn.shape[0], :img_info_gn.shape[1], :] = img_info_gn
        # img_info_gn = cv2.GaussianBlur(img_info_gn, (19, 19), 0)
        # img_cat[:img_info_gn.shape[0], img_info_gn.shape[1]:, :] = img_info_gn
        # cv2.imwrite('/home/pupa/PycharmProjects/cvalgorithms/result.png', img_cat)
        img_info = np.concatenate([img_info_fn, img_info_gn], axis=-1)
        filename = img_pre_path_fn.split('/')[-1]
        ori_image_shape = img_info.shape[:2]
        return img_info, filename, ori_image_shape

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
                bbox_info = bbox_info.split(' ')
                bbox = bbox_info[:4]
                bbox = [*map(lambda x: float(x), bbox)]
                cls_name = bbox_info[4]
                if cls_name.isdigit():
                    label = int(cls_name) - 1
                else:
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
    PairDatasets(data_root)