import cv2
import unicodedata
from .base_dataset import BaseDataset
import os
import json
from .builder import DATASETS
import numpy as np
from models.utils.box_transform import cxcywh_xyxy

CLASS = ["DOS"]
COLOR = [(255, 0, 20)]


template_path = ""


@DATASETS.register_module()
class FPCDatasets(BaseDataset):
    def __init__(self, template_mode=None, *args, **kwargs):
        self.category = CLASS
        self.num_classes = len(self.category)
        self.color = COLOR
        self.template_mode = template_mode
        self.cls_map = {c: i for i, c in enumerate(self.category)}
        super(FPCDatasets, self).__init__(*args, **kwargs)

    def load_image(self, index):
        """
        :param index:
        :return: pair images
        """
        img_pre_fn_path = self.data_infos[index]['filename'].strip('\ufeff')
        if self.template_mode == 'one':
            img_pre_gn_path = template_path
        else:
            img_pre_gn_path = img_pre_fn_path.replace("images", "template")
        img_pre_fn_path = os.path.join(self.data_root, img_pre_fn_path)
        img_pre_gn_path = os.path.join(self.data_root, img_pre_gn_path)
        img_info_fn = cv2.imread(img_pre_fn_path, cv2.IMREAD_UNCHANGED)
        img_info_gn = cv2.imread(img_pre_gn_path, cv2.IMREAD_UNCHANGED)
        img_info = np.concatenate([img_info_fn, img_info_gn], axis=-1)
        filename = self.data_infos[index]['filename'].split('images/')[-1]
        ori_image_shape = img_info.shape[:2]
        return img_info, filename, ori_image_shape

    def load_annotations(self, ann_file):
        data_infos = []
        lines = [line.strip() for line in open(ann_file, 'r').readlines()]
        for line in lines:
            gt_bboxes, gt_labels, gt_poly = [], [], []
            data_info = dict()
            line_parts = line.split('\t')
            img_rel_path = line_parts[0]
            data_info['filename'] = img_rel_path
            label = line_parts[1]
            label_file = os.path.join(self.data_root, label)
            json_data = json.load(open(label_file))
            data_info['ann'] = dict()
            # show image
            # img_show = cv2.imread(os.path.join(self.data_root, img_rel_path), cv2.IMREAD_UNCHANGED)
            for shape in json_data["shapes"]:
                cls_name = shape["label"]
                points = shape["points"]
                if cls_name in self.cls_map:
                    bbox = self.points2bbox_max(points)
                    bbox = np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]], dtype=np.float32)
                    # cv2.rectangle(img_show, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 20), 2)
                    # cv2.imwrite('/home/pupa/PycharmProjects/cvalgorithms/results.png', img_show)
                    label = self.cls_map[cls_name]
                    gt_labels.append(label)
                    gt_bboxes.append(bbox)
                else:
                    pass
            data_info['ann']["bboxes"] = np.array(
                gt_bboxes, dtype=np.float32
            )
            data_info['ann']['labels'] = np.array(
                gt_labels, dtype=np.int64
            )
            data_infos.append(data_info)
        return data_infos

    def points2bbox_max(self, points):
        rect = cv2.boundingRect(np.array(points, np.int32))
        bbox = np.array([rect[0], rect[1], rect[2], rect[3]], dtype=np.float32)
        return bbox


if __name__ == "__main__":
    data_root = '/home/pupa/PycharmProjects/LightRotateDet/data/msratd500'
    FPCDatasets(data_root)