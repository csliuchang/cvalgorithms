import copy
import os
import json
import numpy as np
import cv2
import os.path as osp

CLASS = ["FLC", "DOS", "SCT"]


def _load_image_json(input_path):
    files = os.listdir(input_path)
    image_files = sorted([file for file in files if file.endswith('bmp')])
    json_files = sorted([file for file in files if file.endswith('json')])
    return image_files, json_files


def match_template(sub_img, template):
    w, h = template.shape[:2]
    sub_img = np.array(sub_img, dtype=np.uint8)
    res = cv2.matchTemplate(sub_img, template, cv2.TM_SQDIFF)
    cv2.normalize(res, res, 0, 1, cv2.NORM_MINMAX, -1)
    # min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    # top_left = max_loc
    # bottom_right = (top_left[0] + w, top_left[1] + h)
    # cv2.rectangle(sub_img, top_left, bottom_right, 255, 2)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res, None)
    location = min_loc
    pt1 = (int(location[0]), int(location[1]))
    x = location[0] + template.shape[1] / 2
    y = location[1] + template.shape[0] / 2
    pt2 = (int(x), int(y))
    cv2.rectangle(template, pt1, pt2, (255, 0, 0), 2)
    cv2.imwrite("temp_img.png", template)

    # cv2.imwrite("img.png", sub_img)
    # cv2.imwrite("res.png", res*255)
    # cv2.imwrite("template.png", template)


def polyline2masks(image_shape, labels, points, bg_id=255, bg_first=False):
    """
    default background id is 0
    """
    if bg_first:
        bg_id = 0
    mask = np.ones(shape=image_shape, dtype=np.uint8) * bg_id
    for label_id, polyline in zip(labels, points):
        # color = int(label_id + 1)
        color = int(label_id + 1) if bg_first else int(label_id)
        cv2.fillPoly(mask, np.array([polyline], np.int32), color=color, lineType=cv2.LINE_4)

    return mask


class ImageCutting:
    def __init__(self, rot_width, roi_height):
        super(ImageCutting, self).__init__()
        self.roi_width = rot_width
        self.roi_height = roi_height
        self.cls_map = {c: i for i, c in enumerate(CLASS)}

    def split_roi(self, roi_list, mask, is_mask=True):
        cur_sub_mask_list = []
        for roi in roi_list:
            if is_mask:
                padding_mask = np.zeros((self.roi_width, self.roi_height))
            else:
                padding_mask = np.zeros((self.roi_width, self.roi_height, 3))
            cx, cy, w, h = roi[0], roi[1], roi[2], roi[3]
            cur_sub_mask = mask[max(int(cy - (h / 2)), 0): int(cy + (h / 2)), max(int(cx - (w / 2)), 0): int(cx + (w / 2))]
            padding_mask[0:cur_sub_mask.shape[0], 0:cur_sub_mask.shape[1]] = cur_sub_mask
            cur_sub_mask_list.append(padding_mask)
        return cur_sub_mask_list

    def _parsing_json(self, json_path):
        json_data = json.load(open(json_path, 'r'))
        labels, polys = [], []
        img_file = json_data["imagePath"].strip('.bmp')
        for shape in json_data["shapes"]:
            points = shape["points"]
            label = shape["label"]
            polys.append(points)
            labels.append(label)
        return labels, polys, img_file

    def encode_labels(self, labels):
        cur_labels = []
        for label in labels:
            label_id = self.cls_map[label]
            cur_labels.append(label_id)
        return cur_labels

    def decode_labels(self, sub_mask):
        label_ids = np.unique(sub_mask)
        label_ids = sorted(label_ids)[:-1]
        shapes = []
        for label_id in label_ids:
            cur_sub_mask = copy.deepcopy(sub_mask)
            cur_sub_mask[sub_mask == label_id] = 1
            cur_sub_mask[sub_mask != label_id] = 0
            cur_sub_mask = np.array(cur_sub_mask, dtype=np.uint8)
            contours, hierarchy = cv2.findContours(cur_sub_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # return sub_labels, sub_points
            for contour in contours:
                contour = contour.reshape(-1, 2)
                cur_contour = [[int(con[0]), int(con[1])]for con in contour]
                pcs_dict = {'label': CLASS[int(label_id)], 'points': cur_contour, 'shape_type': 'polygon'}
                shapes.append(pcs_dict)
        return shapes

    def _points2bbox(self, points):
        rect = cv2.minAreaRect(np.array(points, np.int32))
        bbox = np.array([rect[0][0], rect[0][1], rect[1][0], rect[1][1]], dtype=np.float32)
        return bbox

    def _create_roi(self, bbox_list):
        cur_bbox_list = []
        for bbox in bbox_list:
            cx, cy = bbox[0], bbox[1]
            cur_bbox = [cx, cy, self.roi_width, self.roi_height]
            cur_bbox_list.append(cur_bbox)
        return cur_bbox_list

    def __call__(self, input_path, save_path, template_dir, use_match_template=False, *args, **kwargs):
        image_files, json_files = _load_image_json(input_path)
        assert len(image_files) % len(json_files) == 0, "image file must be factor as json file"
        for i in range(len(json_files)):
            img, json_pcs = image_files[i], json_files[i]
            img = cv2.imread(os.path.join(input_path, img), cv2.IMREAD_UNCHANGED)
            # img = img[:, :, 0]
            img_shape = img.shape[:2]
            json_pcs_path = os.path.join(input_path, json_pcs)
            labels, polys, img_file = self._parsing_json(json_pcs_path)
            if len(polys) == 0:
                continue
            # encode labels
            labels = self.encode_labels(labels)
            bbox_list = [self._points2bbox(poly) for poly in polys]
            # for bbox in bbox_list:
            #     cv2.rectangle(img, (int(int(bbox[0]) - int(bbox[2])/2), int(int(bbox[1]) - int(bbox[3])/2)), (int(int(int(bbox[0]) + int(bbox[2])/2)), int(int(bbox[1]) + int(bbox[3])/2)), (255, 0, 0), 2)
            #     cv2.imwrite('check_img.png', img)
            roi_list = self._create_roi(bbox_list)
            if not use_match_template:
                temp = cv2.imread(template_dir, cv2.IMREAD_UNCHANGED)
                sub_temp_pres = self.split_roi(roi_list, temp, is_mask=False)
            mask = polyline2masks(img_shape, labels, polys)
            sub_masks = self.split_roi(roi_list, mask)
            sub_imgs = self.split_roi(roi_list, img, is_mask=False)
            assert len(sub_masks) == len(sub_imgs), "mask must be equal to img"
            # decode labels
            if len(sub_masks) > 0:
                for i in range(len(sub_masks)):
                    shapes = self.decode_labels(sub_masks[i])
                    json_dict = {'imageData': None, 'imagePath': img_file + '_' + str(i) + '.png', 'shapes': shapes}
                    json_current_path = osp.join(save_path, 'labels', img_file + '_' + str(i) + '.json')
                    json.dump(json_dict, open(json_current_path, "w"), sort_keys=True, indent=2)
                    sub_img = sub_imgs[i]
                    if use_match_template:
                        template = cv2.imread(template_dir, cv2.IMREAD_UNCHANGED)
                        # template = template[:, :, 0]
                        sub_temp = match_template(sub_img, template)
                        sub_temp_all_path = osp.join(save_path, 'images', img_file + '_' + str(i) + '.png')
                        cv2.imwrite(sub_temp_all_path, sub_temp)
                    else:
                        sub_temp_all_path = osp.join(save_path, 'template', img_file + '_' + str(i) + '.png')
                        sub_temp_pre = sub_temp_pres[i]
                        cv2.imwrite(sub_temp_all_path, sub_temp_pre)
                    sub_img_all_path = osp.join(save_path, 'images', img_file + '_' + str(i) + '.png')
                    cv2.imwrite(sub_img_all_path, sub_img)
            else:
                pass
            # save img and save json


if __name__ == "__main__":
    roi_width, roi_height = 512, 512
    image_cut = ImageCutting(roi_width, roi_height)
    save_dir = "/home/pupa/Datasets/fpc"
    input_dir = "/home/pupa/Datasets/fpc_dos/pos"
    template_dir = '/home/pupa/Datasets/fpc_dos/temp/dos_82.bmp'
    # temp = cv2.imread(template_dir, 1)
    # temp_rer = cv2.flip(temp, 0)
    # temp_rer = cv2.flip(temp_rer, 1)
    # cv2.imwrite('dos_82_rer.bmp', temp_rer)
    image_cut(input_dir, save_dir, template_dir)
