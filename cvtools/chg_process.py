import copy
import os
import numpy as np
import json
import cv2
import os.path as osp
import shutil


__all__ = ["ChgFlexCrop"]


def trans_wins_format(path):
    assert type(path) == str, "must input str"
    return path.replace("\\", '/')


def pixel_padding(image, size=640, pad=0):
    h, w = image.shape[0], image.shape[1]
    if pad == 0:
        image_pad = np.zeros(shape=(228, 398), dtype=np.uint8)
    else:
        image_pad = np.ones(shape=(228, 398, 3), dtype=np.uint8) * 255
    if h < size or w < size:
        if pad == 0:
            image_pad[0:h, 0:w] = image
        else:
            image_pad[0:h, 0:w, :] = image
        return image_pad
    else:
        return image


def crop_img_with_roi(roi_list, img_info, temp_info, mask_info, image_path, image):
    for i, roi in enumerate(roi_list):
        ori_img = img_info[roi[1]:roi[3], roi[0]: roi[2], :]
        ori_mask = mask_info[roi[1]:roi[3], roi[0]: roi[2]]
        ori_template = temp_info[roi[1]:roi[3], roi[0]: roi[2]]
        ori_img = pixel_padding(ori_img, pad=255)
        ori_mask = pixel_padding(ori_mask, pad=0)
        ori_template = pixel_padding(ori_template, pad=255)
        root_path = image_path.split(image_path.split("/")[-1])[0]
        crop_image_path = root_path + 'crop_image'
        crop_mask_path = root_path + 'crop_mask'
        crop_template_path = root_path + 'crop_template'
        crop_labelme_path = root_path + 'crop_labelme'
        mask_folder(crop_image_path)
        mask_folder(crop_mask_path)
        mask_folder(crop_template_path)
        mask_folder(crop_labelme_path)
        image_root_str = image.split('.png')[0][:-2] + f'_crop{i}_' + image.split('.png')[0][-2:]
        cv2.imwrite(
            os.path.join(crop_image_path,
                         image_root_str + '.png'),
            ori_img)
        cv2.imwrite(
            os.path.join(crop_mask_path,
                         image_root_str + '.png'),
            ori_mask)
        cv2.imwrite(os.path.join(crop_template_path,
                                 image.split('.png')[0][:-2] + f'_crop{i}_' + 'template_' +
                                 image.split('.png')[0][-2:] +
                                 '.png'), ori_template)
        json_name = trans_wins_format(os.path.join(crop_labelme_path,
                                                   image_root_str + '.json'))
        ChgFlexCrop.mask2labelme(ori_mask, image_root_str, json_name)


def mask_folder(fold):
    if not os.path.exists(fold):
        os.mkdir(fold)
    else:
        pass


# CLASS = ["flex_fm", "flex_dos"]
# color_map = {"flex_fm": 1, "flex_dos": 2}

CLASS = ["change"]
color_map = {"change": 255}


class ChgFlexCrop:
    """
    A Norm Data Crop
    """

    def __init__(self,
                 random_scale=320,
                 num_rois=3,
                 channel_num=5,
                 crop_with_obj=False,
                 mask_bg=False,
                 crop_random=False,
                 crop_set=False,
                 resize=False,
                 slide_crop=True
                 ):
        super(ChgFlexCrop, self).__init__()
        self.mask_bg = mask_bg
        self.crop_with_obj = crop_with_obj
        self.random_scale = random_scale
        self.num_rois = num_rois
        self.channel_num = channel_num
        self.crop_random = crop_random
        self.crop_set = crop_set
        self.resize = resize
        self.slide_crop = slide_crop

    @staticmethod
    def label2mask(labels_path, image_shapes, pre_color=1):
        "trans label to mask"
        for i, mask in enumerate(labels_path):
            if pre_color != 0:
                mask_pad = np.zeros(shape=image_shapes[i], dtype=np.uint8)
            else:
                mask_pad = np.ones(shape=image_shapes[i], dtype=np.uint8) * 255
            mask_all_0 = os.path.join(labels_path, mask)
            mask_all_0_info = json.load(open(mask_all_0, 'r'))
            for shape in mask_all_0_info['shapes']:
                polyline = shape['points']
                if shape['label'] == 'flex':
                    cv2.fillPoly(mask_pad, np.array([polyline], np.int32), color=pre_color, lineType=cv2.LINE_4)
            cv2.imwrite(f'mask_{i}.png', mask_pad)

    @staticmethod
    def mask2labelme(mask_info, image_name, json_name, label_mask="changes"):
        # print(np.unique(mask_info))
        shape_list = []
        for i in range(0, len(CLASS)):
            mask_data = copy.deepcopy(mask_info)
            mask_data[mask_info != i + 1] = 0
            # print(i)

            # print(label_mask, np.unique(mask_info))
            if len(np.unique(mask_data)) == 1 and np.unique(mask_data) == 0:
                continue
            label_mask = CLASS[i]
            poly = cv2.findContours(mask_data, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for p in poly[0]:
                p_list = [[int(l[0][0]), int(l[0][1])] for l in p]
                shape = dict(label=label_mask, points=p_list, shape_type="polygon")
                shape_list.append(shape)
        json_data = dict(imageData=None, image_path=image_name, shapes=shape_list)
        json.dump(json_data, open(json_name, "w"), indent=2)

    def _roi_random(self, roi_list, polyline=None, roi_random=None, image_shape=None, roi_set_bbox=[267, 290, 665, 518]):
        if self.crop_with_obj:
            x_list = [point[0] for point in polyline]
            y_list = [point[1] for point in polyline]
            min_x, max_x = min(x_list), max(x_list)
            min_y, max_y = min(y_list), max(y_list)
            # get the random roi
            random_minx = max(min_x - self.random_scale, 0)
            random_miny = max(min_y - self.random_scale, 0)
            roi_list = []
            for i in range(self.num_rois):
                # add if cause miny sometimes equal neg value or zero
                roi_x = 0 if random_minx >= min_x else np.random.randint(random_minx, min_x)
                roi_y = 0 if random_miny >= min_y else np.random.randint(random_miny, min_y)
                roi_bbox = [roi_x, roi_y, roi_x + self.random_scale * 2, roi_y +
                            self.random_scale * 2]
                roi_list.append(roi_bbox)
        if self.crop_random:
            img_w, img_h = image_shape[0], image_shape[1]
            random_x = np.random.randint(0, img_w - self.random_scale * 2)
            random_y = np.random.randint(0, img_h - self.random_scale * 4)
            random_roi_bbox = [random_x, random_y,
                               random_x + self.random_scale * 2, random_y + self.random_scale * 2]
            roi_list.append(random_roi_bbox)
        if self.crop_set:
            roi_list.append(roi_set_bbox)
        if self.slide_crop:
            roi_width, roi_height, roi_padding = 512, 512, 16
            h, w = image_shape
            h_split_base, w_split_base = (h - (roi_height - roi_padding)) // roi_height, (w - (roi_width - roi_padding)) // roi_width
            roi_h_splits = h_split_base + 1 if (h - (roi_height - roi_padding)) % roi_height != 0 else h_split_base
            roi_w_splits = w_split_base + 1 if (w - (roi_width - roi_padding)) % roi_width != 0 else w_split_base
        return roi_list

    def _save_img(self, img_info, temp_info, mask_info, image_path, image):
        root_path = image_path.split(image_path.split("/")[-1])[0]
        crop_image_path = root_path + '_image'
        crop_mask_path = root_path + '_mask'
        crop_template_path = root_path + '_template'
        crop_labelme_path = root_path + '_labelme'
        mask_folder(crop_image_path)
        mask_folder(crop_mask_path)
        mask_folder(crop_template_path)
        mask_folder(crop_labelme_path)
        image_root_str = image.split('.png')[0][:-2] + '_' + image.split('.png')[0][-2:]
        cv2.imwrite(
            os.path.join(crop_image_path,
                         image_root_str + '.png'),
            img_info)
        cv2.imwrite(
            os.path.join(crop_mask_path,
                         image_root_str + '.png'),
            mask_info)
        cv2.imwrite(os.path.join(crop_template_path,
                                 image.split('.png')[0][:-2] + '_' + 'template_' +
                                 image.split('.png')[0][-2:] +
                                 '.png'), temp_info)
        json_name = trans_wins_format(os.path.join(crop_labelme_path,
                                                   image_root_str + '.json'))
        ChgFlexCrop.mask2labelme(mask_info, image_root_str, json_name)

    def __call__(self, image, label, template, roi_random=None, mask_roi=False, key_name=None, *args, **kwargs):
        images = sorted(os.listdir(image))
        templates = sorted(os.listdir(template))
        len_t = len(templates) / self.channel_num
        root_str_keep = [None, ]
        roi_list_full = []
        for i, img in enumerate(images):
            # random get template
            n = np.random.randint(0, len_t - 1)
            temp = templates[i % 5 + n * self.channel_num]
            # get images
            # TODO name
            root_str = img.split(img[-7:])[0]
            label_str_full = trans_wins_format(os.path.join(label, root_str + '_c0.json'))
            img_str_full = osp.join(image, img)
            temp_full = trans_wins_format(os.path.join(template, temp))
            # get info from full path
            if not os.path.exists(label_str_full):
                continue
            json_info = json.load(open(label_str_full, "r"))
            img_info = cv2.imread(img_str_full, cv2.IMREAD_UNCHANGED)
            temp_info = cv2.imread(temp_full, cv2.IMREAD_UNCHANGED)
            # mask
            img_w, img_h = img_info.shape[0], img_info.shape[1]
            mask_info = np.zeros(shape=(img_w, img_h), dtype=np.uint8)
            roi_list = []
            for shape in json_info['shapes']:
                if shape['label'] in key_name:
                    continue
                polyline = shape['points']
                try:
                    color = color_map[shape['label']]
                except Exception as e:
                    print(f'no label {e} in this color dict')
                    color = 1
                cv2.fillPoly(mask_info, np.array([polyline], np.int32), color=color, lineType=cv2.LINE_4)
                # compute roi when root name different
                if root_str != root_str_keep[-1]:
                    roi_list = self._roi_random(roi_list, polyline, roi_random, image_shape=img_info.shape)
                    roi_list_full.append(roi_list)
                else:
                    roi_list = roi_list_full[-1]
                # crop roi
            if self.resize:
                img_info = cv2.resize(img_info, dsize=(1024, 256))
                temp_info = cv2.resize(temp_info, dsize=(1024, 256))
                mask_info = cv2.resize(mask_info, dsize=(1024, 256))
            if roi_list:
                crop_img_with_roi(roi_list, img_info, temp_info, mask_info, image, img)
            else:
                self._save_img(img_info, temp_info, mask_info, image, img)
            print(f'===============program has process {i} images====================')
            if root_str != root_str_keep[-1]:
                root_str_keep.append(root_str)


def match_template(image_path, template_path, channel_num=5):
    images = sorted(os.listdir(image_path))
    templates = sorted(os.listdir(template_path))
    len_t = len(templates) / channel_num
    images = [img_temp for img_temp in images if img_temp.endswith('.png')]
    for i, img in enumerate(images):
            n = np.random.randint(0, len_t - 1)
            img_full_str = osp.join(image_path, img)
            img_info = cv2.imread(img_full_str, cv2.IMREAD_UNCHANGED)
            print(img_full_str)
            temp = templates[i % 5 + n * channel_num]
            temp_full_str = trans_wins_format(osp.join(template_path, temp))
            temp_info = cv2.imread(temp_full_str, cv2.IMREAD_UNCHANGED)
            temp_info = cv2.resize(temp_info, [img_info.shape[1], img_info.shape[0]])
            cur_temp = img.split('.png')[0][:-2] + 'template_' + img.split('.png')[0][-2:] + '.png'
            cur_temp_str = trans_wins_format(osp.join(image_path, cur_temp))
            cv2.imwrite(cur_temp_str, temp_info)


def match_template_channel(image_path, template_path, channel_num=5):
    """
    assign image and template without json label
    image_path: image folder path
    channel_num:  project pic channels
    """
    images = sorted(os.listdir(image_path))
    templates = sorted(os.listdir(template_path))
    len_img = int(len(images) / channel_num)
    len_t = len(templates) / channel_num
    images = [img_temp for img_temp in images if img_temp.endswith('.png')]
    for i in range(len_img):
        n = np.random.randint(0, len_t - 1)
        for j in range(channel_num):
            image_patch = images[channel_num * i + j]
            img_full_str = osp.join(image_path, image_patch)
            img_info = cv2.imread(img_full_str, cv2.IMREAD_UNCHANGED)
            print(f"images have completed {i} picss ", img_full_str)
            temp = templates[j % 5 + n * channel_num]
            temp_full_str = trans_wins_format(osp.join(template_path, temp))
            temp_info = cv2.imread(temp_full_str, cv2.IMREAD_UNCHANGED)
            temp_info = cv2.resize(temp_info, [img_info.shape[1], img_info.shape[0]])
            cur_temp = image_patch.split('.png')[0][:-2] + 'template_' + image_patch.split('.png')[0][-2:] + '.png'
            cur_temp_str = trans_wins_format(osp.join(image_path, cur_temp))
            cv2.imwrite(cur_temp_str, temp_info)


def crop_with_roi(img, roi=None):
    """
    crop numpy data with roi
    """
    roi_img = img[roi[1]:roi[3], roi[0]: roi[2], :]
    return roi_img


def labelme2mask(json_info, img_shape, color_map={"flex_fm": 1, "flex_dos": 2}, key_name=["flex"]):
    mask_info = np.zeros(shape=(img_shape[0], img_shape[1]), dtype=np.uint8)
    for shape in json_info['shapes']:
        if shape['label'] in key_name:
            continue
        polyline = shape['points']
        color = color_map[shape['label']]
        cv2.fillPoly(mask_info, np.array([polyline], np.int32), color=color, lineType=cv2.LINE_4)
        return mask_info


def crop_img_template(img, template, save_path: str, roi=[267, 290, 665, 518], ):
    images = sorted(os.listdir(img))
    templates = sorted(os.listdir(template))
    for i, image in enumerate(images):
        img_info = cv2.imread(os.path.join(img, image), cv2.IMREAD_UNCHANGED)
        template_info = cv2.imread(os.path.join(template, templates[i]), cv2.IMREAD_UNCHANGED)
        img_roi_info = crop_with_roi(img_info, roi)
        template_roi_info = crop_with_roi(template_info, roi)
        image_root_str = image.split(image[-7:])[0]
        cv2.imwrite(
            os.path.join(save_path,
                     image),
            img_roi_info)
        cv2.imwrite(os.path.join(save_path,
                                 image.split('.png')[0][:-2] + 'template_' +
                                 image.split('.png')[0][-2:] +
                                 '.png'), template_roi_info)


class Resize():
    def __init__(self):
        self.rate = 5


def print_shape(path):
    images = os.listdir(path)
    for image in images:
        if image.endswith('png'):
            img_full = os.path.join(path, image)
            img_info = cv2.imread(img_full, cv2.IMREAD_UNCHANGED)
            print(img_info.shape)


if __name__ == "__main__":
    image = input("input your image path:")
    template = input("input your template path:")
    # template = "C:/Users/user1/Desktop/FLEX_CD/RIGHT-CD/templates"
    # crop = ChgFlexCrop()
    # crop(image, label, template, key_name=["c_pian"])
    # crop_img_label(image, label, roi=[267, 290, 665, 518])
    # save_path = "C:/Users/user1/Desktop/career/cpian_test"
    # crop_img_template(image, template, roi=[267, 290, 665, 518], save_path=save_path)
    # print_shape(image)
    match_template_channel(image, template)
