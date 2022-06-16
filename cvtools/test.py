import os
import numpy as np

import cv2


def check_image_info(assign: list):
    clip_stacked = []
    for pair in assign:
        image, template = pair
        image_info, template_info = cv2.imread(image, cv2.IMREAD_UNCHANGED), \
                                    cv2.imread(template, cv2.IMREAD_UNCHANGED)
        if image_info.shape == template_info.shape:
            clip_shape = image_info.shape
        else:
            if len(image_info.shape) == 2:
                clip_shape = [min(image_info.shape[0], template_info.shape[0]),
                              min(image_info.shape[1], template_info.shape[1])]
            else:
                clip_shape = None
        clip_stacked.append(clip_shape)
    return clip_stacked


def assign_multi(image_str, template_str):
    # build template pool
    template_pool = [temp for temp in os.listdir(template_str) if '4.png' in temp]
    assign_list = []
    pool_size = len(template_pool)
    for single_str in os.listdir(image_str):
        # if single_str.endswith('.json') :
        #     continue
        if '4.png' in single_str:
            idx = np.random.randint(pool_size)
            assign_list.append([os.path.join(image_str, single_str),
                                os.path.join(template_str, template_pool[idx])])
    return assign_list


if __name__ == "__main__":
    output_dir = "C:/Users/user1/Desktop/fpc-change/kapton-11960/assign"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    path = "C:/Users/user1/Desktop/fpc-change/kapton-11960"
    template_path = "C:/Users/user1/Desktop/fpc-change/kapton-ok"
    assign_list = assign_multi(path, template_path)
    clip_stacked = check_image_info(assign_list)

    # find c5
    for idx, assign in enumerate(assign_list):
        direction1, template1 = assign
        direction1_root, template1_root = direction1.strip('4.png'), template1.strip('4.png')
        direction2, template2 = direction1_root + '5.png', template1_root + '5.png'

        direction1_sub, template1_sub = direction1.split('\\')[-1], template1.split('\\')[-1]
        direction2_sub, template2_sub = direction2.split('\\')[-1], template2.split('\\')[-1]

        direction1_info = cv2.imread(direction1, cv2.IMREAD_UNCHANGED)
        template1_info = cv2.imread(template1, cv2.IMREAD_UNCHANGED)
        direction2_info = cv2.imread(direction2, cv2.IMREAD_UNCHANGED)
        template2_info = cv2.imread(template2, cv2.IMREAD_UNCHANGED)

        crop_region = clip_stacked[idx]
        crop_direction1_info = direction1_info[0:crop_region[0], 0:crop_region[1]]
        crop_template1_info = template1_info[0:crop_region[0], 0:crop_region[1]]
        crop_direction2_info = direction2_info[0:crop_region[0], 0:crop_region[1]]
        crop_template2_info = template2_info[0:crop_region[0], 0:crop_region[1]]

        cv2.imwrite(os.path.join(output_dir, direction1_sub), crop_direction1_info)
        cv2.imwrite(os.path.join(output_dir, template1_sub), crop_template1_info)
        cv2.imwrite(os.path.join(output_dir, direction2_sub), crop_direction2_info)
        cv2.imwrite(os.path.join(output_dir, template2_sub), crop_template2_info)
        pass
