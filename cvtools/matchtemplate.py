import cv2
import os
import shutil
from chg_process import match_template

# def match_template(template_path, path, pic_sort=4):
#     template = cv2.imread(template_path, cv2.IMREAD_UNCHANGED)
#     matchs = find_match_pic(path)
#     for match in matchs:
#         match_img = cv2.imread(match, cv2.IMREAD_UNCHANGED)
#         res = cv2.matchTemplate(match_img, template, cv2.TM_CCOEFF_NORMED)
#         cv2.normalize(res, res, 0, 1, cv2.NORM_MINMAX, -1)
#         cv2.imwrite("/home/pupa/PycharmProjects/cvalgorithms/result_match.png", res)


def find_match_pic(path, pic_sort=4):
    image_paths = os.listdir(path)
    all_fiter_image_path = []
    for image_path in image_paths:
        if image_path.endswith(f"{pic_sort}.png"):
            all_image_path = os.path.join(path, image_path)
            all_fiter_image_path.append(all_image_path)
    return all_fiter_image_path


def data_pre_div(image_path, template_path, *args, **kwargs):
    if not os.path.exists(os.path.join(image_path, 'flex_00')):
        os.makedirs(os.path.join(image_path, 'flex_00'))
        os.makedirs(os.path.join(image_path, 'flex_01'))
    files = os.listdir(image_path)
    for file in files:
        if file.endswith('c5.png'):
            os.remove(os.path.join(image_path, file))
        elif 'flex_01' in file and file.endswith('.png') or 'flex_01' in file and file.endswith('c0.json'):
            old_file_path = os.path.join(image_path, file)
            cur_file_path = os.path.join(os.path.join(image_path, 'flex_01'), file)
            shutil.move(old_file_path, cur_file_path)
        elif 'flex_00' in file and file.endswith('.png') or 'flex_00' in file and file.endswith('c0.json'):
            old_file_path = os.path.join(image_path, file)
            cur_file_path = os.path.join(os.path.join(image_path, 'flex_00'), file)
            shutil.move(old_file_path, cur_file_path)
        elif file.endswith('.json'):
            os.remove(os.path.join(image_path, file))
    match_template(os.path.join(image_path, 'flex_01'), os.path.join(template_path, 'flex_01'))
    match_template(os.path.join(image_path, 'flex_00'), os.path.join(template_path, 'flex_00'))


if __name__ == "__main__":
    # template = "/home/pupa/Datasets/fpc_dos/dos_82.bmp"
    # path = "/media/pupa/Samsung_T5/pcs_images/11850/Bside/20210928/15"
    # match_template(template, path)
    template_path = "C:/Users/user1/Desktop/career/templates"
    data_pre_div("C:/Users/user1/Desktop/career/buqiangpianyi", template_path)