import cv2
import os 


def match_template(template_path, path, pic_sort=4):
    template = cv2.imread(template_path, cv2.IMREAD_UNCHANGED)
    matchs = find_match_pic(path)
    for match in matchs:
        match_img = cv2.imread(match, cv2.IMREAD_UNCHANGED)
        res = cv2.matchTemplate(match_img, template, cv2.TM_CCOEFF_NORMED)
        cv2.normalize(res, res, 0, 1, cv2.NORM_MINMAX, -1)
        cv2.imwrite("/home/pupa/PycharmProjects/cvalgorithms/result_match.png", res)


def find_match_pic(path, pic_sort=4):
    image_paths = os.listdir(path)
    all_fiter_image_path = []
    for image_path in image_paths:
        if image_path.endswith(f"{pic_sort}.png"):
            all_image_path = os.path.join(path, image_path)
            all_fiter_image_path.append(all_image_path)
    return all_fiter_image_path


if __name__ == "__main__":
    template = "/home/pupa/Datasets/fpc_dos/dos_82.bmp"
    path = "/media/pupa/Samsung_T5/pcs_images/11850/Bside/20210928/15"
    match_template(template, path)