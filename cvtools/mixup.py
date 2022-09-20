import cv2


def region_paste_image(region, image):
    pass


def roi_with_gray(image, outdir, idx):
    image_resize = cv2.resize(image, dsize=None, fx=0.2, fy=0.2)
    h, w = image_resize.shape[0], image_resize.shape[1]
    image_crop = image_resize[40:h-40, 40:w-40]
    total_path = outdir + f'/{idx}.png'
    cv2.imwrite(total_path, image_crop)
    pass


if __name__ == "__main__":
    import os
    import os.path as osp
    path = "C:/Users/user1/Desktop/warp_images/steel"
    outdir = "C:/Users/user1/Desktop/warp_images/steel_crop"
    for idx, pcs in enumerate(os.listdir(path)):
        image = cv2.imread(osp.join(path, pcs), cv2.IMREAD_GRAYSCALE)
        roi_with_gray(image, outdir, idx)
