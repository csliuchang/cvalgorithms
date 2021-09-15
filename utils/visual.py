import cv2
import numpy as np
import torch


class CamMethods:
    GradCAM = "gradcam"
    ScoreCAM = "scorecam"
    GradCAMPlusPlus = "gradcam++"
    AblationCAM = "ablationcam"


def get_cam_on_image(img: object, pred: object, score_id: object = 1) -> object:
    """

    :type pred: object
    :rtype: object
    """
    pred = torch.sigmoid(pred)
    pred = pred[score_id]
    pred = pred.cpu().detach().numpy() * 255
    if len(img.shape) == 2:
        img = img[..., None].repeat(3, -1)
    heat_map = cv2.applyColorMap(np.uint8(pred), cv2.COLORMAP_JET)
    cam = cv2.addWeighted(img, 0.5, heat_map, 0.5, 0)
    return cam

