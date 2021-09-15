import cv2
from utils.transformer import apply_transforms
from cam.score_cam import ScoreCam
from PIL import Image
import torchvision.models as models
import torch
from utils.image_opts import basic_visualize
from utils.checkpoint import load_checkpoint


def load_image(image_path):
    """Loads image as a PIL RGB image.

        Args:
            - **image_path (str) - **: A path to the image

        Returns:
            An instance of PIL.Image.Image in RGB

    """

    return Image.open(image_path).convert('RGB')


def main():
    state_dict = None
    input_image = load_image('/home/pupa/PycharmProjects/Score-CAM/images/ILSVRC2012_val_00000057.JPEG')
    input_ = apply_transforms(input_image)
    model = models.resnet18(pretrained=True).eval()
    if state_dict is not None:
        model = load_checkpoint(model, state_dict)
    cam = ScoreCam(model)
    score_map = cam(input_)
    basic_visualize(input_.cpu(), score_map.type(torch.FloatTensor).cpu(), save_path='resnet.png')


if __name__ == "__main__":
    main()