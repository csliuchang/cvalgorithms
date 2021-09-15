import torch
from utils.checkpoint import load_checkpoint
import torch.nn.functional as F
from .base_cam import BaseCam


class ScoreCam(BaseCam):
    def __init__(self, *args, **kwargs):
        super(ScoreCam, self).__init__(*args, **kwargs)

    @torch.no_grad()
    def forward_cam(self, activations, img, predicted_class):
        _, _, h, w = img.shape
        b, channels, u, v = activations.shape
        score_saliency_map = torch.zeros((1, 1, h, w))
        if torch.cuda.is_available():
            activations = activations.cuda()
            score_saliency_map = score_saliency_map.cuda()
        for i in range(channels):
            saliency_map = torch.unsqueeze(activations[:, i, :, :], 1)
            saliency_map = F.interpolate(saliency_map, size=(h, w), mode='bilinear', align_corners=False)

            if saliency_map.max() == saliency_map.min():
                continue

            norm_saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())
            output = self.model(img * norm_saliency_map)
            output = F.softmax(output)
            output = output[:, :, 8]
            score = output[0][predicted_class]
            score = score.cuda()
            score_saliency_map += score * saliency_map
        score_saliency_map = F.relu(score_saliency_map)
        score_saliency_map_min, score_saliency_map_max = score_saliency_map.min(), score_saliency_map.max()

        if score_saliency_map_min == score_saliency_map_max:
            return None
        score_saliency_map = (score_saliency_map - score_saliency_map_min).div(score_saliency_map_max - score_saliency_map_min).data
        return score_saliency_map



