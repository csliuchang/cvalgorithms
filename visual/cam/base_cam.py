import torch
from utils.checkpoint import load_checkpoint
import torch.nn.functional as F


class ModelLayer:
    RRetinaNet = "cls_convs"


class BaseCam:
    def __init__(self, model):
        # assert model.__class__.__name__ in ModelLayer.__dict__, 'not support this model'
        # layer_name = getattr(ModelLayer, model.__class__.__name__)
        # self.target_layer = getattr(model, layer_name)
        layer_name = getattr(ModelLayer, model.module.__class__.__name__)
        self.target_layer = getattr(model.module.bbox_head, layer_name)[-1]
        self.model = model.cuda()
        self.activations = []
        self.target_layer.register_forward_hook(self.save_activation)
        # get activate

    def __call__(self, img, class_idx=None, retain_graph=False, *args, **kwargs):
        img = img.cuda()
        logits = self.model(img)
        logits = logits[:, :, 9]
        if class_idx is None:
            predicted_class = logits.max(1)[-1]
            score = logits[:, logits.max(1)[-1]].squeeze()
        else:
            predicted_class = torch.LongTensor([class_idx])
            score = logits[:, class_idx].squeeze()
        if torch.cuda.is_available():
            score = score.cuda()
            predicted_class = predicted_class.cuda()
        # score.backward(retain_graph=retain_graph)
        activations = self.activations[0]
        score_saliency_map = self.forward_cam(activations, img, predicted_class)
        return score_saliency_map

    def forward_cam(self, activations, img, predicted_class):
        raise NotImplementedError

    def save_activation(self, module, input, output):
        activation = output
        self.activations.append(activation.cpu().detach())