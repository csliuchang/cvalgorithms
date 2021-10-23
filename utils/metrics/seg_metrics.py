import torch
import numpy as np


class SegEval(object):
    def __init__(self, num_classes=1, ignore_label=255):
        pass
        self.ignore_label = ignore_label
        self.num_classes = num_classes

    def __call__(self, collections):
        if collections[0]["predicts"].shape[0] == 1:
            iou_sum = []
            for collection in collections:
                predicts = collection['predicts']
                predicts = (predicts > 1).float().cpu().numpy()
                masks = collection['gt_masks']
                masks[masks == 255] = 1
                masks = masks.cpu().numpy()
                iou = self.compute_mean_iou(predicts, masks)
                iou_sum.append(iou)
            return np.mean(iou_sum)
        else:
            if self.num_classes == 1:
                n_classes = self.num_classes + 1
            else:
                n_classes = self.num_classes
            # build a matrix
            hist = torch.zeros(n_classes, n_classes).cuda().detach()
            for collection in collections:
                logits = collection['predicts']
                label = collection['gt_masks']
                probs = torch.softmax(logits, dim=0)
                preds = torch.argmax(probs, dim=0)
                keep = label != self.ignore_label
                hist += torch.bincount(
                    label[keep] * n_classes + preds[keep],
                    minlength=n_classes ** 2
                ).view(n_classes, n_classes).float()
            ious = hist.diag() / (hist.sum(dim=0) + hist.sum(dim=1) - hist.diag())
            miou = ious.mean()
            return miou.item()

    def compute_mean_iou(self, pred, label, unique_labels=(0, 1)):
        if unique_labels is None:
            unique_labels = np.unique(label)
            if len(unique_labels) <= 1:
                unique_labels = (0, 1)
                pass
            pass
        num_unique_labels = len(unique_labels)

        I = np.zeros(num_unique_labels)
        U = np.zeros(num_unique_labels)
        IOU = np.zeros(num_unique_labels)

        for index, val in enumerate(unique_labels):
            pred_i = pred == val
            label_i = label == val
            I[index] = float(np.sum(np.logical_and(label_i, pred_i)))
            U[index] = float(np.sum(np.logical_or(label_i, pred_i)))
            if U[index] != 0:
                IOU[index] = I[index] / U[index]
            elif I[index] == 0:
                IOU[index] = 1.0
            else:
                print("can not calc iou on I({}) and U({})".format(I[index], U[index]))
                raise RuntimeError
            pass

        mean_iou = np.mean(IOU)
        return mean_iou

