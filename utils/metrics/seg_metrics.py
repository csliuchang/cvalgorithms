import torch
import numpy as np
from scipy import sparse


EPS = 1e-8


class SegMatrix:
    """computer matrix"""
    def __init__(self, num_classes, mask, pred):
        self.num_classes = num_classes
        self._matrix = sparse.coo_matrix((num_classes, num_classes), dtype=np.float32)
        self._computer_matrix(mask, pred)

    @property
    def sparse_cm(self):
        return self._matrix.toarray()

    def _computer_matrix(self, mask, pred):
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()

        pred, mask = pred.reshape((-1,)), mask.reshape((-1,))
        v = np.ones_like(pred)
        cm = sparse.coo_matrix((v, (mask, pred)), shape=(self.num_classes, self.num_classes), dtype=np.float32)
        self._matrix += cm
        # return cm


class SegEval:
    def __init__(self, num_classes, thousand=0.3, ignore_label=255):
        self.ignore_label = ignore_label
        self.num_classes = num_classes
        self.thousand = thousand

    def __call__(self, collections):
        """
        scipy sparse maybe have mistake
        """
    #     num_classes = self.num_classes + 1 if self.num_classes == 1 else self.num_classes
    #     iou_list = []
    #     for collection in collections:
    #         logits = collection['predicts']
    #         label = collection['gt_masks']
    #         # changes = torch.sigmoid(logits) > self.thousand
    #         changes = logits > 1
    #         pr_change = changes.cpu().numpy().astype(np.uint8)
    #         # label = (label < 1).float()
    #         # load matrix
    #         matrix_cr = SegMatrix(num_classes, pr_change, label)
    #         iou_per_class = SegEval.compute_iou_per_class(matrix_cr.sparse_cm)
    #         iou_list.append(iou_per_class)
    #     return np.mean(iou_list)
    #
    # @staticmethod
    # def compute_iou_per_class(matrix):
    #     sum_over_row = np.sum(matrix, axis=0)
    #     sum_over_col = np.sum(matrix, axis=1)
    #     diag = np.diag(matrix)
    #     # iou = inter / uni
    #     iou_per_class = diag / (sum_over_row + sum_over_col - diag + EPS)
    #     return iou_per_class[1:].mean()

        if collections[0]["predicts"].shape[0] == 1:
            iou_sum = []
            for collection in collections:
                # print(collection['predicts'][collection['predicts']!=0])
                # predicts = collection['predicts'].sigmoid() > self.thousand
                # pr_change = predicts.cpu().numpy().astype(np.uint8)
                predicts = torch.sigmoid(collection['predicts'])
                pr_change = (predicts > 0.5).float().cpu().numpy()
                masks = collection['gt_masks']
                # masks[masks == 0] = 1
                # masks[masks == 255] = 0
                masks = masks.cpu().numpy()
                iou = self.compute_mean_iou(pr_change, masks)
                iou_sum.append(iou)
            return np.mean(iou_sum)
        else:
            n_classes = self.num_classes + 1
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

    def compute_mean_iou(self, pred, label, unique_labels=(1,)):
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
