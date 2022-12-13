from .evaluator import DatasetEvaluator
from collections import OrderedDict
import torch
from utils import logger, comm
import itertools
import copy
import numpy as np
from evaluation.comm.computer_ops import _fast_hist


class LEVIRCDEvaluation(DatasetEvaluator):
    def __init__(self, num_classes, distributed=False):
        self._logger = logger.get_logger(__name__)
        self._cpu_device = torch.device("cpu")
        self._distributed = distributed
        self.num_classes = num_classes + 1

    def evaluate(self, ):
        """
        evaluate F1 and iou for change predict.
            F1:
            iou:
        """
        res = {}

        hist = self._conf_matrix
        tp = np.diag(hist)
        sum_a1, sum_a0 = hist.sum(axis=1), hist.sum(axis=0)
        acc = tp.sum() / (hist.sum() + np.finfo(np.float32).eps)
        recall = tp / (sum_a1 + np.finfo(np.float32).eps)
        precision = tp / (sum_a0 + np.finfo(np.float32).eps)
        # iou = tp / (sum_a1 + sum_a0 - tp + np.finfo(np.float32).eps)
        iou = tp / np.clip(sum_a1 + sum_a0 - tp, a_min=np.finfo(np.float32).eps)

        mean_iou = np.nanmean(iou)
        f1 = 2 * recall * precision / (recall + precision + np.finfo(np.float32).eps)

        res["miou"] = mean_iou
        res["acc"] = acc
        res["precision"] = precision[1]
        res["recall"] = recall[1]
        res["f1_score"] = f1[1]

        results = OrderedDict({"levir": res})

        self._logger.info(results)
        return results

    def reset(self):
        self._predictions = []
        self._conf_matrix = np.zeros((
            self.num_classes, self.num_classes
        ))

    def process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            gt = input["annotations"]["masks"].to(self._cpu_device).numpy()
            pred = output.to(self._cpu_device).numpy()
            self._conf_matrix += _fast_hist(pred.flatten(), gt.flatten(), self.num_classes)

