from .evaluator import DatasetEvaluator
from collections import OrderedDict
import torch
from utils import logger, comm
import itertools
import copy


class LEVIRCDEvaluation(DatasetEvaluator):
    def __init__(self, distributed=False):
        self._logger = logger.get_logger(__name__)
        self._cpu_device = torch.device("cpu")
        self._distributed = distributed

    def evaluate(self, ):
        """
        evaluate F1 and iou for change predict.
            F1:
            iou:
        """
        if self._distributed:
            comm.synchronize()
            predictions = comm.gather(self._predictions, dst=0)
            predictions = list(itertools.chain(*predictions))
        else:
            predictions = self._predictions

        if len(predictions) == 0:
            self._logger.warning("[COCOEvaluator] Did not receive valid predictions.")
            return {}

        self._results = OrderedDict()
        if "proposals" in predictions[0]:
            self._eval_predictions(predictions)

        return copy.deepcopy(self._results)

    def reset(self):
        self._predictions = []

    def process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            prediction = {"image_id": input["filename"],
                          "proposals": output.to(self._cpu_device),
                          "ground_truth": input["annotations"]}

        if len(prediction) > 1:
            self._predictions.append(prediction)

    def _eval_predictions(self, predictions):
        self._logger.info("Preparing results for LEVIR-CD format ...")
        pass
