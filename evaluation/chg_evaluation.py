from .evaluator import DatasetEvaluator
from collections import OrderedDict


class ChangeEvaluation(DatasetEvaluator):
    def __init__(self):
        pass

    def evaluate(self):
        """
        evaluate F1 and iou for change predict.
            F1:
            iou:
        """
        self._results = OrderedDict()
        pass

    def reset(self):
        self._predictions = []
