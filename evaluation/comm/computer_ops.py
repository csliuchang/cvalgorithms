import numpy as np


def _fast_hist(label_gt, label_pred, num_classes):
    """
    Collect values for Confusion Matrix
    For reference, please see: https://en.wikipedia.org/wiki/Confusion_matrix
    :param label_gt: <np.array> ground-truth
    :param label_pred: <np.array> prediction
    :return: <np.ndarray> values for confusion matrix
    """
    mask = (label_gt >= 0) & (label_gt < num_classes)
    hist = np.bincount(num_classes * label_gt[mask].astype(int)
                       + label_pred[mask], minlength=num_classes**2
                       ).reshape(num_classes, num_classes)
    return hist
