import numpy as np
from shapely.geometry import Polygon


def combine_predicts_gt(predicts, metas, gt, network_type='segmentation'):
    if network_type == 'segmentation':
        predicts, mask = predicts.squeeze(0), gt['gt_masks'].squeeze(0)
        return dict(img_metas=metas, predicts=predicts, gt_masks=mask)
    else:

        gt_bboxes, gt_labels, gt_masks = gt["gt_bboxes"], gt["gt_labels"], gt["gt_masks"]
        return dict(img_metas=metas, predictions=predicts.squeeze(0),
                    gt_bboxes=gt_bboxes[0], gt_labels=gt_labels[0], gt_masks=gt_masks[0])


class RotateDetEval(object):
    def __init__(self, num_classes=1, min_score_threshold=0.4, min_iou_threshold=0.35):
        self.min_score_threshold = min_score_threshold
        self.min_iou_threshold = min_iou_threshold
        self.num_classes = num_classes

    def val_per_measure(self):
        """
        Evaluate val datasets with batch
        """
        pass

    def __call__(self, collections):
        tp_per_class = [[] for _ in range(self.num_classes)]
        fp_per_class = [[] for _ in range(self.num_classes)]
        gt_counter_class = [0] * self.num_classes
        for collection in collections:
            preds = np.array(collection["predictions"], np.float32)
            fiter_flags = preds[:, 9] > self.min_score_threshold
            preds = preds[fiter_flags.reshape(-1), :]
            gt_bboxes = collection["gt_bboxes"]
            gt_labels = collection["gt_labels"]
            num_of_gt_per_img = gt_bboxes.shape[0]
            for gt_idx in range(num_of_gt_per_img):
                gt_cls = gt_labels[gt_idx]
                gt_counter_class[gt_cls] += 1

            already_match = [False] * num_of_gt_per_img

            for pred in preds:
                # default lable is start 0, so need to reduce 1
                pred_bbox, score, pred_cls = pred[:8], pred[8], int(pred[9]) - 1
                pred_bbox = np.array(pred_bbox, dtype=np.float32).reshape(4, 2)
                tp, fp, max_iou = 0, 0, 0
                match_gt_idx = -1
                for gt_idx in range(num_of_gt_per_img):
                    gt_bbox = gt_bboxes[gt_idx].reshape(4, 2)
                    gt_cls = gt_labels[gt_idx]

                    if pred_cls != gt_cls:
                        continue
                    iou = self._get_intersection_over_union(pred_bbox, gt_bbox)
                    if iou <= max_iou:
                        continue
                    max_iou = iou
                    match_gt_idx = gt_idx
                    pass
                if max_iou >= self.min_iou_threshold:
                    if not already_match[match_gt_idx]:
                        # tp
                        already_match[match_gt_idx] = True
                        tp = 1
                    else:
                        # repeat match
                        fp = 1
                else:
                    # fp
                    fp = 1

                tp_per_class[pred_cls].append(tp)
                fp_per_class[pred_cls].append(fp)

                pass
            pass

        ap_sum = 0
        prec_sum = 0
        rec_sum = 0
        for cls_idx in range(self.num_classes):
            fp = fp_per_class[cls_idx]
            tp = tp_per_class[cls_idx]

            cumsum = 0
            for idx, val in enumerate(fp):
                fp[idx] += cumsum
                cumsum += val
            cumsum = 0
            for idx, val in enumerate(tp):
                tp[idx] += cumsum
                cumsum += val

            rec = tp[:]
            for idx, val in enumerate(tp):
                rec[idx] = float(tp[idx]) / (gt_counter_class[cls_idx] + 1e-6)

            prec = tp[:]
            for idx, val in enumerate(tp):
                prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx] + 1e-6)

            ap, mrec, mprec = self.voc_ap(rec, prec)

            ap_sum += ap
            prec_sum += prec[-2]
            rec_sum += rec[-2]

        mAP = ap_sum / self.num_classes
        prec = prec_sum / self.num_classes
        rec = rec_sum / self.num_classes

        return prec, rec, mAP

    @staticmethod
    def _get_intersection_over_union(pred, gt_bbox):
        """
        A simple polygon iou compute function
        """
        union = Polygon(pred).union(Polygon(gt_bbox)).area
        inter = Polygon(pred).intersection(Polygon(gt_bbox)).area
        return inter / union

    @staticmethod
    def voc_ap(rec, prec):
        """
        --- Official matlab code VOC2012---
        mrec=[0 ; rec ; 1];
        mpre=[0 ; prec ; 0];
        for i=numel(mpre)-1:-1:1
            mpre(i)=max(mpre(i),mpre(i+1));
        end
        i=find(mrec(2:end)~=mrec(1:end-1))+1;
        ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
        """
        rec.insert(0, 0.0)  # insert 0.0 at begining of list
        rec.append(1.0)  # insert 1.0 at end of list
        mrec = rec[:]
        prec.insert(0, 0.0)  # insert 0.0 at begining of list
        prec.append(0.0)  # insert 0.0 at end of list
        mpre = prec[:]
        """
         This part makes the precision monotonically decreasing
          (goes from the end to the beginning)
          matlab:  for i=numel(mpre)-1:-1:1
                      mpre(i)=max(mpre(i),mpre(i+1));
        """
        # matlab indexes start in 1 but python in 0, so I have to do:
        #   range(start=(len(mpre) - 2), end=0, step=-1)
        # also the python function range excludes the end, resulting in:
        #   range(start=(len(mpre) - 2), end=-1, step=-1)
        for i in range(len(mpre) - 2, -1, -1):
            mpre[i] = max(mpre[i], mpre[i + 1])
        """
         This part creates a list of indexes where the recall changes
          matlab:  i=find(mrec(2:end)~=mrec(1:end-1))+1;
        """
        i_list = []
        for i in range(1, len(mrec)):
            if mrec[i] != mrec[i - 1]:
                i_list.append(i)  # if it was matlab would be i + 1
        """
         The Average Precision (AP) is the area under the curve
          (numerical integration)
          matlab: ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
        """
        ap = 0.0
        for i in i_list:
            ap += ((mrec[i] - mrec[i - 1]) * mpre[i])
        return ap, mrec, mpre



