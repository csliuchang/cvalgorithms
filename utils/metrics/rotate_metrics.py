import numpy as np
try:
    from shapely.geometry import Polygon
except:
    Polygon=None
    print("shapely is not this version")


def combine_predicts_gt(predicts, metas, gt, network_type='segmentation'):
    if network_type == 'segmentation':
        predicts, mask = predicts.squeeze(0), gt['gt_masks'].squeeze(0)
        return dict(img_metas=metas, predicts=predicts, gt_masks=mask)
    else:

        gt_bboxes, gt_labels, gt_masks = gt["gt_bboxes"], gt["gt_labels"], gt["gt_masks"]
        return dict(img_metas=metas, predictions=predicts.squeeze(0),
                    gt_bboxes=gt_bboxes[0], gt_labels=gt_labels[0], gt_masks=gt_masks[0])


class DetEval(object):
    def __init__(self, num_classes=1, min_score_threshold=0.4, min_iou_threshold=0.35, rotate_eval=False):
        self.min_score_threshold = min_score_threshold
        self.min_iou_threshold = min_iou_threshold
        self.num_classes = num_classes
        self.rotate_eval = rotate_eval

    def val_per_measure(self):
        """
        Evaluate val datasets with batch
        """
        pass

    def _iou_function(self, pred_bbox, gt_bbox):
        if self.rotate_eval:
            return self._get_intersection_over_union(pred_bbox, gt_bbox)
        else:
            return self._get_hbb_over_union(pred_bbox, gt_bbox)

    def __call__(self, collections):
        tp_per_class = [[] for _ in range(self.num_classes)]
        fp_per_class = [[] for _ in range(self.num_classes)]
        gt_counter_class = [0] * self.num_classes
        for collection in collections:
            preds = np.array(collection["predictions"], np.float32)
            if self.rotate_eval:
                fiter_flags = preds[:, 8] > self.min_score_threshold
            else:
                fiter_flags = preds[:, 4] > self.min_score_threshold
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
                if self.rotate_eval:
                    pred_bbox, score, pred_cls = pred[:8], pred[8], int(pred[9])
                    pred_bbox = np.array(pred_bbox, dtype=np.float32).reshape(4, 2)
                else:
                    pred_bbox, score, pred_cls = pred[:4], pred[4], int(pred[5])
                    pred_bbox = np.array(pred_bbox, dtype=np.float32)
                tp, fp, max_iou = 0, 0, 0
                match_gt_idx = -1
                for gt_idx in range(num_of_gt_per_img):
                    if self.rotate_eval:
                        gt_bbox = gt_bboxes[gt_idx].reshape(4, 2)
                    else:
                        gt_bbox = gt_bboxes[gt_idx]
                    gt_cls = gt_labels[gt_idx]

                    if pred_cls != gt_cls:
                        continue
                    iou = self._iou_function(pred_bbox, gt_bbox)
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
    def _get_hbb_over_union(boxA, boxB):
        """
        calc iou
        :param boxA: [xmin, ymin, xmax, ymax]
        :param boxB: [xmin, ymin, xmax, ymax]
        :return: iou value
        """
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)

        # return the intersection over union value
        return iou

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




