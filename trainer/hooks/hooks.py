from trainer.tools.runner import HookBase
from utils.metrics import SegEval, DetEval
import os.path as osp
from utils import save_checkpoint
import cv2
import os
from datasets.pairdataset import COLOR, CLASS
from utils import mkdir_or_exist


class EvalHook(HookBase):
    def __init__(self, cfg, eval_function):
        self._period = cfg.eval_period
        num_classes = cfg.num_classes
        network_type = cfg.network_type
        if network_type == "segmentation":
            self._eval_func = SegEval(num_classes)
        elif network_type == "rotate_detection":
            self._eval_func = DetEval(num_classes, rotate_eval=True)
        else:
            self._eval_func = DetEval(num_classes)
        self._func = eval_function

    def _do_eval(self):
        results = self._func()
        self.trainer.results = results
        metrics = self._eval_func(results)
        return metrics

    def after_step(self):
        next_epoch = self.trainer.epoch + 1
        if self._period > 0 and next_epoch % self._period == 0:
            # do the last eval in after_train
            if next_epoch != self.trainer.max_epoch:
                self.trainer.metrics = self._do_eval()

    def after_train(self):
        pass


class VisualPrediction(HookBase):
    def __init__(self, cfg):
        self.img_root_path = cfg.dataset.data_root
        self.color = COLOR
        self.text = CLASS

    def after_step(self):
        self.save_dir = osp.join(self.trainer.work_dir, "predicts")
        mkdir_or_exist(self.save_dir)
        results = self.trainer.results
        for result in results:
            img_file_name = result['img_metas']['filename']
            img_all_path = os.path.join(self.img_root_path, 'images', img_file_name)
            img = cv2.imread(img_all_path, cv2.IMREAD_UNCHANGED)
            img1, img2 = img, img
            predicts = result["predictions"]
            # filter predicts
            for predict in predicts:
                bbox = predict[:4]
                label = int(predict[5]) - 1
                save_image_file = osp.join(self.save_dir, img_file_name)
                if label == -1:
                    pass
                else:
                    pred_color = self.color[label]
                    show_text = self.text[label]
                    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                    cv2.rectangle(img, (x1, y1), (x2, y2), pred_color, 2)
                    cv2.putText(img, show_text, (x1 - 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, pred_color, thickness=1,
                                lineType=cv2.LINE_AA)
            cv2.imwrite(save_image_file, img)

    def after_train(self):
        pass


class CheckpointContainer(HookBase):
    def __init__(self):
        self.metrics = {'precision': 0., 'recall': 0., 'mAP': 0., 'train_loss': float('inf'), 'best_model_epoch': 0}

    def after_step(self):
        self.save_dir = osp.join(self.trainer.work_dir, "checkpoints")
        logger = self.trainer.logger
        precision, recall, mAP = self.trainer.metrics
        net_save_path_best = osp.join(self.save_dir, 'model_best.pth')
        if mAP > self.metrics["mAP"]:
            self.metrics.update({'precision': precision, 'recall': recall, 'mAP': mAP})
            save_checkpoint(self.trainer.model, net_save_path_best)
        best_str = 'current best, '
        for k, v in self.metrics.items():
            best_str += '{}: {:.4f}, '.format(k, v)
        logger.info(best_str)
        logger.info('--' * 10 + f'finish {self.trainer.epoch} epoch training.' + '--' * 10)

    def after_train(self):
        pass
