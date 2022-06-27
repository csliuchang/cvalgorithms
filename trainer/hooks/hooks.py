import copy
import logging
import datetime
import numpy as np
import torch
import time
import datasets
from trainer.tools.runner import HookBase
from utils.metrics import SegEval, DetEval
from utils.events import EventWriter
import os.path as osp
from utils import save_checkpoint
import cv2
import os
from utils.common import Timer

from utils import mkdir_or_exist


class IterationTimer(HookBase):
    """
    Track the time spent for each iteration (each run_step call in the trainer).
    Print a summary in the end of training.

    This hook uses the time between the call to its :meth:`before_step`
    and :meth:`after_step` methods.
    Under the convention that :meth:`before_step` of all hooks should only
    take negligible amount of time, the :class:`IterationTimer` hook should be
    placed at the beginning of the list of hooks to obtain accurate timing.
    """

    def __init__(self, warmup_iter=3):
        """
        Args:
            warmup_iter (int): the number of iterations at the beginning to exclude
                from timing.
        """
        self._warmup_iter = warmup_iter
        self._step_timer = Timer()
        self._start_time = time.perf_counter()
        self._total_timer = Timer()

    def before_train(self):
        self._start_time = time.perf_counter()
        self._total_timer.reset()
        self._total_timer.pause()

    def after_train(self):
        logger = logging.getLogger(__name__)
        total_time = time.perf_counter() - self._start_time
        total_time_minus_hooks = self._total_timer.seconds()
        hook_time = total_time - total_time_minus_hooks

        num_iter = self.trainer.storage.iter + 1 - self.trainer.start_iter - self._warmup_iter

        if num_iter > 0 and total_time_minus_hooks > 0:
            # Speed is meaningful only after warmup
            # NOTE this format is parsed by grep in some scripts
            logger.info(
                "Overall training speed: {} iterations in {} ({:.4f} s / it)".format(
                    num_iter,
                    str(datetime.timedelta(seconds=int(total_time_minus_hooks))),
                    total_time_minus_hooks / num_iter,
                )
            )

        logger.info(
            "Total training time: {} ({} on hooks)".format(
                str(datetime.timedelta(seconds=int(total_time))),
                str(datetime.timedelta(seconds=int(hook_time))),
            )
        )

    def before_step(self):
        self._step_timer.reset()
        self._total_timer.resume()

    def after_step(self):
        # +1 because we're in after_step, the current step is done
        # but not yet counted
        iter_done = self.trainer.storage.iter - self.trainer.start_iter + 1
        if iter_done >= self._warmup_iter:
            sec = self._step_timer.seconds()
            self.trainer.storage.put_scalars(time=sec)
        else:
            self._start_time = time.perf_counter()
            self._total_timer.reset()

        self._total_timer.pause()


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
        datasets_type = cfg.dataset.type
        dataset_type = getattr(datasets, datasets_type.lower().strip(''))
        self.color = dataset_type.COLOR
        self.text = dataset_type.CLASS
        self.network_type = cfg.network_type
        if self.network_type == "segmentation":
            self.color.insert(0, (0, 0, 0))

    def before_step(self):
        pass

    def draw_text_rangle(self, img, label, bbox, network_type='detection'):
        pred_color = self.color[int(label)]
        show_text = self.text[int(label)]
        if network_type == 'detection':
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            cv2.rectangle(img, (x1, y1), (x2, y2), pred_color, 2)
            cv2.putText(img, show_text, (x1 - 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, pred_color, thickness=1,
                        lineType=cv2.LINE_AA)
        elif network_type == "rotate_detection":
            pts = np.array(bbox, np.float32)
            pts = np.array([pts.reshape((4, 2))], dtype=np.int32)
            cv2.drawContours(img, pts, 0, color=pred_color, thickness=2)
            cv2.putText(img, show_text, (int(bbox[0] - 5), int(bbox[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, pred_color,
                        thickness=1,
                        lineType=cv2.LINE_AA)
        else:
            raise NotImplementedError

    def after_step(self):
        self.save_dir = osp.join(self.trainer.work_dir, "predicts")
        mkdir_or_exist(self.save_dir)
        results = self.trainer.results
        if self.network_type == 'detection' or self.network_type == 'rotate_detection':
            self.det_draw(results)
        elif self.network_type == 'segmentation':
            self.seg_draw(results)

    def det_draw(self, results):
        for result in results:
            img_file_name = result['img_metas']['filename']
            cur_image_shape = result['img_metas']['image_shape']
            img_all_path = os.path.join(self.img_root_path, 'images', img_file_name)
            img = cv2.imread(img_all_path, cv2.IMREAD_UNCHANGED)
            img = cv2.resize(img, cur_image_shape)
            h, w = img.shape[0], img.shape[1]
            if len(img.shape) == 3 and img.shape[-1] == 1:
                img = img[:, :, 0][..., None].repeat(3, -1)
            elif len(img.shape) == 3 and img.shape[-1] == 3:
                img = img
            else:
                img = img[..., None].repeat(3, -1)
            img1, img2 = copy.deepcopy(img), copy.deepcopy(img)
            gt_bboxes, gt_labels = result['gt_bboxes'], result['gt_labels']
            for i in range(len(gt_labels)):
                gt_bbox, gt_label = gt_bboxes[i], gt_labels[i]
                self.draw_text_rangle(img1, gt_label, gt_bbox, self.network_type)
            predicts = result['predictions']
            # filter predicts
            for predict in predicts:
                if self.network_type == "detection":
                    bbox, label = predict[:4], int(predict[5])
                elif self.network_type == "rotate_detection":
                    bbox, label = predict[:8], int(predict[9])
                else:
                    raise NotImplementedError
                save_image_file = osp.join(self.save_dir, img_file_name)
                if label == -1:
                    pass
                else:
                    self.draw_text_rangle(img2, label, bbox, self.network_type)

            img = np.zeros((h, w * 2 + 20, 3), np.uint8)
            img[0:h, 0:w] = img1
            img[0:h, w + 20:2 * w + 20] = img2
            cv2.imwrite(save_image_file, img)

    def seg_draw(self, results):
        """
        Visualize for change detection and one class Segmentation
        :param results:
        :return:
        """
        # add background
        for result in results:
            img_file_name = result['img_metas']['filename']
            cur_image_shape = result['img_metas']['image_shape']
            # img_all_path = os.path.join(self.img_root_path, 'val', 'label', img_file_name)
            # img = cv2.imread(img_all_path, cv2.IMREAD_UNCHANGED)
            img = result['gt_masks']
            img = img.cpu().numpy()
            if len(img.shape) == 2:
                img = np.expand_dims(img, axis=-1)
            if result['predicts'].shape[0] > 1:
                probs = torch.softmax(result['predicts'], dim=0)
                predict = torch.argmax(probs, dim=0).cpu().numpy()
                img2 = np.zeros((predict.shape[0], probs.shape[1], 3), dtype=np.uint8)
                for label, color in enumerate(self.color):
                    img2[predict == label, :] = color

            else:
                # pred = (result['predicts'] > 1).float()
                pred = (torch.sigmoid(result['predicts']) > 0.5).float()
                img2 = np.array(pred.reshape(cur_image_shape[0], cur_image_shape[1], -1).cpu().detach() * 255,
                                dtype=np.float32)
            h, w = img.shape[0], img.shape[1]
            img_placeholder = np.zeros((h, w * 2 + 20, 3), np.uint8)
            img_placeholder[0:h, 0:w] = img * 255
            img_placeholder[0:h, w + 20:2 * w + 20] = img2
            save_image_file = osp.join(self.save_dir, img_file_name)
            cv2.imwrite(save_image_file, img_placeholder)

    def after_train(self):
        pass


class CheckpointContainer(HookBase):
    def __init__(self, cfg):
        self.network_type = cfg.network_type
        if self.network_type == 'detection' or self.network_type == 'rotate_detection':
            self.metrics = {'precision': 0., 'recall': 0., 'mAP': 0., 'train_loss': float('inf'), 'best_model_epoch': 0}
        elif self.network_type == 'segmentation':
            self.metrics = {'miou': 0.}

    def after_step(self):
        self.save_dir = osp.join(self.trainer.work_dir, "checkpoints")
        logger = self.trainer.logger
        net_save_path_best = osp.join(self.save_dir, 'model_best.pth')
        if self.network_type == 'detection' or self.network_type == 'rotate_detection':
            precision, recall, mAP = self.trainer.metrics
            if mAP >= self.metrics["mAP"]:
                self.metrics.update({'precision': precision, 'recall': recall, 'mAP': mAP})
                self.metrics.update({'best_model_epoch': self.trainer.epoch})
                self.metrics.update({'train_loss': self.trainer.train_loss})
                save_checkpoint(self.trainer.model, net_save_path_best)
        elif self.network_type == 'segmentation':
            miou = self.trainer.metrics
            if miou >= self.metrics["miou"]:
                self.metrics.update({'miou': miou})
                self.metrics.update({'best_model_epoch': self.trainer.epoch})
                self.metrics.update({'train_loss': self.trainer.train_loss})
                save_checkpoint(self.trainer.model, net_save_path_best)
        best_str = 'current best, '
        for k, v in self.metrics.items():
            best_str += '{}: {:.4f}, '.format(k, v)
        logger.info(best_str)
        logger.info('--' * 10 + f'finish {self.trainer.epoch} epoch training.' + '--' * 10)

    def after_train(self):
        pass


class PeriodicWriter(HookBase):
    """
    Write events to EventStorage (by calling ``writer.write()``) periodically.

    It is executed every ``period`` iterations and after the last iteration.
    Note that ``period`` does not affect how data is smoothed by each writer.
    """

    def __init__(self, writers, period=20):
        """
        Args:
            writers (list[EventWriter]): a list of EventWriter objects
            period (int):
        """
        self._writers = writers
        for w in writers:
            assert isinstance(w, EventWriter), w
        self._period = period

    def after_step(self):
        if (self.trainer.iter + 1) % self._period == 0 or (
            self.trainer.iter == self.trainer.max_iter - 1
        ):
            for writer in self._writers:
                writer.write()

    def after_train(self):
        for writer in self._writers:
            # If any new data is found (e.g. produced by other after_train),
            # write them before closing
            writer.write()
            writer.close()


