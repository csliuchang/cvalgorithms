from trainer.tools.runner import HookBase
from utils.metrics import SegEval, RotateDetEval
import os.path as osp
from utils import save_checkpoint


class EvalHook(HookBase):
    def __init__(self, cfg, eval_function):
        self._period = cfg.eval_period
        num_classes = cfg.num_classes
        network_type = cfg.network_type
        if network_type == "segmentation":
            self._eval_func = SegEval(num_classes)
        elif network_type == "rotate_detection":
            self._eval_func = RotateDetEval(num_classes)
        self._func = eval_function

    def _do_eval(self):
        results = self._func()
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

