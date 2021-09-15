import logging
import weakref
from typing import List, Optional
from tools.hooks.hookbase import HookBase
import time
import torch
import numpy as np


class BaseRunner:
    """
    Base Runner for train detection, segmentation, rotate object detection

    Attributes:
    iter(int): the current iteration.

    start_iter(int): The iteration to start with.
        By convention the minimum possible value is 0.


    storage(EventStorage): An EventStorage that's opened during the course of training.
    """
    def __init__(self):
        self._hooks: List[HookBase] = []
        self.epoch: int = 0
        self.start_epoch: int = 0
        self.global_step: int = 0

    def register_hooks(self, hooks: List[Optional[HookBase]]) -> None:
        hooks = [h for h in hooks if h is not None]
        for h in hooks:
            assert isinstance(h, HookBase)
            # To avoid circular reference, create Proxy objects
            h.trainer = weakref.proxy(self)
        self._hooks.extend(hooks)

    def train(self, start_epoch: int, max_epoch: int):
        """
        train logic from detectron2
        """
        logger = logging.getLogger(__name__)
        logger.info("Starting training from iteration {}".format(start_epoch))
        self.epoch = self.start_epoch = start_epoch
        self.max_epoch = max_epoch
        try:
            self.before_train()
            for self.epoch in range(start_epoch, max_epoch):
                self.before_step()
                self.run_step()
                self.after_step()
            self.epoch += 1
        except Exception:
            logger.exception("Exception during training:")
            raise
        finally:
            self.after_train()

    def before_train(self):
        for h in self._hooks:
            h.before_train()

    def after_train(self):
        for h in self._hooks:
            h.after_train()

    def before_step(self):
        for h in self._hooks:
            h.before_step()

    def after_step(self):
        for h in self._hooks:
            h.after_step()

    def run_step(self):
        raise NotImplementedError


class TrainerBase(BaseRunner):
    """
    A simple trainer for the most common type of task:
    """
    def __init__(self, model, dataloader, optimizer, logger):
        super().__init__()

        model.train()
        self.logger = logger
        self.model = model
        self._data_loader_iter = iter(dataloader)
        self.optimizer = optimizer

    def run_step(self):
        assert self.model.training, 'mode is eval, u need change model to eval mode!'
        # TODO  will change in future
        all_losses = []
        logger_batch = 0
        for count, data in self._data_loader_iter:
            _img, _ground_truth = data['images_collect']['img'], data['ground_truth']
            _img = _img.cuda()
            batch = _img.shape[0]
            logger_batch += batch
            for key, value in _ground_truth.items():
                if value is not None:
                    if isinstance(value, torch.Tensor):
                        _ground_truth[key] = value.cuda()
            # train model
            start = time.perf_counter()
            loss_dict = self.model(_img, ground_truth=_ground_truth, return_metric=True)
            losses = loss_dict["loss"]
            self.optimizer.zero_grad()
            losses.backward()
            losses = losses.detach().cpu().numpy()
            all_losses.append(losses)
            self.optimizer.step()
            batch_time = time.perf_counter() - start
            self._write_metrics(batch, count, all_losses, batch_time, logger_batch)

    def _write_metrics(self, batch, count, all_losses, batch_time, logger_batch):
        self.logger.info(
            'epochs=>[%d/%d], pers=>[%d/%d], training step: %d, running loss: %f, time/pers: %d ms' % (
                self.epoch, self.max_epoch, (count + 1) * batch, len(next(self._data_loader_iter)), self.global_step,
                np.array(all_losses).mean(), (batch_time * 1000) / logger_batch))














