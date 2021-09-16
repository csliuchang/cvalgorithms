import logging
import weakref
from typing import List, Optional
import time
import torch
import numpy as np


class HookBase:
    """
    Base class for hooks that can be registered with :class:`TrainerBase`.

    Each hook can implement 4 methods. The way they are called is demonstrated
    in the following snippet:
    ::
        hook.before_train()
        for iter in range(start_iter, max_iter):
            hook.before_step()
            trainer.run_step()
            hook.after_step()
        iter += 1
        hook.after_train()

    Notes:
        1. In the hook method, users can access ``self.trainer`` to access more
           properties about the context (e.g., model, current iteration, or config
           if using :class:`DefaultTrainer`).

        2. A hook that does something in :meth:`before_step` can often be
           implemented equivalently in :meth:`after_step`.
           If the hook takes non-trivial time, it is strongly recommended to
           implement the hook in :meth:`after_step` instead of :meth:`before_step`.
           The convention is that :meth:`before_step` should only take negligible time.

           Following this convention will allow hooks that do care about the difference
           between :meth:`before_step` and :meth:`after_step` (e.g., timer) to
           function properly.

    """

    trainer: "BaseRunner" = None
    """
    A weak reference to the trainer object. Set by the trainer when the hook is registered.
    """

    def before_train(self):
        """
        Called before the first iteration.
        """
        pass

    def after_train(self):
        """
        Called after the last iteration.
        """
        pass

    def before_step(self):
        """
        Called before each iteration.
        """
        pass

    def after_step(self):
        """
        Called after each iteration.
        """
        pass

    def state_dict(self):
        """
        Hooks are stateless by default, but can be made checkpointable by
        implementing `state_dict` and `load_state_dict`.
        """
        return {}


class BaseRunner:
    """
    Base Runner for train detection, segmentation, rotate object detection

    Attributes:
    iter(int): the current iteration.

    start_iter(int): The iteration to start with.
        By convention the minimum possible value is 0.


    storage(EventStorage): An EventStorage that's opened during the course of training.
    """
    def __init__(self) -> None:
        self._hooks: List[HookBase] = []
        self.epoch: int = 0
        self.start_epoch: int = 1
        self.global_step: int = 0
        self.max_epoch: int

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

    def state_dict(self):
        ret = {"iteration": self.iter}
        hooks_state = {}
        for h in self._hooks:
            sd = h.state_dict()
            if sd:
                name = type(h).__qualname__
                if name in hooks_state:
                    # TODO handle repetitive stateful hooks
                    continue
                hooks_state[name] = sd
        if hooks_state:
            ret["hooks"] = hooks_state
        return ret

    def load_state_dict(self, state_dict):
        logger = logging.getLogger(__name__)
        self.iter = state_dict["iteration"]
        for key, value in state_dict.get("hooks", {}).items():
            for h in self._hooks:
                try:
                    name = type(h).__qualname__
                except AttributeError:
                    continue
                if name == key:
                    h.load_state_dict(value)
                    break
            else:
                logger.warning(f"Cannot find the hook '{key}', its state_dict is ignored.")


class TrainerBase(BaseRunner):
    """
    A simple trainer for the most common type of task:
    """
    def __init__(self, model, dataloader, optimizer, logger):
        super().__init__()
        self.logger = logger
        self.model = model
        self._data_loader_iter = dataloader
        self.optimizer = optimizer
        self.max_epoch: int
        self.log_iter: int

    def run_step(self):
        self.model.train()
        all_losses = []
        logger_batch = 0
        start = time.time()
        for count, data in enumerate(self._data_loader_iter):
            if count >= len(self._data_loader_iter):
                break
            self.global_step += 1
            _img, _ground_truth = data['images_collect']['img'], data['ground_truth']
            _img = _img.cuda()
            batch = _img.shape[0]
            logger_batch += batch
            for key, value in _ground_truth.items():
                if value is not None:
                    if isinstance(value, torch.Tensor):
                        _ground_truth[key] = value.cuda()
            # train model

            loss_dict = self.model(_img, ground_truth=_ground_truth, return_metrics=True)
            losses = loss_dict["loss"]
            self.optimizer.zero_grad()
            losses.backward()
            losses = losses.detach().cpu().numpy()
            all_losses.append(losses)
            self.optimizer.step()
            if self.global_step % self.log_iter == 0:
                batch_time = time.time() - start
                self._write_metrics(batch, count, all_losses, batch_time, logger_batch)

    def _write_metrics(self, batch, count, all_losses, batch_time, logger_batch):
        self.logger.info(
            'epochs=>[%d/%d], pers=>[%d/%d], training step: %d, running loss: %f, time/pers: %d ms' % (
                self.epoch, self.max_epoch, (count + 1) * batch, len(self._data_loader_iter.dataset), self.global_step,
                np.array(all_losses).mean(), (batch_time * 1000) / logger_batch))

    def state_dict(self):
        ret = super().state_dict()
        ret["optimizer"] = self.optimizer.state_dict()
        return ret

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.optimizer.load_state_dict(state_dict["optimizer"])














