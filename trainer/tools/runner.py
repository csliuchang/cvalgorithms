import logging
import weakref
from typing import List, Optional, Mapping
import time
import torch
import numpy as np
from utils import comm
from utils.events import EventStorage, get_event_storage


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
        self.iter: int = 0
        self.start_iter: int = 1
        self.global_step: int = 0
        self.storage: EventStorage
        self.max_iter: int

    def register_hooks(self, hooks: List[Optional[HookBase]]) -> None:
        hooks = [h for h in hooks if h is not None]
        for h in hooks:
            assert isinstance(h, HookBase)
            # To avoid circular reference, create Proxy objects
            h.trainer = weakref.proxy(self)
        self._hooks.extend(hooks)

    def train(self, start_iter: int, max_iter: int):
        """
        train logic from detectron2
        """
        logger = logging.getLogger(__name__)
        logger.info("Starting training from iteration {}".format(start_iter))
        self.iter = self.start_iter = start_iter
        self.max_iter = max_iter
        with EventStorage(start_iter) as self.storage:
            try:
                self.before_train()
                for self.epoch in range(start_iter, max_iter):
                    self.before_step()
                    self.run_step()
                    self.after_step()
                self.iter += 1
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

    def __init__(self, model, dataloader, optimizer, logger, scheduler):
        super().__init__()
        self.logger = logger
        self.model = model
        self._data_loader_iter = iter(dataloader)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.max_epoch: int
        self.log_iter: int

    def run_step(self):
        """
        Implement the standard training logic described above
        """
        if not self.model.training:
            self.model.train()

        start = time.perf_counter()
        data = next(self._data_loader_iter)
        _img, _ground_truth = data['images_collect']['img'].cuda(), data['ground_truth']
        for key, value in _ground_truth.items():
            if value is not None:
                if isinstance(value, torch.Tensor):
                    _ground_truth[key] = value.cuda()
        data_time = time.perf_counter() - start
        loss_dict = self.model(_img, ground_truth=_ground_truth, return_metrics=True)
        if isinstance(loss_dict, torch.Tensor):
            losses = loss_dict
            loss_dict = {"total_loss": loss_dict}
        else:
            losses = \
                    loss_dict['loss']

        self.optimizer.zero_grad()
        losses.backward()
        self._write_metrics(loss_dict, data_time)
        self.optimizer.step()
        print("process ")


    # def run_step(self):
    #     self.model.train()
    #     all_losses = []
    #     logger_batch = 0
    #     start = time.time()
    #     for count, data in enumerate(self._data_loader_iter):
    #         if count >= len(self._data_loader_iter):
    #             break
    #         self.global_step += 1
    #         _img, _ground_truth = data['images_collect']['img'],
    #         # img show
    #         # img = _img.permute(0, 2, 3, 1)
    #         # img = img.detach().cpu().numpy().reshape(448, 448, -1)
    #         # img_1, img_2 = img[:, :, :3]*255, img[:, :, 3:]*255
    #         # ground_truth = _ground_truth['gt_masks'].permute(1, 2, 0).cpu().numpy()
    #         # ground_truth = ground_truth.reshape(448, 448, -1)
    #         # import cv2
    #         # cv2.imwrite('result_n.png', img_1)
    #         # cv2.imwrite('result_g.png', img_2)
    #         # cv2.imwrite('gt.png', ground_truth)
    #         _img = _img.cuda()
    #         batch = _img.shape[0]
    #         logger_batch += batch
    #         for key, value in _ground_truth.items():
    #             if value is not None:
    #                 if isinstance(value, torch.Tensor):
    #                     _ground_truth[key] = value.cuda()
    #         # train model
    #         self.optimizer.zero_grad()
    #         loss_dict = self.model(_img, ground_truth=_ground_truth, return_metrics=True)
    #         losses = loss_dict["loss"]
    #         losses.backward()
    #         losses = losses.detach().cpu().numpy()
    #         all_losses.append(losses)
    #         self.optimizer.step()
    #         self.scheduler.step()
    #         if self.global_step % self.log_iter == 0:
    #             batch_time = time.time() - start
    #             self._write_metrics(batch, count, all_losses, batch_time, logger_batch)
    #     self.train_loss = sum(all_losses) / len(self._data_loader_iter.dataset)

    def _write_metrics(
            self,
            loss_dict: Mapping[str, torch.Tensor],
            data_time: float,
            prefix: str = "",
    ) -> None:
        TrainerBase.write_metrics(loss_dict, data_time, prefix)

        # def _write_metrics(self, batch, count, all_losses, batch_time, logger_batch):

    #     self.logger.info(
    #         'epochs=>[%d/%d], pers=>[%d/%d], training step: %d, running loss: %f, time/pers: %d ms' % (
    #             self.epoch, self.max_epoch, (count + 1) * batch, len(self._data_loader_iter.dataset), self.global_step,
    #             np.array(all_losses).mean(), (batch_time * 1000) / logger_batch))

    @staticmethod
    def write_metrics(
        loss_dict: Mapping[str, torch.Tensor],
        data_time: float,
        prefix: str = "",
    ) -> None:
        """
        Args:
            loss_dict (dict): dict of scalar losses
            data_time (float): time taken by the dataloader iteration
            prefix (str): prefix for logging keys
        """
        metrics_dict = {k: v.detach().cpu().item() for k, v in loss_dict.items()}
        metrics_dict["data_time"] = data_time

        # Gather metrics among all workers for logging
        # This assumes we do DDP-style training, which is currently the only
        # supported method in detectron2.
        all_metrics_dict = comm.gather(metrics_dict)

        if comm.is_main_process():
            storage = get_event_storage()

            # data_time among workers can have high variance. The actual latency
            # caused by data_time is the maximum among workers.
            data_time = np.max([x.pop("data_time") for x in all_metrics_dict])
            storage.put_scalar("data_time", data_time)

            # average the rest metrics
            metrics_dict = {
                k: np.mean([x[k] for x in all_metrics_dict]) for k in all_metrics_dict[0].keys()
            }
            total_losses_reduced = sum(metrics_dict.values())
            if not np.isfinite(total_losses_reduced):
                raise FloatingPointError(
                    f"Loss became infinite or NaN at iteration={storage.iter}!\n"
                    f"loss_dict = {metrics_dict}"
                )

            storage.put_scalar("{}total_loss".format(prefix), total_losses_reduced)
            if len(metrics_dict) > 1:
                storage.put_scalars(**metrics_dict)

    def state_dict(self):
        ret = super().state_dict()
        ret["optimizer"] = self.optimizer.state_dict()
        return ret

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.optimizer.load_state_dict(state_dict["optimizer"])
