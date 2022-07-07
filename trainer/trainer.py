from trainer.tools import BaseRunner, TrainerBase
from models import build_detector, build_segmentor
from datasets.builder import build_train_dataloader, build_val_dataloader
from engine.optimizer import build_optimizer
from utils import load_checkpoint, model_info, comm, path, events, logging
import copy
from datasets import build_dataset
import time
import torch.nn as nn
import os.path as osp
import trainer.hooks as hooks
import torch
import os
from utils.bar import ProgressBar
from utils.metrics.rotate_metrics import combine_predicts_gt
from typing import Optional
import torch.distributed as dist
from collections import OrderedDict
from evaluation import LEVIRCDEvaluation, DatasetEvaluators, DatasetEvaluator, inference_on_dataset

ModelBuilder = {"segmentation": build_segmentor,
                "rotate_detection": build_detector,
                "detection": build_detector}


class TrainerContainer(BaseRunner):
    def __init__(self, cfg):
        """
        A trainer contain three tasks:
        """
        super().__init__()
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        self.work_dir = osp.join(cfg.checkpoint_dir, cfg.dataset.type, cfg.model.type, timestamp)
        cfg.dataset.update({"stage": "train"})
        datasets = [build_dataset(cfg.dataset)]
        cfg.dataset.pop("stage")
        if len(cfg.workflow) == 2:
            cfg.dataset.update({"stage": "val"})
            val_dataset = copy.deepcopy(cfg.dataset)
            datasets.append(build_dataset(val_dataset))
            self.train_dataset, self.val_dataset = datasets
        else:
            self.train_dataset = datasets, self.val_dataset = None

        self.cfg = cfg

        # self.logger = get_root_logger(log_file=self.work_dir, log_level='INFO')
        self.network_type = cfg.network_type
        cfg.model.backbone.in_channels = cfg.input_channel
        cfg.model.backbone.input_size = (cfg.input_width, cfg.input_height)

        # build dataloader
        self.train_dataloader = self.build_train_loader(cfg)

        # build model
        self.model = ModelBuilder[self.network_type](cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
        self.mode_info_printer()
        self.resume_or_load()
        self.model = self.model.cuda()
        if cfg.distributed:
            self.model = self._build_ddp_model(cfg, self.model)

        # build_optimizer
        self.optimizer = build_optimizer(cfg, self.model)
        self.scheduler = self._initialize('lr_scheduler', torch.optim.lr_scheduler, self.optimizer)
        self.start_iter = 0
        self.max_iter = cfg.total_epochs
        self.log_iter = cfg.log_iter
        self._trainer = TrainerBase(self.model, self.train_dataloader, self.optimizer)
        self.save_val_pred = cfg.save_val_pred

        self.register_hooks(self.build_hooks())

    def build_hooks(self):
        """
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.

        Returns:
            list[HookBase]:
        """
        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(),
        ]

        def test_and_save_results():
            self._last_eval_results = self._eval(self.cfg, self.model)
            return self._last_eval_results

        ret.append(hooks.EvalHook(self.cfg, test_and_save_results))

        if comm.is_main_process():
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=2))
        return ret

    def build_writers(self):
        return default_writers(self.cfg.checkpoint_dir, self.max_iter)

    def resume_or_load(self):
        """
        If `resume==True` and `cfg.OUTPUT_DIR` contains the last checkpoint (defined by
        a `last_checkpoint` file), resume from the file. Resuming means loading all
        available states (eg. optimizer and scheduler) and update iteration counter
        from the checkpoint. ``cfg.MODEL.WEIGHTS`` will not be used.

        Otherwise, this is considered as an independent training. The method will load model
        weights from the file `cfg.MODEL.WEIGHTS` (but will not load other states) and start
        from iteration 0.

        Args:
            resume (bool): whether to do resume or not
        """
        if os.path.exists(self.cfg.pretrained):
            load_checkpoint(self.model, self.cfg.pretrained, map_location='cpu', strict=True)
            logger = logging.get_logger(__name__)
            logger.info('pretrained checkpoint is loaded.')

    def mode_info_printer(self):
        # model_str = model_info(self.model)
        # self.logger.info(model_str)
        logger = logging.get_logger(__name__)
        logger.info(self.model)

    def run_step(self):
        self._trainer.iter = self.iter
        self._trainer.run_step()

    def train(self):
        super().train(self.start_iter, self.max_iter)

    def build_train_loader(self, cfg):
        train_dataloader = build_train_dataloader(self.train_dataset,
                                                  cfg.dataloader.samples_per_gpu,
                                                  cfg.dataloader.workers_per_gpu,
                                                  seed=cfg.seed,
                                                  )
        return train_dataloader

    def build_val_loader(self, cfg):
        val_dataloader = build_val_dataloader(self.val_dataset,
                                              1,
                                              cfg.dataloader.workers_per_gpu,
                                              )
        return val_dataloader

    def _eval(self, cfg, model, evaluators=None):

        logger = logging.get_logger(__name__)

        if isinstance(evaluators, DatasetEvaluator):
            evaluators = [evaluators]

        results = OrderedDict()

        try:
            evaluator = self.build_evaluator(cfg)
        except NotImplementedError:
            logger.warn(
                "No evaluator found. Use `DefaultTrainer.test(evaluators=)`, "
                "or implement its `build_evaluator` method."
            )
            results[cfg.network_type] = {}

        results_i = inference_on_dataset(model, self.build_val_loader(cfg), evaluator)
        results[cfg.network_type] = results_i

        if len(results) == 1:
            results = list(results.values())[0]
        return results

    def build_evaluator(self, cfg):
        output_dir, evaluator_type = cfg.checkpoint_dir, cfg.network_type
        evaluator_list = []
        if evaluator_type == "segmentation":
            evaluator_list.append(LEVIRCDEvaluation())
        return DatasetEvaluators(evaluator_list)

    def _initialize(self, name, module, *args, **kwargs):
        module_name = self.cfg[name]['type']
        module_args = self.cfg[name]['args']
        assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
        module_args.update(kwargs)
        return getattr(module, module_name)(*args, **module_args)

    @staticmethod
    def _build_ddp_model(cfg, model):
        model = nn.parallel.DistributedDataParallel(model,
                                                    device_ids=[cfg.local_rank, ],
                                                    output_device=cfg.local_rank,
                                                    find_unused_parameters=True
                                                    )
        return model


def default_writers(output_dir: str, max_iter: Optional[int]):
    path.mkdir_or_exist(output_dir)
    return [
        events.CommonMetricPrinter(max_iter),
        events.JSONWriter(os.path.join(output_dir, "metrics.json")),
    ]
