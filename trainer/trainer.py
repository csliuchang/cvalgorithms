from trainer.tools import BaseRunner, TrainerBase
from models import build_detector, build_segmentor
from datasets.builder import build_dataloader
from engine.optimizer import build_optimizer
from utils import get_root_logger, load_checkpoint, model_info, comm, path, events
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

        self.logger = get_root_logger(log_file=self.work_dir, log_level='INFO')
        self.network_type = cfg.network_type
        cfg.model.backbone.in_channels = cfg.input_channel
        cfg.model.backbone.input_size = (cfg.input_width, cfg.input_height)

        # build dataloader
        self.train_dataloader = self.build_train_loader(cfg)
        self.val_dataloader = self.build_val_loader(cfg)

        # build model
        self.model = ModelBuilder[self.network_type](cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
        self.mode_info_printer()
        self.resume_or_load()

        self.model.cuda()
        if cfg.distributed:
            self.model = self._build_ddp_model(cfg, self.model)

        # build_optimizer
        optimizer = build_optimizer(cfg, self.model)
        self.scheduler = self._initialize('lr_scheduler', torch.optim.lr_scheduler, optimizer)
        self.start_iter = 0
        self.max_iter = cfg.total_epochs
        self.log_iter = cfg.log_iter
        self._trainer = TrainerBase(self.model, self.train_dataloader, optimizer, self.logger, self.scheduler)
        self.save_val_pred = cfg.save_val_pred

        self.metric: dict
        self.results: dict

        self.register_hooks(self.build_hooks())

    def build_hooks(self):
        """
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.

        Returns:
            list[HookBase]:
        """
        cfg = self.cfg

        ret = [
        ]

        def test_and_save_results():
            self.model.eval()
            self._last_eval_results = self._eval()
            return self._last_eval_results

        # ret.append(hooks.EvalHook(cfg, test_and_save_results))
        # if self.save_val_pred:
        #     ret.append(hooks.VisualPrediction(cfg))
        # # if dist.get_rank() == 0:
        #     ret.append(hooks.CheckpointContainer(cfg))

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
            self.logger.info('pretrained checkpoint is loaded.')

    def mode_info_printer(self):
        # model_str = model_info(self.model)
        # self.logger.info(model_str)
        self.logger.info(self.model)

    def run_step(self):
        self._trainer.iter = self.iter
        self._trainer.run_step()

    def train(self):
        super().train(self.start_iter, self.max_iter)

    def build_train_loader(self, cfg):
        train_dataloader = build_dataloader(self.train_dataset, cfg.dataloader.samples_per_gpu,
                                            cfg.dataloader.workers_per_gpu,
                                            len([cfg.local_rank, ]), seed=cfg.seed,
                                            shuffle=True,
                                            drop_last=True,
                                            distributed=cfg.distributed)
        return train_dataloader

    def build_val_loader(self, cfg):
        val_dataloader = build_dataloader(self.val_dataset, 1, cfg.dataloader.workers_per_gpu, len([cfg.local_rank, ]),
                                          seed=cfg.seed
                                          )
        return val_dataloader

    @torch.no_grad()
    def _eval(self):
        """
        Eval logic
        """
        final_collection = []
        total_frame = 0.0
        total_time = 0.0
        self.logger.info('Start to eval in val dataset:')
        prog_bar = ProgressBar(len(self.val_dataloader))
        for i, data in enumerate(self.val_dataloader):
            _img, _ground_truth = data['images_collect']['img'], data['ground_truth']
            _img = _img.cuda()
            for key, value in _ground_truth.items():
                if value is not None:
                    if isinstance(value, torch.Tensor):
                        _ground_truth[key] = value.cuda()
            cur_batch = _img.shape[0]
            total_frame += cur_batch
            start_time = time.time()
            predicts = self.model(_img)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_time += (time.time() - start_time)
            predict_gt_collection = combine_predicts_gt(predicts, data['images_collect']['img_metas'][0],
                                                        _ground_truth, self.network_type)
            final_collection.append(predict_gt_collection)
            for _ in range(cur_batch):
                prog_bar.update()
        print('\t %2f FPS' % (total_frame / total_time))
        return final_collection

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
        events.CommonMetricPrinter(max_iter)
    ]