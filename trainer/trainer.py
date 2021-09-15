from tools import BaseRunner, TrainerBase
from models import build_detector, build_segmentor
from datasets.builder import build_dataloader
from engine.optimizer import build_optimizer
from utils import Config, get_root_logger, model_info
import copy
from datasets import build_dataset
import time
import torch
import torch.nn as nn
import os.path as osp
import torch.distributed as dist

ModelBuilder = {"segmentation": build_segmentor,
                "rotate_detection": build_detector}


class TrainerContainer(BaseRunner):
    def __init__(self, cfg):
        """
        A trainer contain three tasks:
        """
        super().__init__()
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        work_dir = osp.join(cfg.checkpoint_dir, cfg.dataset.type, cfg.model.type, timestamp)
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

        self.logger = get_root_logger(log_file=work_dir, log_level='INFO')
        network_type = cfg.network_type
        # build dataloader
        train_dataloader = build_dataloader(self.train_dataset, cfg.dataloader.samples_per_gpu,
                                            cfg.dataloader.workers_per_gpu,
                                            len([cfg.local_rank, ]), seed=cfg.seed,
                                            drop_last=True)
        # build model
        model = ModelBuilder[network_type](cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
        # choose trainer from tools
        model = create_ddp_model(cfg, model)
        optimizer = build_optimizer(cfg, model)
        self._trainer = TrainerBase(model, train_dataloader, optimizer, self.logger)

    def run_step(self):
        self._trainer.epoch = self.epoch
        self._trainer.run_step()

    def train(self):
        super().train(self.start_epoch, self.max_epoch)

    @classmethod
    def _eval(cls, cfg, model, evaluators=None):
        pass


def create_ddp_model(cfg, model):
    dist.init_process_group(
        backend='nccl',
        init_method="tcp://127.0.0.1:33274",
        world_size=torch.cuda.device_count(),
        rank=cfg.local_rank

    )
    model = nn.parallel.DistributedDataParallel(model,
                                                device_ids=[cfg.local_rank, ],
                                                output_device=cfg.local_rank,
                                                find_unused_parameters=True
                                                )
    return model
