import argparse
from utils import Config
import os
from trainer.trainer import TrainerContainer
import sys
import pathlib
import torch.distributed as dist
import torch


def parse_args():
    parser = argparse.ArgumentParser(description='cvalgorithms')
    parser.add_argument('--seed', type=int, default=2, help='random seed')
    parser.add_argument('--config', default='../config/rretinanet/train_retinanet_msra.json', help='train config file path')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--local_rank',
        default=0,
        type=int)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    torch.backends.cudnn.benchmark = True
    __dir__ = pathlib.Path(os.path.abspath(__file__))
    sys.path.append(str(__dir__))
    sys.path.append(str(__dir__.parent.parent))
    args = parse_args()
    cfg = Config.fromjson(args.config)
    if args.local_rank is not None:
        cfg.local_rank = args.local_rank
    if args.seed is not None:
        cfg.seed = args.seed
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    rank = int(os.environ['LOCAL_RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(
        backend='nccl',
        init_method="tcp://127.0.0.1:12345",
        world_size=torch.cuda.device_count(),
        rank=cfg.local_rank)
    trainer = TrainerContainer(cfg)
    return trainer.train()


if __name__ == "__main__":
    main()