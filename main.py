import argparse
from utils import Config
import os
from trainer.trainer import TrainerContainer
import sys
import pathlib


def parse_args():
    parser = argparse.ArgumentParser(description='cvalgorithms')
    parser.add_argument('--seed', type=int, default=2, help='random seed')
    parser.add_argument('--config', default='../config/rretinanet/train_retinanet_msra.json', help='train config file path')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--local_rank',
        default=0,
        type=int)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    __dir__ = pathlib.Path(os.path.abspath(__file__))
    sys.path.append(str(__dir__))
    sys.path.append(str(__dir__.parent.parent))
    args = parse_args()
    cfg = Config.fromjson(args.config)
    trainer = TrainerContainer(cfg)
    return trainer.train()


if __name__ == "__main__":
    main()