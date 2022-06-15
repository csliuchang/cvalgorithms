from utils import Registry, build_from_cfg, get_dist_info
from .samper import GroupSampler
import torch
from functools import partial
import numpy as np
import torch.nn.functional as F
import random
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

DATASETS = Registry('dataset')
PIPELINES = Registry('pipeline')


def build_dataset(cfg, default_args=None):
    dataset = build_from_cfg(cfg, DATASETS, default_args)
    return dataset


def build_dataloader(dataset,
                     samples_per_gpu,
                     workers_per_gpu,
                     num_gpus=1,
                     shuffle=False,
                     seed=None,
                     distributed=False,
                     **kwargs):
    """Build Pytorch Dataloader.


    Returns:
        Dataloader: A Pytorch dataloader
    """
    rank, world_size = get_dist_info()
    if distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=shuffle, seed=seed)
        batch_size = samples_per_gpu
        num_workers = workers_per_gpu
    else:
        sampler = GroupSampler(dataset, samples_per_gpu) if shuffle else None
        batch_size = num_gpus * samples_per_gpu
        num_workers = 0

    init_fn = partial(
        worker_init_fn, num_workers=num_workers, rank=rank, seed=seed
    ) if seed is not None else None

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=partial(collate, samples_per_gpu=samples_per_gpu),
        pin_memory=False,
        worker_init_fn=init_fn,
        **kwargs
    )

    return data_loader


def worker_init_fn(worker_id, num_workers, rank, seed):
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def collate(batch, samples_per_gpu=1):
    """
    A batch collator that does nothing.
    """
    """Puts each data field into a tensor/DataContainer with outer dimension
    batch size.

    Extend default_collate to add support for
    :type:`~mmcv.parallel.DataContainer`. There are 3 cases.

    1. cpu_only = True, e.g., meta data
    2. cpu_only = False, stack = True, e.g., images tensors
    3. cpu_only = False, stack = False, e.g., gt bboxes
    """

    """
    A batch collator that does nothing.
    """
    # images, img_metas, gt_bboxes, gt_labels, gt_masks = [], [], [], [], []
    # for sample in batch:
    #     image = sample['img']
    #     img_meta = sample['img_metas']
    #     images.append(image)
    #     img_metas.append(img_meta)
    #     if 'gt_bboxes' in sample:
    #         gt_bboxes.append(sample['gt_bboxes'])
    #         gt_labels.append(sample['gt_labels'])
    #         gt_masks.append(sample['gt_masks'])
    #     elif 'masks' in sample:
    #         gt_masks.append(sample['masks'])
    # images = torch.stack(images, dim=0)
    # images_collect = dict(img=images, img_metas=img_metas)
    # if gt_bboxes:
    #     ground_truth = dict(gt_bboxes=gt_bboxes, gt_labels=gt_labels, gt_masks=gt_masks)
    # else:
    #     gt_masks = torch.stack(gt_masks, dim=0)
    #     ground_truth = dict(gt_masks=gt_masks)
    # return dict(images_collect=images_collect, ground_truth=ground_truth)
    for idx in range(0, len(batch), samples_per_gpu):
        stacked_img = [sample['image'] for sample in
                       batch[idx: idx + samples_per_gpu]
                       ]
        if 'masks' in batch[0]['annotations']:
            stacked_mask = [sample['annotations']['masks'] for sample in
                            batch[idx: idx + samples_per_gpu]
                            ]
    inputs_img = torch.stack(stacked_img, dim=0)
    gt_mask = torch.stack(stacked_mask, dim=0)
    images_collect = {'img': inputs_img}
    ground_truth = {"gt_masks": gt_mask}
    return {'images_collect': images_collect, 'ground_truth': ground_truth
            }
