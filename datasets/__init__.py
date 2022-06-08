import random
import importlib
import numpy as np
from functools import partial
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler
from mmcv.runner import get_dist_info
from .pipeline.transform import LoadImageFromFile, RandomFlip, RandomAffine, FormatGroundTruth, ResizeAlign, \
    NormalizeImage
from .pipeline.loading import collate_function, Compose, Collect
from .coco import COCOPose
from .crowdpose import CrowdPose


def build(lib, args):
    assert 'type' in args
    model = getattr(lib, args.type)
    args.pop('type')

    return model(**args)


def build_dataset(cfg, data_cfg, is_training=True):
    args = cfg.copy()
    data_lib = importlib.import_module('datasets')

    # init pipeline objects
    pipeline = args.pipeline
    transforms = list()
    for item in pipeline:
        transforms.append(build(data_lib, item))
    args.pipeline = transforms

    # init dataset
    args['data_cfg'] = data_cfg
    args['test_mode'] = not is_training
    dataset = build(data_lib, args)

    return dataset


def worker_init_fn(worker_id, num_workers, rank, seed):
    """Worker init func for dataloader.

    The seed of each worker equals to num_worker * rank + worker_id + user_seed

    Args:
        worker_id (int): Worker id.
        num_workers (int): Number ofhrnet_w32-36af842 workers.
        rank (int): The rank of current process.
        seed (int): The random seed to use.
    """

    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_loader(cfg, distributed, dataset, seed, num_gpus):
    rank, world_size = get_dist_info()
    if distributed:
        sampler = DistributedSampler(dataset, world_size, rank, shuffle=True)
        shuffle = False
        batch_size = cfg.samples_per_gpu
        num_workers = cfg.workers_per_gpu
    else:
        sampler = None
        shuffle = True
        batch_size = num_gpus * cfg.samples_per_gpu
        num_workers = num_gpus * cfg.workers_per_gpu

    collate_fn = partial(collate_function, samples_per_gpu=cfg.samples_per_gpu)
    init_fn = partial(worker_init_fn, num_workers=num_workers, rank=rank, seed=seed) if seed is not None else None
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        shuffle=shuffle,
        worker_init_fn=init_fn,
        drop_last=True)

    return data_loader


def build_val_loader(dataset, num_workers=0, distributed=False):
    rank, world_size = get_dist_info()
    if distributed:
        sampler = DistributedSampler(dataset, world_size, rank, shuffle=True)
    else:
        sampler = None

    collate_fn = partial(collate_function, samples_per_gpu=1)
    data_loader = DataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn)

    return data_loader
