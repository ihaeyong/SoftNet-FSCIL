import importlib
import numpy as np
import random
import torch
import torch.utils.data
from functools import partial
from os import path as osp
import os

from data.prefetch_dataloader import PrefetchDataLoader
from utils import get_root_logger

__all__ = ['create_dataset', 'create_sampler', 'create_dataloader']

# automatically scan and import dataset modules
# scan all the files under the data folder with '_dataset' in file names
data_folder = osp.dirname(osp.abspath(__file__))

dataset_filenames = [
    osp.splitext(osp.basename(v))[0] for v in os.scandir(data_folder)
    if '_dataset.py' in v.name
]

sampler_filenames = [
    osp.splitext(osp.basename(v))[0] for v in os.scandir(data_folder)
    if '_sampler.py' in v.name
]

# import all the dataset modules
_dataset_modules = [
    importlib.import_module(f'data.{file_name}')
    for file_name in dataset_filenames
]

# import all the sampler modules
_sampler_modules = [
    importlib.import_module(f'data.{file_name}')
    for file_name in sampler_filenames
]


def create_dataset(dataset_opt, info=True):
    """Create dataset.

    Args:
        dataset_opt (dict): Configuration for dataset. It constains:
            name (str): Dataset name.
            type (str): Dataset type.
    """
    dataset_type = dataset_opt['type']

    # dynamic instantiation
    for module in _dataset_modules:
        dataset_cls = getattr(module, dataset_type, None)
        if dataset_cls is not None:
            break
    if dataset_cls is None:
        raise ValueError(f'Dataset {dataset_type} is not found.')

    # training type
    dataset = dataset_cls(dataset_opt)

    if info:
        logger = get_root_logger()
        logger.info(
            f'Dataset {dataset.__class__.__name__} - {dataset_opt["name"]} '
            'is created.')
    return dataset


def create_sampler(train_set, sampler_opt):
    """Create sampler.

    Args:
        train_set (torch.utils.data.Dataset): Training Set
        sampler_opt (dict): Configuration for dataset. It constains:
            num_classes (int): Number of classes for training.
            num_samples (int): Number of samples per training class
    """
    sampler_type = sampler_opt['type']
    if sampler_type is not None:
        for module in _sampler_modules:
            sampler_cls = getattr(module, sampler_type, None)
            if sampler_cls is not None:
                break
        if sampler_cls is None:
            raise ValueError(f'Sampler {sampler_type} is not found.')
        sampler = sampler_cls(train_set, sampler_opt)
    else:
        sampler = None
    return sampler


def create_dataloader(dataset,
                      dataset_opt,
                      sampler=None,
                      seed=None):
    """Create dataloader.

    Args:
        dataset (torch.utils.data.Dataset): Dataset.
        dataset_opt (dict): Dataset options. It contains the following keys:
            phase (str): 'train' or 'val'.
            num_worker_per_gpu (int): Number of workers for each GPU.
            batch_size (int): Training batch size.
        sampler (torch.utils.data.sampler): Data sampler. Default: None.
        seed (int | None): Seed. Default: None
    """
    phase = dataset_opt['phase']
    rank, _ = 0, 1
    if phase == 'train':
        batch_size = dataset_opt['batch_size']
        num_workers = dataset_opt['num_worker_per_gpu']
        if sampler is not None and 'Batch' in sampler.__class__.__name__:
            dataloader_args = dict(
                dataset=dataset,
                batch_sampler=sampler,
                num_workers=num_workers,
            )
        else:
            dataloader_args = dict(
                dataset=dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                sampler=sampler,
                drop_last=True)
            if sampler is None:
                dataloader_args['shuffle'] = True
        dataloader_args['worker_init_fn'] = partial(
            worker_init_fn, num_workers=num_workers, rank=rank,
            seed=seed) if seed is not None else None
    elif phase in ['val', 'test']:  # validation
        dataloader_args = dict(
            dataset=dataset, batch_size=128, num_workers=4, shuffle=False, drop_last=False)
    else:
        raise ValueError(f'Wrong dataset phase: {phase}. '
                         "Supported ones are 'train', 'val' and 'test'.")

    dataloader_args['pin_memory'] = dataset_opt.get('pin_memory', False)

    # prefetch_mode = dataset_opt.get('prefetch_mode')
    # if prefetch_mode == 'cpu':  # CPUPrefetcher
    #     num_prefetch_queue = dataset_opt.get('num_prefetch_queue', 1)
    #     logger = get_root_logger()
    #     logger.info(f'Use {prefetch_mode} prefetch dataloader: '
    #                 f'num_prefetch_queue = {num_prefetch_queue}')
    #     return PrefetchDataLoader(
    #         num_prefetch_queue=num_prefetch_queue, **dataloader_args)
    # else:
    #     # prefetch_mode=None: Normal dataloader
    #     # prefetch_mode='cuda': dataloader for CUDAPrefetcher
    #     return torch.utils.data.DataLoader(**dataloader_args)
    return torch.utils.data.DataLoader(**dataloader_args)

# set random seed for all workers (defualt: using the worker id as random seed)
def worker_init_fn(worker_id, num_workers, rank, seed):
    # Set the worker seed to num_workers * rank + worker_id + seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)
