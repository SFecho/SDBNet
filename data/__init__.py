import glob
import os
from enum import Enum, auto
from importlib import import_module
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler
from data.dataloader import DataLoader_


class DataloaderMode(Enum):
    train = auto()
    test = auto()
    eval = auto()
    inference = auto()

def make_dataset(cfg):
    dataset = cfg.model
    module = import_module(
        'data.dataset')
    dataset_class = getattr(module, dataset)
    # print(cfg.params, dataset)
    return dataset_class(**cfg.params)

def make_dataloader(cfg, mode, rank):
    if cfg.data.use_background_generator:
        data_loader = DataLoader_
    else:
        data_loader = DataLoader

    train_use_shuffle = False
    sampler = None

    if mode is DataloaderMode.train:
        dataset = make_dataset(cfg.data.dataset.train)
        if cfg.dist.gpus > 0 and cfg.data.divide_dataset_per_gpu:
            sampler = DistributedSampler(dataset, cfg.dist.gpus, rank)
            train_use_shuffle = False
        return data_loader(
            dataset=dataset,
            shuffle=train_use_shuffle,
            sampler=sampler,
            pin_memory=True,
            drop_last=True,
            **cfg.data.dataloader.train
        )
    elif mode is DataloaderMode.test:
        dataset = make_dataset(cfg.data.dataset.test)
        # if cfg.dist.gpus > 0 and cfg.data.divide_dataset_per_gpu:
        #     sampler = DistributedSampler(dataset, cfg.dist.gpus, rank)
        return data_loader(
            dataset=dataset,
            shuffle=False,
            sampler=None,
            pin_memory=True,
            drop_last=True,
            **cfg.data.dataloader.test
        )
    elif mode is DataloaderMode.eval:
        dataset = make_dataset(cfg.dataloader.dataset.eval)
        # if cfg.dist.gpus > 0 and cfg.data.divide_dataset_per_gpu:
        #     sampler = DistributedSampler(dataset, cfg.dist.gpus, rank)
        return data_loader(
            dataset=dataset,
            shuffle=False,
            sampler=sampler,
            pin_memory=True,
            drop_last=True,
            **cfg.data.dataloader.eval
        )
    else:
        raise ValueError(f"invalid dataloader mode {mode}")
