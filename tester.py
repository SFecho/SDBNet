import argparse
import datetime
import itertools
import math
import os
import random
import traceback
from decimal import Decimal

import hydra
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torchvision
import yaml
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf, open_dict
from tqdm import tqdm

from data import DataloaderMode, make_dataloader
from layer import make_scheduler
from model import Model
from utils.logger import get_logger, is_logging_process
from utils.random import set_random_seed
from utils.writer import Writer


def setup(cfg, rank):
    # os.environ["MASTER_ADDR"] = cfg.dist.master_addr
    # os.environ["MASTER_PORT"] = cfg.dist.master_port
    # timeout_sec = 1800
    # if cfg.dist.timeout is not None:
    #     os.environ["NCCL_BLOCKING_WAIT"] = "1"
    #     timeout_sec = cfg.dist.timeout
    # timeout = datetime.timedelta(seconds=timeout_sec)
    init_method = 'tcp://{}:{}'.format(cfg.dist.master_addr, cfg.dist.master_port)
    # initialize the process group
    dist.init_process_group(
        backend=cfg.dist.mode,
        init_method=init_method,
        rank=rank,
        world_size=cfg.dist.gpus
        # timeout=timeout,
    )


def cleanup():
    dist.destroy_process_group()


def distributed_run(fn, cfg):
    mp.spawn(fn, args=(), nprocs=cfg.dist.gpus, join=True)

class Tester(object):
    def __init__(self, cfg):
        super(Tester, self).__init__()
        self.cfg = cfg
        self.logger = get_logger(cfg, os.path.basename(__file__))
        rank = 0
        # train_datadir = cfg.data.dataset.train.params.data_dir
        test_datadir = cfg.data.dataset.test.params.data_dir

        self.batch_size = cfg.data.dataloader.train.batch_size
        self.writer = Writer(cfg)
        # if is_logging_process():
        os.makedirs(cfg.log.chkpt_dir, exist_ok=True)

        cfg_str = OmegaConf.to_yaml(cfg)
        self.logger.info("Config:\n" + cfg_str)

        if test_datadir == "":
            self.logger.error("train or test data directory cannot be empty.")
            raise Exception("Please specify directories of data")


        if is_logging_process():
            self.logger.info("Making test dataloader...")
        self.test_loader = make_dataloader(cfg, DataloaderMode.test, rank)

        self.model = Model(cfg, self.writer)

        if self.cfg.load.resume_state_path is not None:
            self.model.load_training_state()


    def __call__(self, is_save=True):
        logger = get_logger(self.cfg, os.path.basename(__file__))
        self.model.eval()
        self.model.reset_test_info()
        with torch.no_grad():
            for *model_input, target, filename in tqdm(self.test_loader):
                output = self.model.run_test(model_input, target)
                self.writer.save_image(filename, output)

            if is_logging_process():
                logger.info("Test Loss: {:.4f}, PSNR: {:.4f}, SSIM: {:.4f}".format(
                    self.model.test_loss,
                    self.model.test_psnr,
                    self.model.test_ssim
                ))


@hydra.main(config_path="config", config_name="default-testing")
def main(hydra_cfg):

    hydra_cfg.device = hydra_cfg.device.lower()

    with open_dict(hydra_cfg):
        hydra_cfg.job_logging_cfg = HydraConfig.get().job_logging
    # print(hydra_cfg.job_logging_cfg)
    # random seed
    if hydra_cfg.random_seed is None:
        hydra_cfg.random_seed = random.randint(1, 10000)
    set_random_seed(hydra_cfg.random_seed)

    if hydra_cfg.dist.gpus < 0:
        hydra_cfg.dist.gpus = torch.cuda.device_count()

    # hydra_cfg.
    if hydra_cfg.device == "cpu" or hydra_cfg.dist.gpus == 0:
        hydra_cfg.dist.gpus = 0
    print('rank:', hydra_cfg.dist.rank)
    #     train()
    # else:
    #     distributed_run(train, hydra_cfg)
    test = Tester(hydra_cfg)
    test()



if __name__ == "__main__":
    main()
