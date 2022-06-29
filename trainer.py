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
from utils.image import imwrite
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

class Trainer(object):
    def __init__(self, cfg):
        super(Trainer, self).__init__()
        self.cfg = cfg
        self.logger = get_logger(cfg, os.path.basename(__file__))
        rank = cfg.dist.rank
        train_datadir = cfg.data.dataset.train.params.data_dir
        test_datadir = cfg.data.dataset.test.params.data_dir


        if cfg.device == "cuda" and cfg.dist.gpus != 0:
            self.cfg.device = rank
            # turn off background generator when distributed run is on
            self.cfg.data.use_background_generator = False
            setup(cfg, rank)
            # torch.cuda.set_device(cfg.device)

        self.batch_size = cfg.data.dataloader.train.batch_size
        self.writer = Writer(cfg)
        if is_logging_process():
            os.makedirs(cfg.log.chkpt_dir, exist_ok=True)

            cfg_str = OmegaConf.to_yaml(cfg)
            self.logger.info("Config:\n" + cfg_str)

            if train_datadir == "" or test_datadir == "":
                self.logger.error("train or test data directory cannot be empty.")
                raise Exception("Please specify directories of data")

            self.logger.info("Set up train process")
            self.logger.info("BackgroundGenerator is turned off when Distributed running is on")


        if cfg.dist.gpus != 0:
            dist.barrier()

        # make dataloader
        if is_logging_process():
            self.logger.info("Making train dataloader...")
        self.train_loader = make_dataloader(cfg, DataloaderMode.train, rank)

        if is_logging_process():
            self.logger.info("Making test dataloader...")
        self.test_loader = make_dataloader(cfg, DataloaderMode.test, rank)

        self.n_traindata = len(self.train_loader.dataset)
        self.model = Model(cfg, self.writer)

        self.is_distribute = True if cfg.dist.gpus > 0 and cfg.data.divide_dataset_per_gpu else False
        self.sample_train = self.train_loader.sampler if self.is_distribute == True else None

        if self.cfg.load.resume_state_path is not None:
            self.model.load_training_state()
        else:
            if is_logging_process():
                self.logger.info("Starting new training run.")



    def train_per_epoch(self):
        def is_writelog(idx):
            return (idx) % self.cfg.log.summary_interval == 0 or (idx) * self.batch_size >= self.n_traindata

        self.model.lrs_step()
        lr = self.model.get_lr()[0]
        logger = get_logger(self.cfg, os.path.basename(__file__))
        logger.info('[Epoch {}]\tGenerator Learning rate: {:.4e}'.format(self.model.epoch + 1, Decimal(lr)))

        self.model.train()
        self.model.reset_epoch_loss()
        for idx, (*input_, target, _) in enumerate(self.train_loader):
            # print(input_, len(target))
            save = self.model.run_train(input_, target)
            loss = self.model.train_step_loss
            # print('loss:', loss)
            # print(loss)
            # print(loss)
            # print('idxidx:', idx)
            if is_logging_process() and (loss > 1e8 or math.isnan(loss)):
                logger.error("Loss exploded to %.02f at step %d!" % (loss, self.model.batch_step))
                raise Exception("Loss exploded")

            if self.writer is not None:
                self.writer.logging_info(loss, self.model.batch_step, "train/loss_iter", self.writer.Option.Loss.train_step)

            if is_writelog(idx + 1):
                if is_logging_process():
                    logger.info('[{}/{}]\tTrain Loss:{:.4f}'.format(
                            (idx + 1) * self.batch_size,
                            self.n_traindata,
                            loss))
                if self.writer is not None:
                    self.writer.add_image('train/result', save[0])
                    imwrite('save.jpg', save)

        # if is_logging_process():
        #     logger.info('[{}/{}]\tTrain Loss:{:.4f}'.format(
        #         (idx) * self.batch_size,
        #         self.n_traindata,
        #         loss))

        if self.writer is not None:
            self.writer.logging_info(self.model.train_epoch_loss, self.model.epoch + 1, "train/loss_epoch", Writer.Option.Loss.train_epoch)
            self.writer.plot_log(self.model.epoch + 1, mode=Writer.Option.Loss.train_epoch)
            self.writer.plot_log(self.model.batch_step, mode=Writer.Option.Loss.train_step)



    def test(self):
        logger = get_logger(self.cfg, os.path.basename(__file__))
        self.model.eval()
        self.model.reset_test_info()

        with torch.no_grad():
            for (*model_input, target, filename) in tqdm(self.test_loader):
                # print(model_input[0].shape, model_input[1].shape)
                output = self.model.run_test(model_input, target)
                # print(output[-1].shape)
                self.writer.save_image(filename, output)

            # if self.writer is not None:
            #     self.writer.logging_info(self.model.test_loss, self.model.epoch + 1, 'test/loss', mode=Writer.Option.Loss.test)
            #     self.writer.logging_info(self.model.test_psnr, self.model.epoch + 1, 'test/psnr', mode=Writer.Option.Metrics.psnr)
            #     self.writer.logging_info(self.model.test_ssim, self.model.epoch + 1, 'test/ssim', mode=Writer.Option.Metrics.ssim)
            #
            #     self.writer.plot_log(self.model.epoch + 1, mode=Writer.Option.Loss.test)
            #     self.writer.plot_log(self.model.epoch + 1, mode=Writer.Option.Metrics.psnr)
            #     self.writer.plot_log(self.model.epoch + 1, mode=Writer.Option.Metrics.ssim)

            if is_logging_process():
                logger.info("Test Loss: {:.4f}, PSNR: {:.4f}, SSIM: {:.4f}".format(
                    self.model.test_loss,
                    self.model.test_psnr,
                    self.model.test_ssim,
                    self.model.epoch + 1)
                )


    def __call__(self, *args):

        try:
            if self.cfg.dist.gpus == 0 or self.cfg.data.divide_dataset_per_gpu:
                epoch_step = 1
            else:
                epoch_step = self.cfg.dist.gpus


            for self.model.epoch in itertools.count(self.model.epoch, epoch_step):
                if self.model.epoch + 1 > self.cfg.num_epoch:
                    break

                if is_logging_process() and self.is_distribute:
                    self.sample_train.set_epoch(self.model.epoch)
                self.train_per_epoch()

                if is_logging_process():
                    # self.test()
                    self.model.save_training_state(is_latest=True) # save current epoch state

                    if self.model.epoch % self.cfg.log.chkpt_interval == 0 or self.model.epoch + 1 == self.cfg.num_epoch:
                        self.test()
                        self.model.save_network()
                        self.model.save_training_state(is_latest=False)


            if is_logging_process():
                self.logger.info("End of Train")

        except Exception as e:
            if is_logging_process():
                self.logger.error(traceback.format_exc())
            else:
                traceback.print_exc()
        finally:
            if self.cfg.dist.gpus != 0:
                cleanup()

@hydra.main(config_path="config", config_name="default-training")
def main(hydra_cfg):

    hydra_cfg.device = hydra_cfg.device.lower()

    with open_dict(hydra_cfg):
        hydra_cfg.job_logging_cfg = HydraConfig.get().job_logging

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
    train = Trainer(hydra_cfg)
    train()


if __name__ == "__main__":
    main()
