import os
import threading

import torch
import numpy as np
import wandb
from torch.utils.tensorboard import SummaryWriter
from enum import Enum, auto
import matplotlib.pyplot as plt

from utils.image import imwrite


class Writer:

    class Option(Enum):
        @staticmethod
        class Loss(Enum):
            train_epoch = auto()
            train_step = auto()
            test = auto()
        @staticmethod
        class Metrics(Enum):
            psnr = auto()
            ssim = auto()

    def __init__(self, cfg):
        # super(Writer, self).__init__()
        self.cfg = cfg
        self.tensorboard = 'tensorboard'
        self.testout_imagedir = cfg.data.dataset.test.name

        if not os.path.exists(self.testout_imagedir):
            os.mkdir(self.testout_imagedir)

        self.infos = {
                Writer.Option.Loss.train_epoch: torch.Tensor(),
                Writer.Option.Loss.train_step: torch.Tensor(),
                Writer.Option.Loss.test: torch.Tensor(),
                Writer.Option.Metrics.psnr: torch.Tensor(),
                Writer.Option.Metrics.ssim: torch.Tensor()
        }

        if cfg.log.use_tensorboard:
            self.tensorboard = SummaryWriter(self.tensorboard)

        if cfg.log.use_wandb:
            wandb_init_conf = cfg.log.wandb_init_conf
            wandb.init(config=cfg, **wandb_init_conf)


    def logging_info(self, value, step, logging_name, mode=Option.Loss.train_step):
        info = self.infos[mode]
        self.infos[mode] = torch.cat((info, torch.zeros(1)))
        self.infos[mode][-1] = value

        if self.cfg.log.use_wandb:
            wandb.log({logging_name: value}, step=step)

        if self.cfg.log.use_tensorboard:
            self.tensorboard.add_scalar(logging_name, value, step)

        # torch.save(self.infos, 'metrics.pt')

    @property
    def metrics(self):
        return self.infos

    @metrics.setter
    def metrics(self, infos):
        self.infos = infos

    def add_image(self, tag, img_tensor, global_step=None, walltime=None, dataformats='CHW'):
        if self.cfg.log.use_tensorboard:
            self.tensorboard.add_image(tag, img_tensor, global_step=global_step, walltime=walltime, dataformats=dataformats)

    def plot_log(self, step, linewidth = '0.5', color='#1f77b4', mode=Option.Loss.train_epoch):
        fig = plt.figure()

        if mode in [Writer.Option.Loss.train_epoch, Writer.Option.Loss.train_step]:
            legend = self.cfg.data.dataset.train.name
            title = 'Train-Loss'
            y_label = 'Loss'
        elif mode is Writer.Option.Loss.test:
            legend = self.cfg.data.dataset.test.name
            title = 'Test Loss'
            y_label = 'Loss'
        elif mode is Writer.Option.Metrics.psnr:
            legend = self.cfg.data.dataset.test.name
            title = 'Test PSNR'
            y_label = 'PSNR'
        elif mode is Writer.Option.Metrics.ssim:
            legend = self.cfg.data.dataset.test.name
            title = 'Test SSIM'
            y_label = 'SSIM'
        else:
            raise Exception('mode expection!')

        plt.title(title)
        axis = np.linspace(1, step, step)

        if mode in [Writer.Option.Loss.train_epoch, Writer.Option.Metrics.psnr, Writer.Option.Metrics.ssim]:
            x_label = 'Epoch'
        else:
            x_label = 'Step'
        values = self.infos[mode]

        line, = plt.plot(axis, values, linewidth=linewidth, color=color)
        plt.legend(handles=[line], labels=[legend], loc='best')
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.grid(True)
        filename = '{}_{}'.format(title.lower(), x_label.lower())
        plt.savefig(os.path.join('{}.pdf'.format(filename)))
        plt.close(fig)

    def save_image(self, filename, img):
        for batch_idx, name, in enumerate(filename):
            name, ext = os.path.splitext(name)
            # print(name.split('.'))
            # name, ext = name.split('.')
            if ext not in ['png', 'jpg', 'PNG', 'JPG', 'bmp', 'BMP']:
                ext = 'png'
            name = name + '.' + ext
            imwrite(os.path.join(self.testout_imagedir, name), img, batch_idx=batch_idx)



