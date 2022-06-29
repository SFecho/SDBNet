import os
import os.path as osp
from collections import OrderedDict

import torch
import torch.nn
import torch.nn.functional as F
import wandb
from omegaconf import OmegaConf
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP
from importlib import import_module

from layer import Loss, make_optimizer, Regular, make_scheduler
from utils.image import psnr, ssim
from utils.logger import get_logger, is_logging_process


def make_model(cfg):
    model = cfg.model
    # urls = cfg.urls
    params = cfg.params
    module = import_module(
        'model.' + model.lower())  # GOPRO data.dataset.GOPRO # JDBC a = Class.forname("com.aaa.bbb.ABCD").newInstance()
    dataset_class = getattr(module, model)

    model = dataset_class(**params)

    # if urls is not None:
    #     state_dict = torch.load(urls)
    #     model.load_state_dict(state_dict=state_dict)

    return model




class Model(nn.Module):
    def __init__(self, cfg, writer):
        super(Model, self).__init__()
        self.cfg = cfg
        self.device = self.cfg.device

        self.net_test = make_model(cfg.network).to(self.device)
        self.rank = self.cfg.dist.rank

        if self.device != "cpu" and self.cfg.dist.gpus != 0:
            self.net_train = DDP(self.net_test)#,device_ids=[self.rank])#, output_device=[self.rank], find_unused_parameters=True)
        else:
            self.net_train = self.net_test

        self.writer = writer
        self.epoch = 0
        self.per_epoch_step = 0
        self.per_test_step = 0
        self.batch_step = 0
        self._logger = get_logger(cfg, os.path.basename(__file__))
        self.grad_clip = cfg.network.grad_clip
        self.loss = Loss(cfg.loss).to(self.device)

        self.optimizer = make_optimizer(cfg.optimizer, self.net_train.parameters())
        # self.optimizer_loss = make_optimizer(cfg.optimizer, self.loss.parameters())
        # init loss
        self.scheduler = make_scheduler(cfg.scheduler, self.optimizer, self.epoch - 1)
        # self.scheduler_loss = make_scheduler(cfg.scheduler, self.optimizer_loss, self.epoch - 1)

        self.regular = Regular(cfg.regular).to(self.device)
        self.log = OmegaConf.create()

        # self._scheduler_dict = self.scheduler.state_dict()

        self.log.loss_epoch = 0
        self.log.loss_v = 0
        self.log.loss_test = 0
        self.log.psnr_test = 0
        self.log.ssim_test = 0


    def to_device(self, data):  # data's keys: input, GT
        if isinstance(data, Tensor):
            data_dev = data.to(self.device)
        elif isinstance(data, (tuple, list)):
            data_dev = []
            for item in data:
                data_dev.append(item.to(self.device))
        else:
            raise Exception('data format error!')

        return data_dev

    def clip_gradient(self):
        """
        Clips gradients computed during backpropagation to avoid explosion of gradients.

        :param optimizer: optimizer with the gradients to be clipped
        :param grad_clip: clip value
        """
        for group in self.optimizer.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    param.grad.data.clamp_(-self.grad_clip, self.grad_clip)

    def reset_epoch_loss(self):
        self.per_epoch_step = 0
        self.log.loss_epoch = 0

    def lrs_step(self):
        # self._scheduler_dict = self.scheduler.state_dict()
        self.scheduler.step()
        # self.scheduler_loss.step()

    def get_lr(self):
        return self.scheduler.get_lr()

    @property
    def train_epoch_loss(self):
        return self.log.loss_epoch / self.per_epoch_step

    @property
    def train_step_loss(self):
        return self.log.loss_v

    def forward(self, inputs):
        if isinstance(inputs, Tensor):
            inputs  = [inputs]
        modules = self.net_train if self.training and self.cfg.dist.gpus != 0 else self.net_test

        return modules(*inputs)


    def run_train(self, inputs, label):
        # self.net_train.train()
        inputs = self.to_device(inputs)
        label = self.to_device(label)

        self.optimizer.zero_grad()
        output = self(inputs)

        self.batch_step += 1
        self.per_epoch_step += 1
        # print(len(output), type(output), type(label), len(label))
        loss_v = self.loss(output, label) + self.regular(output)

        if self.cfg.dist.gpus > 0:
            # Aggregate loss_v from all GPUs. loss_v is set as the sum of all GPUs' loss_v.
            torch.distributed.all_reduce(loss_v)
            loss_v /= float(self.cfg.dist.gpus)

        loss_v.backward()
        # print(loss_v)
        self.optimizer.step()
        # self.optimizer_loss.step()

        loss_scalar = loss_v.cpu().detach().item()
        self.log.loss_v = loss_scalar
        self.log.loss_epoch += loss_scalar

        with torch.no_grad():

            if isinstance(output, (tuple, list)):
                output = output[-1]

            if isinstance(label, (tuple, list)):
                label = label[-1]

            save = torch.cat([inputs[0], output, label], dim=-2)

            # [blur, noise]
            # [blur_o, noise_gto, noise_o, blur_nonoise]
            # [blur, noise, noise, blur - noise]
            # save1 = torch.cat([inputs[0] - inputs[1], output[-1], label[-1]], dim=-1)
            # save3 = torch.cat([inputs[1] + 0.5, output[1] + 0.5, label[1] + 0.5], dim=-1)
            # save0 = torch.cat([inputs[0], output[0], label[0]], dim=-1)
            # save2 = torch.cat([inputs[1] + 0.5, output[2] + 0.5, label[2] + 0.5], dim=-1)
            # save = torch.cat([save0, save1, save2, save3], dim=-2)
        return save

    def reset_test_info(self):
        self.per_test_step = 0
        self.log.loss_test = 0
        self.log.psnr_test = 0
        self.log.ssim_test = 0


    def train(self, mode: bool = True):
        self.net_train.train()
        self.training = mode

    def eval(self):
        self.net_test.eval()
        self.training = False

    @property
    def test_loss(self):
        return self.log.loss_test / self.per_test_step


    @torch.no_grad()
    def run_test(self, inputs, label):
        # self.net_test.eval()
        self.per_test_step += 1

        inputs = self.to_device(inputs)
        label = self.to_device(label)

        output = self(inputs)

        if isinstance(output, (tuple, list)):
            save = output[-1]
        else:
            save = output


        # else:
        #

        loss_v = self.loss(output, label) + self.regular(output)

        loss_scalar = loss_v.cpu().detach().item()

        if isinstance(label, (tuple, list)):
            label = label[-1]

        self.log.loss_test += loss_scalar
        self.log.psnr_test += psnr(save, label)
        self.log.ssim_test += ssim(save, label)

        return save

    @property
    def test_psnr(self):
        return self.log.psnr_test / self.per_test_step

    @property
    def test_ssim(self):
        return self.log.ssim_test / self.per_test_step

    def inference(self):
        self.net.eval()
        output = self.run_network()

        return output

    def save_network(self, save_file=True):
        if is_logging_process():
            net = self.net_train#self.net.module if isinstance(self.net, DDP) else self.net
            state_dict = net.state_dict()
            for key, param in state_dict.items():
                state_dict[key] = param.to("cpu")
            if save_file:
                save_filename = "{}_{}.pth".format(self.cfg.name, self.epoch + 1)
                save_path = osp.join(self.cfg.log.chkpt_dir, save_filename)
                torch.save(state_dict, save_path)
                if self.cfg.log.use_wandb:
                    wandb.save(save_path)
                if is_logging_process():
                    self._logger.info("Saved network checkpoint to: {}".format(save_path))
            return state_dict

    def load_network(self, loaded_net=None):
        add_log = False
        if loaded_net is None:
            add_log = True
            if self.cfg.load.wandb_load_path is not None:
                self.cfg.load.network_chkpt_path = wandb.restore(
                    self.cfg.load.network_chkpt_path,
                    run_path=self.cfg.load.wandb_load_path,
                ).name

            loaded_net = torch.load(
                self.cfg.load.network_chkpt_path,
                map_location=torch.device(self.device),
            )

        loaded_clean_net = OrderedDict()  # remove unnecessary 'module.'
        for k, v in loaded_net.items():
            if k.startswith("module."):
                loaded_clean_net[k[7:]] = v
            else:
                loaded_clean_net[k] = v

        self.net_test.load_state_dict(loaded_clean_net, strict=self.cfg.load.strict_load)
        if is_logging_process() and add_log:
            self._logger.info(
                "Checkpoint {} is loaded".format(self.cfg.load.network_chkpt_path)
            )

    def save_training_state(self, is_latest=False):
        if is_logging_process():
            save_filename = "{}_epoch_{}.state".format(self.cfg.name, self.epoch + 1) if not is_latest else '{}_latest.state'.format(self.cfg.name)
            save_path = osp.join(self.cfg.log.chkpt_dir, save_filename)
            net_state_dict = self.save_network(False)
            state = {
                "model": net_state_dict,
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "step": self.batch_step,
                "epoch": self.epoch,
                "metrics": self.writer.metrics
            }
            torch.save(state, save_path)

            if self.cfg.log.use_wandb:
                wandb.save(save_path)

            if is_logging_process():
                self._logger.info("Saved training state to: %s" % save_path)

    def load_training_state(self):
        if self.cfg.load.wandb_load_path is not None:
            self.cfg.load.resume_state_path = wandb.restore(
                self.cfg.load.resume_state_path,
                run_path=self.cfg.load.wandb_load_path,
            ).name
        resume_state = torch.load(
            self.cfg.load.resume_state_path,
            map_location=torch.device(self.device),
        )
        # print(resume_state["optimizer"], resume_state["step"])
        self.load_network(loaded_net=resume_state["model"])
        self.optimizer.load_state_dict(resume_state["optimizer"])
        self.batch_step = resume_state["step"]
        self.epoch = resume_state["epoch"] + 1
        self.scheduler.load_state_dict(resume_state["scheduler"])

        for key in self.writer.metrics.keys():
            self.writer.metrics[key] = resume_state["metrics"][key].detach().cpu()

        if is_logging_process():
            self._logger.info(
                "Resuming from training state: %s" % self.cfg.load.resume_state_path
            )
