from importlib import import_module
from torch.nn.modules.loss import _Loss
import torch
from torch import nn
# torch.optim.AdamW
import torch.nn.functional as F

def make_optimizer(cfg, trainable):
    trainable = filter(lambda x: x.requires_grad, trainable)
    optim_name = cfg.model
    module = import_module('torch.optim')
    params = cfg.params
    dataset_class = getattr(module, optim_name)

    if params is None:
        raise NotImplementedError("Parameter cannot be None!")
    return dataset_class(trainable, **params)


def make_loss(cfg):
    loss_name = cfg.model
    params = cfg.params
    module = import_module('layer.loss')  # GOPRO data.datasetorch.GOPRO # JDBC a = Class.forname("com.aaa.bbb.ABCD").newInstance()
    loss_class = getattr(module, loss_name)
    return loss_class(**params) if params is not None else loss_class()

def make_regular(cfg):
    regular_name = cfg.model

    if regular_name is None:
        return nn.Identity()

    params = cfg.params
    module = import_module('layer.regular')  # GOPRO data.datasetorch.GOPRO # JDBC a = Class.forname("com.aaa.bbb.ABCD").newInstance()
    regular_class = getattr(module, regular_name)
    return regular_class(**params) if params is not None else regular_class()

def make_scheduler(cfg, optimizer, last_epoch):

    sche_name = cfg.model
    # print('sche_name：', sche_name)
    module = import_module('layer.lr_scheduler')
    params = cfg.params
    if hasattr(module, sche_name):
        model_class = getattr(module, sche_name)
    else:
        del module
        module = import_module('torch.optim.lr_scheduler')
        model_class = getattr(module, sche_name)


    if params is None:
        raise NotImplementedError("Parameter cannot be None!")

    return model_class(optimizer, last_epoch=last_epoch, **params) # scheduler


class Loss(_Loss):
    def __init__(self, cfg):
        super(Loss, self).__init__()

        self.reg_coefficient = {}
        self.loss_modules = nn.ModuleDict()
        self.loss_cfg = cfg
        self.lossnames = cfg.keys()

        for lossname in self.lossnames:
            cur_loss = cfg[lossname]
            model = make_loss(cur_loss)

            loss_reg_coeff = cur_loss.reg_coefficient
            self.reg_coefficient[lossname] = loss_reg_coeff
            self.loss_modules[lossname] = model


    def forward(self, input, label):

        loss = 0
        for key in self.lossnames:
            loss += (self.reg_coefficient[key] * self._compute_loss(self.loss_modules[key], input, label))

        return loss

    def _compute_loss(self, model, input, label):
        # loss = 0
        # if isinstance(input, (list, tuple)) and isinstance(label, (list, tuple)):
        #     for (x, y) in zip(input, label):
        #         loss = loss + model(x, y)
        # else:
        #     loss = model(input, label)

        loss = model(input, label)

        return loss


class Regular(nn.Module):
    def __init__(self, cfg):
        super(Regular, self).__init__()

        self.reg_coefficient = {}
        self.regular_modules = nn.ModuleDict()
        self.regular_cfg = cfg
        self.regular_names = cfg.keys()

        for regular_name in self.regular_names:
            cur_regular = cfg[regular_name]
            model = make_regular(cur_regular)

            loss_reg_coeff = cur_regular.reg_coefficient
            self.reg_coefficient[regular_name] = loss_reg_coeff
            self.regular_modules[regular_name] = model

    def forward(self, input):
        reg = 0
        for key in self.regular_names:
            reg += (self.reg_coefficient[key] * self._compute_reg(self.regular_modules[key], input))
        return reg

    def _compute_reg(self, model, input):
        reg = 0
        if isinstance(input, (list, tuple)):
            for x in input:
                reg = reg + model(x)
        elif isinstance(input, torch.Tensor):
            reg = model(input)
        else:
            raise Exception('input data type error!')

        return reg


# def make_discriminator(**kwargs):
#     discriminator_name = kwargs['model']
#     params = kwargs['params']
#     module = import_module('layer.discriminator')
#     discriminator_class = getattr(module, discriminator_name)
#
#
#     return discriminator_class(**params) if params is not None else discriminator_class()
