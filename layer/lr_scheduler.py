# _LRScheduler
import warnings

import torch as t
import torch.nn as nn
import torch.optim.lr_scheduler as lrs
from torch.optim.lr_scheduler import ReduceLROnPlateau

from layer import make_scheduler


class StepLR(lrs.StepLR):
    def __init__(self, optimizer, step_size, gamma=0.1, last_epoch=-1, verbose=False, **kwargs):
        super(StepLR, self).__init__(optimizer=optimizer, step_size=step_size, gamma=gamma, last_epoch=last_epoch,verbose=verbose)
        
class MultiStepLR(lrs.MultiStepLR):
    def __init__(self, optimizer, milestones, gamma=0.1, last_epoch=-1, verbose=False, **kwargs):
        super(MultiStepLR, self).__init__(optimizer=optimizer, milestones=milestones, gamma=gamma, last_epoch=last_epoch, verbose=verbose)

class CosineAnnealingLR(lrs.CosineAnnealingLR):
    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1, verbose=False, **kwargs):
        super(CosineAnnealingLR, self).__init__(optimizer=optimizer, T_max=T_max, eta_min=eta_min, last_epoch=last_epoch, verbose=verbose)

class LineStepLR(lrs._LRScheduler):
    def __init__(self, optimizer, end_lr, max_epoch, last_epoch=-1, verbose=False, **kwargs):
        super(LineStepLR, self).__init__(optimizer, last_epoch, verbose)
        self.max_epoch = max_epoch
        self.end_lr = end_lr
        self.grad_lrs = [(end_lr - base_lr) / max_epoch for base_lr in self.base_lrs]

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        if self.last_epoch == 0 or self.last_epoch == self.max_epoch:
            return [group['lr'] for group in self.optimizer.param_groups]

        # print(self.base_lrs)

        return [group['lr'] + grad_lr
                for grad_lr, group in zip(self.grad_lrs, self.optimizer.param_groups)]

    def _get_closed_form_lr(self):
        # print(self.base_lrs)
        if self.last_epoch < self.max_epoch:
            return [base_lr + ((self.end_lr - base_lr) / self.max_epoch) * self.last_epoch
                    for base_lr in self.base_lrs]
        return [group['lr'] for group in self.optimizer.param_groups]





class GradualWarmupScheduler(lrs._LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, multiplier, total_epoch, last_epoch=-1, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier < 1.:
            raise ValueError('multiplier should be greater thant or equal to 1.')
        self.total_epoch = total_epoch

        # self.after_scheduler_info = after_scheduler
        # self.after_scheduler_params = after_scheduler.params

        if after_scheduler is not None:
            after_scheduler = make_scheduler(after_scheduler, optimizer, -1)

        self.after_scheduler = after_scheduler
        # print('self.after_scheduler：', self.after_scheduler.T_max)
        self.finished = False
        super(GradualWarmupScheduler, self).__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True

                # print(self.last_epoch, self.after_scheduler.get_lr()[0])
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch if epoch != 0 else 1  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                    # print('aaabbb')
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        params = {}
        for key, value in self.__dict__.items():
            if key != 'optimizer' and key != "after_scheduler":
                params[key] = value

        if self.after_scheduler is not None:
            params['after_scheduler'] = {}
            for key, value in self.after_scheduler.__dict__.items():
                params['after_scheduler'][key] = value
        else:
            params['after_scheduler'] = None

        return params #{key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        after_scheduler = state_dict.pop('after_scheduler')

        if after_scheduler is not None:
            self.after_scheduler.__dict__.update(after_scheduler)

        # self.__dict__.update(state_dict)
        """Loads the schedulers state.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)