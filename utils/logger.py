import logging
import logging.config
from datetime import datetime
import torch
import pynvml
import psutil



import torch.distributed as dist
from omegaconf import OmegaConf

def is_logging_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def get_logger(cfg, name=None):
    # log_file_path is used when unit testing
    if is_logging_process():
        logging.config.dictConfig(
            OmegaConf.to_container(cfg.job_logging_cfg, resolve=True)
        )
        return logging.getLogger(name)