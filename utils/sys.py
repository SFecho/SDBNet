import psutil
import pynvml
import torch


def get_avaliable_memory(device):
    if device == torch.device('cuda:0') or device == 'cuda:0':
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        ava_mem=round(meminfo.free/1024**2)
        print('current available video memory is' +' : '+ str(round(meminfo.free/1024**2)) +' MIB')
    elif device == torch.device('cpu') or device == 'cpu':
        mem = psutil.virtual_memory()
        print('current available memory is' +' : '+ str(round(mem.used/1024**2)) +' MIB')
        ava_mem=round(mem.used/1024**2)
    else:
        raise Exception('no supported!')
    return ava_mem