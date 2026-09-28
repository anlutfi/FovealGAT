import torch
from torch import nn
from torch import optim
import pynvml

losses = {"mse": nn.MSELoss,
          "l2": nn.MSELoss,
          "l1": nn.L1Loss,
          "bce": nn.BCELoss,
          "bce_with_logits": nn.BCEWithLogitsLoss,
          "kl": nn.KLDivLoss,
          "kl_divergence": nn.KLDivLoss,
          "huber": nn.HuberLoss}

optims = {"sgd": optim.SGD,
          "adam": optim.Adam,
          "adamw": optim.AdamW,
          "rmsprop": optim.RMSprop,
          "adagrad": optim.Adagrad,
          "adamax": optim.Adamax,
          "asgd": optim.ASGD,
          "rprop": optim.Rprop}

acts = {'relu': nn.ReLU,
        'gelu': nn.GELU,
        'leaky relu': nn.LeakyReLU}


def get_obj(nm, obj_dict):
    nm = nm.lower().strip()
    return obj_dict[nm]

get_loss = lambda nm: get_obj(nm, losses)
get_optim = lambda nm: get_obj(nm, optims)
get_act = lambda nm: get_obj(nm, acts)


def choose_device():
    """chooses GPU with smallest current memory load
    """
    if not torch.cuda.is_available():
        return 'cpu'

    pynvml.nvmlInit()

    best_device = None
    min_load = float('inf')

    for i in range(pynvml.nvmlDeviceGetCount()):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        mem_used = pynvml.nvmlDeviceGetMemoryInfo(handle).used

        if mem_used < min_load:
            min_load = mem_used
            best_device = i

    pynvml.nvmlShutdown()
    return 'cuda:{}'.format(best_device)


    