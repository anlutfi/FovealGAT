import torch
from torch import nn
from torch import optim
import pynvml
from torch_geometric.nn.norm import GraphNorm
import torchvision.transforms as T
from feature_extraction.pretrained import pyramid_feature_extractor

import model.gat_conv as convs
from model.xyt_gat_conv import CausalXYTGATConv, CausalEuclideanTGATConv
from model.iou_xyt_gat_conv import CausalIOU_EuclideanTGATConv
from model.residual_sequential import MarkovResidualSequential, DenseResidualSequential
import model.readout as readouts


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
        'leaky relu': nn.LeakyReLU,
        'Identity': nn.Identity}

norms = {'GraphNorm': GraphNorm,
         'BatchNorm2d': nn.BatchNorm2d,
         'GroupNorm': nn.GroupNorm,
         'Identity': nn.Identity
        }

F_acts = {'leaky_relu': nn.functional.leaky_relu,
          'elu': nn.functional.elu,
          'identity': nn.Identity(),
          'Identity': nn.Identity()
         }

winits = {'xavier': lambda w: nn.init.xavier_uniform_(w),
          'kaiming': lambda w: nn.init.kaiming_uniform_(w, nonlinearity='relu'),
          'kaiming_normal': lambda w: nn.init.kaiming_normal_(w, mode='fan_out', nonlinearity='relu'),
         }

transforms = {'RandomHorizontalFlip': T.RandomHorizontalFlip,
              'RandomRotation': T.RandomRotation,
              'ColorJitter': T.ColorJitter}

feat_exts = {'pyramid_feature_extractor': pyramid_feature_extractor}

gat_convs = {'CausalXYTGATConv': CausalXYTGATConv,
             'CausalEuclideanTGATConv': CausalEuclideanTGATConv,
             'CausalIOU_EuclideanTGATConv': CausalIOU_EuclideanTGATConv}

res_seqs = {'MarkovResidualSequential': MarkovResidualSequential,
            'DenseResidualSequential': DenseResidualSequential}

ros = {'CausalReadout': readouts.CausalReadout}


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


f_maker = lambda d: lambda arg: d[arg]

class ArgParser:
    identity_keys = ['train_val_test_split',
                     'batch_size',
                     'acc_steps',
                     'dl_num_workers',
                     'fe_channels',
                     'fe_bias',
                     'seq_encoding_feat_count',
                     'grph_layers',
                     'grph_channels',
                     'grph_residual_gat_levels',
                     'grph_output_gat_levels',
                     'grph_att_heads',
                     'grph_xy_heads_ratio',
                     'grph_t_heads_ratio',
                     'grph_iou_heads_ratio',
                     'grph_dropout',
                     'grph_bias',
                     'ro_channels',
                     'ro_out_channels',
                     'ro_dropout',
                     'ro_bias']
    class_table = {"transform": lambda ts: T.Compose([transforms[t[0]](**t[1]) for t in ts]),
                   "fe_tail_feat_ext": f_maker(feat_exts),
                   "fe_act": f_maker(acts),
                   "fe_norm": f_maker(norms),
                   "fe_w_init_f": f_maker(winits),
                   "grph_gat_conv": f_maker(gat_convs),
                   "grph_res_seq": f_maker(res_seqs),
                   "grph_act_att": f_maker(F_acts),
                   "grph_act_gat": f_maker(F_acts),
                   "grph_norm": f_maker(norms),
                   "grph_w_init_f": f_maker(winits),
                   "readout": f_maker(ros),
                   "ro_hidden_act": f_maker(acts),
                   "ro_pred_act": f_maker(acts),
                   "ro_norm": f_maker(norms),
                   "ro_w_init_f": f_maker(winits)}

    def __init__(self, d):
        self.d = d

    def __call__(self, key):
        assert key in ArgParser.identity_keys or key in ArgParser.class_table
        if key not in self.d:
            return None
        if key in ArgParser.identity_keys:
            return self.d[key]
        return ArgParser.class_table[key](self.d[key])
        


    