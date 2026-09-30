from torch import nn
from torch_geometric.nn.norm import GraphNorm
import torch.nn.functional as F

from model.tools import adjust_args
from model.gat_conv import CausalGATConv
from model.xyt_gat_conv import CausalXYTGATConv
from model.residual_sequential import MarkovResidualSequential
from gaze_datasets.dataset_tools import get_xyt_pair_matrices


class GazeGrapher(nn.Module):
    def __init__(self,
                 gat_conv=CausalGATConv,
                 output_gat_levels=None,
                 residual_gat_levels=None,
                 gat_conv_args={'channels': [67, 134, 134, 134, 67],
                                'att_heads': 8,
                                'act_att': F.leaky_relu,
                                'act_gat': F.elu,
                                'norm': GraphNorm,
                                'dropout': 0.1,
                                'bias': True,
                                'w_init_f': lambda w: nn.init.xavier_uniform_(w)},
                 device='cpu'):
        """A module that applies a series of graph convolutions
        param:gat_conv which type of convolution to apply among those in model.GATConv
        param:readout_gat_levels a list of indices of the layers' outputs to be returned by forward
        param:channels a list of input channels, each pair will correspond to a convolution layer. ALL PARAMS from att_heads to bias need to be either a single value or a list of values OF THE SAME LENGTH as channels
        param:act_att activation function for the attention scores of each convolution layer. if a single value, it will be replicated for all
        param:act_gat activation function for the node update of each convolution layer. if a single value, it will be replicated for all
        """
        
        adjargs = adjust_args(list(gat_conv_args.values()))
        gat_conv_args = {k: adjargs[i] for i, k in enumerate(gat_conv_args)}
        channels = gat_conv_args['channels']

        if output_gat_levels is None or len(output_gat_levels) == 0:
            output_gat_levels = list(range(len(channels)))

        assert len(output_gat_levels) <= len(channels)
        for l in output_gat_levels:
            assert l < len(channels)

        if residual_gat_levels is None or len(residual_gat_levels) == 0:
            residual_gat_levels = [True for i in range(len(channels) - 1)]

        assert len(residual_gat_levels) == len(channels) - 2
        for i, l in enumerate(residual_gat_levels):
            assert l.__class__ == bool
        
        super(GazeGrapher, self).__init__()

        self.device = device

        self.gats = MarkovResidualSequential(output_gat_levels,
                                             residual_gat_levels,
                                             gat_conv).to(device)
        
        for i, args in enumerate([{k: gat_conv_args[k][i]
                                   for k in gat_conv_args
                                   if k != 'channels'}
                                  for i in range(1, len(gat_conv_args['channels']))],
                                 start=1):
            args['in_channels'] = channels[i - 1]
            args['out_channels'] = channels[i]
            args['device'] = self.device
            self.gats.add_conv('GAT_{}'.format(i - 1), **args)
            

    def forward(self, x):
        nodes, splits, *args = x
        graphs = self.gats((nodes, splits, *args))
        return graphs


class XYTGazeGrapher(GazeGrapher):
    def __init__(self,
                 gat_conv=None,
                 output_gat_levels=None,
                 residual_gat_levels=None,
                 gat_conv_args={'channels': [67, 134, 134, 134, 67],
                                'att_heads': 8,
                                'xy_heads_ratio': 0.2,
                                't_heads_ratio': 0.1,
                                'act_att': F.leaky_relu,
                                'act_gat': F.elu,
                                'norm': GraphNorm,
                                'dropout': 0.1,
                                'bias': True,
                                'w_init_f': lambda w: nn.init.xavier_uniform_(w)},
                 device='cpu'):
        super().__init__(CausalXYTGATConv,
                         output_gat_levels,
                         residual_gat_levels,
                         gat_conv_args,
                         device)

    def forward(self, x):
        nodes, splits, fixations = x
        xy_mats, t_mats = get_xyt_pair_matrices(fixations, splits)
        graphs = self.gats((nodes, splits, xy_mats, t_mats))
        return graphs