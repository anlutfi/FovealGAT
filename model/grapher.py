from torch import nn
from torch_geometric.nn.norm import GraphNorm
import torch.nn.functional as F

from model.tools import adjust_args
from model.gat_conv import CausalGATConv
from model.residual_sequential import ResidualSequential


class GazeGrapher(nn.Module):
    def __init__(self,
                 gat_conv=CausalGATConv,
                 output_gat_levels=None,
                 residual_gat_levels=None,
                 channels=[67, 134, 134, 134, 67],
                 att_heads=8,
                 act_att=F.leaky_relu,
                 act_gat=F.elu,
                 norm=GraphNorm,
                 dropout=0.1,
                 bias=True,
                 w_init_f=lambda w: nn.init.xavier_uniform_(w),
                 device='cpu'):
        """A module that applies a series of graph convolutions
        param:gat_conv which type of convolution to apply among those in model.GATConv
        param:readout_gat_levels a list of indices of the layers' outputs to be returned by forward
        param:channels a list of input channels, each pair will correspond to a convolution layer. ALL PARAMS from att_heads to bias need to be either a single value or a list of values OF THE SAME LENGTH as channels
        param:act_att activation function for the attention scores of each convolution layer. if a single value, it will be replicated for all
        param:act_gat activation function for the node update of each convolution layer. if a single value, it will be replicated for all
        """
        (channels,
         att_heads,
         act_att,
         act_gat,
         norm,
         dropout,
         bias) = adjust_args([channels,
                              att_heads,
                              act_att,
                              act_gat,
                              norm,
                              dropout,
                              bias])

        if output_gat_levels is None or len(output_gat_levels) == 0:
            output_gat_levels = list(range(len(channels)))

        assert len(output_gat_levels) <= len(channels)
        for l in output_gat_levels:
            assert l < len(channels)

        if residual_gat_levels is None or len(residual_gat_levels) == 0:
            residual_gat_levels = [True for i in range(len(channels) - 2)]

        assert len(residual_gat_levels) == len(channels) - 2
        for i, l in enumerate(residual_gat_levels):
            assert l.__class__ == bool
        
        super(GazeGrapher, self).__init__()

        self.device = device

        self.gats = ResidualSequential(output_gat_levels,
                                       residual_gat_levels,
                                       gat_conv).to(device)
        for i, (chn, heads, act_a, act_g, n, drop, b) in enumerate(zip(channels[1:],
                                                                       att_heads[1:],
                                                                       act_att[1:],
                                                                       act_gat[1:],
                                                                       norm[1:],
                                                                       dropout[1:],
                                                                       bias[1:]),
                                                                   start=1):
            self.gats.add_conv('GAT_{}'.format(i - 1),
                               channels[i - 1],
                               chn,
                               heads,
                               act_a,
                               act_g,
                               n,
                               drop,
                               b,
                               w_init_f,
                               device)
            

    def forward(self, x):
        nodes, splits = x
        graphs = self.gats((nodes, splits))
        return graphs
