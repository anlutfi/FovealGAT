from torch import nn
from torch_geometric.nn.norm import GraphNorm
import torch.nn.functional as F

from model.tools import adjust_args
from model.GATConv import CausalGATConv
from model.residual_sequential import ResidualSequential


class GazeGrapher(nn.Module):
    def __init__(self,
                 gat_conv=CausalGATConv,
                 readout_gat_levels=None,
                 channels=[67, 134, 134, 134, 67],
                 att_heads=8,
                 act_att=F.leaky_relu,
                 act_gat=F.elu,
                 norm=GraphNorm,
                 dropout=[0.0, 0.1, 0.1, 0.1, 0.1],
                 bias=True,
                 w_init_f=lambda w: nn.init.xavier_uniform_(w),
                 device='cpu'):

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

        if readout_gat_levels is None or len(readout_gat_levels) == 0:
            readout_gat_levels = list(range(len(channels)))

        assert len(readout_gat_levels) <= len(channels)
        for l in readout_gat_levels:
            assert l < len(channels)
        
        super(GazeGrapher, self).__init__()

        self.device = device

        self.gats = ResidualSequential(readout_gat_levels).to(device)
        for i, (chn, heads, act_a, act_g, n, drop, b) in enumerate(zip(channels[1:],
                                                                       att_heads[1:],
                                                                       act_att[1:],
                                                                       act_gat[1:],
                                                                       norm[1:],
                                                                       dropout[1:],
                                                                       bias[1:]),
                                                                   start=1):
            self.gats.add_module('GAT_{}'.format(i - 1),
                                 gat_conv(channels[i - 1],
                                          chn,
                                          heads,
                                          act_a,
                                          act_g,
                                          n,
                                          drop,
                                          b,
                                          w_init_f=w_init_f,
                                          device=device))
            

    def forward(self, x):
        nodes, splits = x
        graphs = self.gats((nodes, splits))
        return graphs
