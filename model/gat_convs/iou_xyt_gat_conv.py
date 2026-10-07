import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn.norm import GraphNorm

from math import ceil

from model.tools import init_weights
from model.gat_convs.xyt_gat_conv import XYTGATConv

class IOU_XYTGATConv(XYTGATConv):
    def __init__(self,
                 in_channels,
                 out_channels,
                 att_heads,
                 xy_heads_ratio=0.2,
                 t_heads_ratio=0.1,
                 iou_heads_ratio=0.2,
                 act_att=F.leaky_relu,
                 act_gat=F.elu,
                 norm=GraphNorm,
                 dropout=0.1,
                 bias=True,
                 w_init_f=lambda w: nn.init.xavier_uniform_(w),
                 device='cpu'):
        """A Graph Attention Convolution Layer
        param:act_att activation function for the attention scores
        param:act_gat activation function for the node update
        """
        super().__init__(in_channels,
                         out_channels,
                         att_heads,
                         xy_heads_ratio,
                         t_heads_ratio,
                         act_att,
                         act_gat,
                         norm,
                         dropout,
                         bias,
                         w_init_f,
                         device)
        iou_heads = max(1, ceil(att_heads * iou_heads_ratio))
        self.iou_scores = nn.Conv2d(1,# ious
                                    iou_heads,
                                    1,
                                    device=device,
                                    bias=bias)
        xy_heads = self.xy_scores.out_channels
        t_heads = self.t_scores.out_channels

        self.node_update = nn.Conv1d((att_heads + xy_heads + t_heads + iou_heads) * in_channels,
                                      out_channels,
                                      1,
                                      device=device,
                                      bias=bias)

        self.iou_scores.apply(lambda m: init_weights(m, w_init_f))
        self.node_update.apply(lambda m: init_weights(m, w_init_f))
               

    def forward(self, x):
        nodes, splits, spatial_mats, t_mats, iou_mats = x
        subnodes, adjs = self.get_subnodes_adjs(nodes, splits)

        sem_raw_scores = self.get_raw_scores(self.att_scores, adjs)
        spatial_raw_scores = self.get_raw_scores(self.xy_scores, spatial_mats)
        t_raw_scores = self.get_raw_scores(self.t_scores, t_mats)
        iou_raw_scores = self.get_raw_scores(self.iou_scores, iou_mats)
        
        raw_scores = [torch.cat((sem, xy, t, iou), dim=0)
                      for sem, xy, t, iou in zip(sem_raw_scores,
                                                 spatial_raw_scores,
                                                 t_raw_scores,
                                                 iou_raw_scores)]        

        scores = (self.get_causal_scores(raw_scores)
                  if self.causal
                  else raw_scores)

        # generate final att weights
        att_ws = [F.softmax(ms, dim=-1) for ms in scores]

        return self.convolve(subnodes, att_ws)



class CausalIOU_XYTGATConv(IOU_XYTGATConv):
    def __init__(self,
                 in_channels,
                 out_channels,
                 att_heads,
                 xy_heads_ratio=0.2,
                 t_heads_ratio=0.1,
                 iou_heads_ratio=0.2,
                 act_att=F.leaky_relu,
                 act_gat=F.elu,
                 norm=GraphNorm,
                 dropout=0.1,
                 bias=True,
                 w_init_f=lambda w: nn.init.xavier_uniform_(w),
                 device='cpu'):
        super().__init__(in_channels,
                         out_channels,
                         att_heads,
                         xy_heads_ratio,
                         t_heads_ratio,
                         iou_heads_ratio,
                         act_att,
                         act_gat,
                         norm,
                         dropout,
                         bias,
                         w_init_f,
                         device)
        self.causal = True
    """A GATConv layer for a causal graph,
    where each node sees only itself and their predecessors
    """
    

class CausalIOU_EuclideanTGATConv(CausalIOU_XYTGATConv):
    """A GATConv layer for a causal graph,
        where each node sees only itself and their predecessors
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 att_heads,
                 xy_heads_ratio=0.2,
                 t_heads_ratio=0.1,
                 iou_heads_ratio=0.2,
                 act_att=F.leaky_relu,
                 act_gat=F.elu,
                 norm=GraphNorm,
                 dropout=0.1,
                 bias=True,
                 w_init_f=lambda w: nn.init.xavier_uniform_(w),
                 device='cpu'):
        super().__init__(in_channels,
                         out_channels,
                         att_heads,
                         xy_heads_ratio,
                         t_heads_ratio,
                         iou_heads_ratio,
                         act_att,
                         act_gat,
                         norm,
                         dropout,
                         bias,
                         w_init_f,
                         device)

        xy_heads = self.xy_scores.out_channels
        self.xy_scores = nn.Conv2d(7,# x1, y1, x2, y2, deltax, deltay, euc_dist
                                   xy_heads,
                                   1,
                                   device=device,
                                   bias=bias)
        self.xy_scores.apply(lambda m: init_weights(m, w_init_f))
                