import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn.norm import GraphNorm

from math import floor

from model.tools import init_weights

class XYTGATConv(nn.Module):
    @staticmethod
    def adj_mat(nodes):
        nodes = nodes.unsqueeze(-1)
        source = torch.tile(nodes, (1, 1, nodes.shape[1]))
        dest = source.transpose(1, 2)
        return torch.cat((source, dest), 0)

    def __init__(self,
                 in_channels,
                 out_channels,
                 att_heads,
                 xy_heads_ratio=0.2,
                 t_heads_ratio=0.1,
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
        super(XYTGATConv, self).__init__()
        self.device = device
        self.act_att = act_att
        self.act_gat = act_gat

        self.att_scores = nn.Conv2d(in_channels * 2,
                                    att_heads,
                                    1,
                                    device=device,
                                    bias=bias)

        xy_heads = max(1, floor((att_heads * xy_heads_ratio) + 0.5))#TODO fix this to ceil
        self.xy_scores = nn.Conv2d(4,# x1,y1,x2,y2
                                   xy_heads,
                                   1,
                                   device=device,
                                   bias=bias)

        t_heads = max(1, floor((att_heads * t_heads_ratio) + 0.5))
        self.t_scores = nn.Conv2d(2,# t1,t2
                                  t_heads,
                                  1,
                                  device=device,
                                  bias=bias)

        self.node_update = nn.Conv1d((att_heads + xy_heads + t_heads) * in_channels,
                                     out_channels,
                                     1,
                                     device=device,
                                     bias=bias)

        self.norm = norm(in_channels).to(self.device)
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        self.att_scores.apply(lambda m: init_weights(m, w_init_f))
        self.xy_scores.apply(lambda m: init_weights(m, w_init_f))
        self.t_scores.apply(lambda m: init_weights(m, w_init_f))
        self.node_update.apply(lambda m: init_weights(m, w_init_f))

        nn.init.uniform_(self.xy_scores.weight, a=-0.01, b=0.01)
        nn.init.uniform_(self.t_scores.weight, a=-0.01, b=0.01)

        # Zero out their biases so there is no static offset at launch
        if self.xy_scores.bias is not None:
            nn.init.zeros_(self.xy_scores.bias)
        if self.t_scores.bias is not None:
            nn.init.zeros_(self.t_scores.bias)
                

    def forward(self, x):
        pass
        


class CausalXYTGATConv(XYTGATConv):
    def __init__(self,
                 in_channels,
                 out_channels,
                 att_heads,
                 xy_heads_ratio=0.2,
                 t_heads_ratio=0.1,
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
                         act_att,
                         act_gat,
                         norm,
                         dropout,
                         bias,
                         w_init_f,
                         device)
    """A GATConv layer for a causal graph,
    where each node sees only itself and their predecessors
    """
    def forward(self, x):
        nodes, splits, xy_mats, t_mats = x
        # normalize graph
        batch_idx = torch.cat([torch.ones(s, dtype=torch.int) * i
                               for i, s in enumerate(splits)]).to(self.device)
        nodes = torch.transpose(nodes, 1, 0)
        nodes = self.dropout(self.norm(nodes, batch_idx.to(torch.long)))
        nodes = torch.transpose(nodes, 1, 0)
        
        # generate adjacency matrix for whole batch
        subnodes = torch.split(nodes, splits, dim=1)
        adjs = [XYTGATConv.adj_mat(ns) for ns in subnodes]

        # generate raw semantic scores
        sem_raw_scores = [self.act_att(self.att_scores(adj)) for adj in adjs]
        
        # generate raw XY scores
        xy_raw_scores = [self.act_att(self.xy_scores(xy_mat)) for xy_mat in xy_mats]

        # generate raw T scores
        t_raw_scores = [self.act_att(self.t_scores(t_mat)) for t_mat in t_mats]
        
        rawscores = [torch.cat((sem, xy, t), dim=0)
                     for sem, xy, t in zip(sem_raw_scores,
                                           xy_raw_scores,
                                           t_raw_scores)]        

        # set upper triangle to zero, so a node doesn't see future nodes
        adj_mask = [torch.tril(torch.ones(rs.shape,
                                dtype=torch.bool,
                                device=self.device))
                    for rs in rawscores]
        masked_scores = [torch.where(am,
                                     rs,
                                     torch.tensor(float('-inf'),
                                     device=self.device))
                         for am, rs in zip(adj_mask, rawscores)]

        # generate final att weights
        att_ws = [F.softmax(ms, dim=-1) for ms in masked_scores]

        # pass messages and generate output nodes
        gatconv = [torch.matmul(sn, s) for sn, s in zip(subnodes, att_ws)]
        gatconv = torch.cat(gatconv, -1)
        gatconv = gatconv.reshape((gatconv.shape[0] * gatconv.shape[1], gatconv.shape[2]))
        return self.act_gat(self.node_update(gatconv))


class CausalEuclideanTGATConv(CausalXYTGATConv):
    """A GATConv layer for a causal graph,
        where each node sees only itself and their predecessors
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 att_heads,
                 xy_heads_ratio=0.2,
                 t_heads_ratio=0.1,
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


class CausalSummaryXYTGATConv(XYTGATConv):
    def __init__(self,
                 in_channels,
                 out_channels,
                 att_heads,
                 xy_heads_ratio=0.2,
                 t_heads_ratio=0.1,
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
                         act_att,
                         act_gat,
                         norm,
                         dropout,
                         bias,
                         w_init_f,
                         device)

        self.pad = nn.ZeroPad3d((0, 1, 0, 1, 0, 0)) #zero pad right and bottom
    """A GATConv layer for a causal graph,
    where each node sees only itself and their predecessors
    """
    def forward(self, x):
        nodes, splits, xy_mats, t_mats = x
        # normalize graph
        if nodes.shape[-1] != sum(splits) + len(splits): #there is no summary node yet
            summary_node = False
            batch_idx = torch.cat([torch.ones(s, dtype=torch.int) * i
                                   for i, s in enumerate(splits)]).to(self.device)
        else:
            summary_node = True
            batch_idx = torch.cat([torch.ones(s, dtype=torch.int) * i
                                   for i, s in enumerate([s + 1 for s in splits])]).to(self.device)
                        
        nodes = torch.transpose(nodes, 1, 0)
        nodes = self.dropout(self.norm(nodes, batch_idx.to(torch.long)))
        nodes = torch.transpose(nodes, 1, 0)

        if not summary_node: 
            subnodes = torch.split(nodes, splits, dim=1)
            subnodes = [torch.cat([sn, torch.zeros(sn.shape[0], 1).to(self.device)], dim=-1)
                        for sn in subnodes]
        else:
            subnodes = torch.split(nodes, [s + 1 for s in splits], dim=1)
        
        # generate raw XY scores
        xy_raw_scores = [self.act_att(self.xy_scores(xy_mat)) for xy_mat in xy_mats]
        # generate raw T scores
        t_raw_scores = [self.act_att(self.t_scores(t_mat)) for t_mat in t_mats]

        xyt_raw_scores = [torch.cat((xy, t), dim=0)
                          for xy, t in zip(xy_raw_scores,
                                           t_raw_scores)]
        
        #zero score upper triangle
        xyt_adj_mask = [torch.tril(torch.ones(rs.shape,
                                              dtype=torch.bool,
                                              device=self.device))
                        for rs in xyt_raw_scores]
        xyt_masked_scores = [torch.where(am,
                                         rs,
                                         torch.tensor(float('-inf'),
                                         device=self.device))
                             for am, rs in zip(xyt_adj_mask, xyt_raw_scores)]
        
        xyt_att_ws = [F.softmax(ms, dim=-1) for ms in xyt_masked_scores]

        # generate adjacency matrix for whole batch
        adjs = [XYTGATConv.adj_mat(ns) for ns in subnodes]

        # generate raw semantic scores
        sem_raw_scores = [self.act_att(self.att_scores(adj)) for adj in adjs]
        
        # set upper triangle to zero, so a node doesn't see future nodes
        adj_mask = [torch.tril(torch.ones(rs.shape,
                                          dtype=torch.bool,
                                          device=self.device))
                    for rs in sem_raw_scores]
        sem_masked_scores = [torch.where(am,
                                         rs,
                                         torch.tensor(float('-inf'),
                                         device=self.device))
                            for am, rs in zip(adj_mask, sem_raw_scores)]
        sem_att_ws = [F.softmax(ms, dim=-1) for ms in sem_masked_scores]
        xyt_att_ws = [self.pad(ws) for ws in xyt_att_ws]
        
        att_ws = [torch.cat((xyt, sem), dim=0)
                  for xyt, sem in zip(xyt_att_ws, sem_att_ws)]

        # pass messages and generate output nodes
        gatconv = [torch.matmul(sn, s) for sn, s in zip(subnodes, att_ws)]
        gatconv = torch.cat(gatconv, -1)
        gatconv = gatconv.reshape((gatconv.shape[0] * gatconv.shape[1], gatconv.shape[2]))
        return self.act_gat(self.node_update(gatconv))
    

class CausalEuclideanSummaryTGATConv(CausalSummaryXYTGATConv):
    """A GATConv layer for a causal graph,
        where each node sees only itself and their predecessors
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 att_heads,
                 xy_heads_ratio=0.2,
                 t_heads_ratio=0.1,
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
   