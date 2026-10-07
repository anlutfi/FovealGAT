import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn.norm import GraphNorm

from model.tools import init_weights

class GATConv(nn.Module):
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
        super(GATConv, self).__init__()
        self.device = device
        self.act_att = act_att
        self.act_gat = act_gat

        self.att_scores = nn.Conv2d(in_channels * 2,
                                    att_heads,
                                    1,
                                    device=device,
                                    bias=bias)

        self.node_update = nn.Conv1d(att_heads * in_channels,
                                     out_channels,
                                     1,
                                     device=device,
                                     bias=bias)

        self.norm = norm(in_channels).to(self.device)
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        self.att_scores.apply(lambda m: init_weights(m, w_init_f))
        self.node_update.apply(lambda m: init_weights(m, w_init_f))
                

    def forward(self, x):
        pass
        


class CausalGATConv(GATConv):
    def __init__(self,
                 in_channels,
                 out_channels,
                 att_heads,
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
        nodes, splits, *_ = x
        # normalize graph
        batch_idx = torch.cat([torch.ones(s, dtype=torch.int) * i
                               for i, s in enumerate(splits)]).to(self.device)
        nodes = torch.transpose(nodes, 1, 0)
        nodes = self.norm(nodes, batch_idx.to(torch.long))
        nodes = torch.transpose(nodes, 1, 0)
        
        # generate adjacency matrix for whole batch
        subnodes = torch.split(nodes, splits, dim=1)
        adjs = [GATConv.adj_mat(ns) for ns in subnodes]

        # generate raw attention scores
        rawscores = [self.act_att(self.att_scores(adj)) for adj in adjs]

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
        return self.dropout(self.act_gat(self.node_update(gatconv)))
        

class FullGATConv(GATConv):
    def __init__(self,
                     in_channels,
                     out_channels,
                     att_heads,
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
                             act_att,
                             act_gat,
                             norm,
                             dropout,
                             bias,
                             w_init_f,
                             device)
    """A GATConv layer for a complete graph
    """
    def forward(self, x):
        nodes, splits, *_ = x
        # normalize graph
        batch_idx = torch.cat([torch.ones(s, dtype=torch.int) * i
                               for i, s in enumerate(splits)]).to(self.device)
        nodes = torch.transpose(nodes, 1, 0)
        nodes = self.norm(nodes, batch_idx)
        nodes = torch.transpose(nodes, 1, 0)
                
        # generate adjacency matrix for whole batch
        subnodes = torch.split(nodes, splits, dim=1)
        adjs = [GATConv.adj_mat(ns) for ns in subnodes]

        # generate raw attention scores
        rawscores = [self.act_att(self.att_scores(adj)) for adj in adjs]

        # generate final att weights
        att_ws = [F.softmax(rs, dim=-1) for rs in rawscores]

        # pass messages and generate output nodes
        gatconv = [torch.matmul(sn, s) for sn, s in zip(subnodes, att_ws)]
        gatconv = torch.cat(gatconv, -1)
        gatconv = gatconv.reshape((gatconv.shape[0] * gatconv.shape[1], gatconv.shape[2]))
        return self.dropout(self.act_gat(self.node_update(gatconv)))
        