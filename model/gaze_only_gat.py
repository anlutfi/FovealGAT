import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn.norm import GraphNorm

from feature_extraction.extractors import GazePatchFeatureExtractor
from node_encodings.sequence import GazeSinCosPosEncoding
from model.gat_conv import CausalGATConv
from model.xyt_gat_conv import CausalXYTGATConv
from model.residual_sequential import MarkovResidualSequential
from model.readout import CausalReadout
from model.grapher import GazeGrapher, XYTGazeGrapher


class GazeOnlyGAT(nn.Module):
    graphers = {CausalXYTGATConv.__name__: XYTGazeGrapher,
                CausalGATConv.__name__: GazeGrapher}
    
    def __init__(self,
                 feat_extractor,
                 encoder,
                 gat_conv=CausalGATConv,
                 residual_sequential=MarkovResidualSequential,
                 gat_layers=8,
                 output_gat_levels=None,
                 residual_gat_levels=None,
                 gat_att_heads=8,
                 gat_conv_args={'channels': 256,
                                'act_att': F.leaky_relu,
                                'act_gat': F.elu,
                                'norm': GraphNorm,
                                'dropout': 0.1,
                                'bias': True,
                                'w_init_f': lambda w: nn.init.xavier_uniform_(w)},
                 readout=CausalReadout,
                 ro_hidden_features=[1024, 1024, 1024],
                 out_channels = 1,
                 ro_act=nn.GELU,
                 ro_pred_act=nn.Identity,
                 ro_dropout=0.0,
                 ro_norm=nn.GroupNorm,
                 ro_bias=True,
                 ro_w_init_f=lambda w: nn.init.kaiming_uniform_(w, nonlinearity='relu'),
                 device='cpu'):
        super(GazeOnlyGAT, self).__init__()
        self.device=device

        if feat_extractor is None:
            feat_extractor = GazePatchFeatureExtractor(state_dict_pth='../pretrained/pvig_ti_224_gelu.pt',
                                                       w_init_f=lambda w: nn.init.kaiming_uniform_(w, nonlinearity='relu'),
                                                       device=device)
        self.feat_ext = feat_extractor
        self.encoder = encoder

        node_dims = self.feat_ext.head[-3].in_channels + self.encoder.feature_count + 3 #x, y, t
        self.expand_dims = nn.Conv1d(node_dims, gat_conv_args['channels'], 1).to(self.device)
        nn.init.xavier_uniform_(self.expand_dims.weight)
        if self.expand_dims.bias is not None:
            nn.init.zeros_(self.expand_dims.bias)

        self.grapher = GazeOnlyGAT.graphers[gat_conv.__name__](gat_conv,
                                                               residual_sequential,
                                                               gat_layers,
                                                               output_gat_levels,
                                                               residual_gat_levels,
                                                               gat_att_heads,
                                                               gat_conv_args,
                                                               device=device)

        ro_in_channels = sum([c.node_update.out_channels
                              for i, c in enumerate(self.grapher.gats)
                              if i in self.grapher.gats.output_lvls])
        ro_channels = [ro_in_channels] + ro_hidden_features
        self.readout = readout(ro_channels,
                               out_channels,
                               ro_act,
                               ro_pred_act,
                               ro_dropout,
                               ro_norm,
                               ro_bias,
                               w_init_f=ro_w_init_f,
                               device=device)

    def forward(self, x):
        splits, fixations, patches, labels = x
        fixations = fixations.to(self.device)
        patches = patches.to(self.device)
        labels = labels.to(self.device)

        feats = self.feat_ext(patches)
        encodings = self.encoder(splits, adjust_dim=True).to(self.device)

        nodes = torch.cat([fixations, encodings, feats], 1).float()
        nodes = nodes.squeeze(-1).squeeze(-1).permute((1, 0))
        nodes = self.expand_dims(nodes)
        
        graphs = self.grapher((nodes, splits, fixations))
        predictions = self.readout((graphs, splits))
        return predictions