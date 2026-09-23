import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn.norm import GraphNorm

from feature_extraction.extractors import GazePatchFeatureExtractor
from node_encodings.sequence import GazeSinCosPosEncoding
from model.gat_conv import CausalGATConv
from model.readout import CausalReadout
from model.grapher import GazeGrapher


class GazeOnlyGAT(nn.Module):
    def __init__(self,
                 feat_extractor=GazePatchFeatureExtractor(w_init_f=lambda w: nn.init.kaiming_uniform_(w, nonlinearity='relu')),
                 encoder=GazeSinCosPosEncoding(386, 16), #for reflacx
                 readout_gat_levels=None,
                 gat_conv=CausalGATConv,
                 readout=CausalReadout,
                 gat_channels=[67, 134, 134, 134, 67],
                 att_heads=8,
                 act_att=F.leaky_relu,
                 act_gat=F.elu,
                 gat_norm=GraphNorm,
                 gat_dropout=[0.0, 0.1, 0.1, 0.1, 0.1],
                 gat_bias=True,
                 ro_hidden_features=[1024, 1024, 1024],
                 out_channels = 1,
                 ro_act=nn.GELU,
                 ro_pred_act=nn.Identity,
                 ro_dropout=0.0,
                 ro_norm=nn.BatchNorm1d,
                 ro_bias=True,
                 gat_w_init_f=lambda w: nn.init.xavier_uniform_(w),
                 ro_w_init_f=lambda w: nn.init.kaiming_uniform_(w, nonlinearity='relu'),
                 device='cpu'):
        super(GazeOnlyGAT, self).__init__()
        self.device=device

        self.feat_ext = feat_extractor.to(device)
        self.encoder = encoder

        if readout_gat_levels is None or len(readout_gat_levels) == 0:
            readout_gat_levels = list(range(len(gat_channels)))
        self.grapher = GazeGrapher(gat_conv,
                                   readout_gat_levels,
                                   gat_channels,
                                   att_heads,
                                   act_att,
                                   act_gat,
                                   gat_norm,
                                   gat_dropout,
                                   gat_bias,
                                   w_init_f=gat_w_init_f,
                                   device=device)

        ro_in_channels = sum([gat_channels[i] for i in readout_gat_levels])
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
        
        graphs = self.grapher((nodes, splits))
        predictions = self.readout((graphs, splits))
        return predictions