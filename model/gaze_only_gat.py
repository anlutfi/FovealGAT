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
                 feat_extractor=None,
                 encoder=GazeSinCosPosEncoding(386, 16), #for reflacx
                 output_gat_levels=None,
                 residual_gat_levels=None,
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

        if feat_extractor is None:
            feat_extractor = GazePatchFeatureExtractor(state_dict_pth='../pretrained/pvig_ti_224_gelu.pt',
                                                       w_init_f=lambda w: nn.init.kaiming_uniform_(w, nonlinearity='relu'),
                                                       device=device)
        self.feat_ext = feat_extractor
        self.encoder = encoder

        self.grapher = GazeGrapher(gat_conv,
                                   output_gat_levels,
                                   residual_gat_levels,
                                   gat_channels,
                                   att_heads,
                                   act_att,
                                   act_gat,
                                   gat_norm,
                                   gat_dropout,
                                   gat_bias,
                                   w_init_f=gat_w_init_f,
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
        
        graphs = self.grapher((nodes, splits))
        predictions = self.readout((graphs, splits))
        return predictions