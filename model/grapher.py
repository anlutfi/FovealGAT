import torch
from torch import nn
from torch_geometric.nn.norm import GraphNorm
import torch.nn.functional as F
import copy

from model.tools import adjust_arg_to_len
from model.gat_convs.gat_conv import CausalGATConv
from model.gat_convs.xyt_gat_conv import CausalXYTGATConv, CausalEuclideanTGATConv, CausalSummaryXYTGATConv, CausalEuclideanSummaryTGATConv
from model.gat_convs.iou_xyt_gat_conv import CausalIOU_XYTGATConv, CausalIOU_EuclideanTGATConv, CausalSummaryIOU_XYTGATConv, CausalEuclideanSummaryIOU_TGATConv
from model.residual_sequential import MarkovResidualSequential
from gaze_datasets.dataset_tools import all_pairs, get_xyt_pair_matrices, get_euclidean_encodings, get_bbox_iou_encodings


class GazeGrapher(nn.Module):
    def __init__(self,
                 gat_conv=CausalGATConv,
                 residual_sequential=MarkovResidualSequential,
                 layers=8,
                 output_gat_levels=None,
                 residual_gat_levels=None,
                 att_heads=8,
                 gat_conv_args={'channels': 256,
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
        
        att_heads = adjust_arg_to_len(layers, att_heads)
        gat_conv_args = copy.deepcopy(gat_conv_args)
        channels = gat_conv_args.pop('channels')
        gat_conv_args['in_channels'] = channels
        gat_conv_args['out_channels'] = channels
        gat_conv_args['device'] = device
                
        
        if output_gat_levels is None or len(output_gat_levels) == 0:
            output_gat_levels = list(range(layers))

        assert len(output_gat_levels) <= layers
        for l in output_gat_levels:
            assert l < layers

        if residual_gat_levels is None or len(residual_gat_levels) == 0:
            residual_gat_levels = [True if i > 0 else False for i in range(layers)]

        else:
            assert len(residual_gat_levels) == layers
            for i, l in enumerate(residual_gat_levels):
                assert l.__class__ == bool
            residual_gat_levels[0] = False
        
        super(GazeGrapher, self).__init__()

        self.device = device

        self.gats = residual_sequential(output_gat_levels,
                                        residual_gat_levels,
                                        gat_conv).to(device)
        
        for i in range(layers):
            gat_conv_args['att_heads'] = att_heads[i]
            self.gats.add_conv('GAT_{}'.format(i), **gat_conv_args)
                    

    def forward(self, x):
        nodes, splits, *args = x
        graphs = self.gats((nodes, splits, *args))
        return graphs


class XYTGazeGrapher(GazeGrapher):
    gat_convs = [CausalXYTGATConv.__name__,
                 CausalSummaryXYTGATConv.__name__]
    def __init__(self,
                 gat_conv=CausalXYTGATConv,
                 residual_sequential=MarkovResidualSequential,
                 layers=8,
                 output_gat_levels=None,
                 residual_gat_levels=None,
                 att_heads=8,
                 gat_conv_args={'channels': 256,
                                'xy_heads_ratio': 0.2,
                                't_heads_ratio': 0.1,
                                'act_att': F.leaky_relu,
                                'act_gat': F.elu,
                                'norm': GraphNorm,
                                'dropout': 0.1,
                                'bias': True,
                                'w_init_f': lambda w: nn.init.xavier_uniform_(w)},
                 device='cpu'):
        assert gat_conv.__name__ in XYTGazeGrapher.gat_convs
        super().__init__(gat_conv,
                         residual_sequential,
                         layers,
                         output_gat_levels,
                         residual_gat_levels,
                         att_heads,
                         gat_conv_args,
                         device)

    def forward(self, x):
        nodes, splits, fixations = x
        xy_mats, t_mats = get_xyt_pair_matrices(fixations, splits)
        graphs = self.gats((nodes, splits, xy_mats, t_mats))
        return graphs


class EuclideanTGazeGrapher(GazeGrapher):
    gat_convs = [CausalEuclideanTGATConv.__name__,
                 CausalEuclideanSummaryTGATConv.__name__]
    def __init__(self,
                 gat_conv=CausalEuclideanTGATConv,
                 residual_sequential=MarkovResidualSequential,
                 layers=8,
                 output_gat_levels=None,
                 residual_gat_levels=None,
                 att_heads=8,
                 gat_conv_args={'channels': 256,
                                'xy_heads_ratio': 0.2,
                                't_heads_ratio': 0.1,
                                'act_att': F.leaky_relu,
                                'act_gat': F.elu,
                                'norm': GraphNorm,
                                'dropout': 0.1,
                                'bias': True,
                                'w_init_f': lambda w: nn.init.xavier_uniform_(w)},
                 device='cpu'):
        assert gat_conv.__name__ in EuclideanTGazeGrapher.gat_convs
        super().__init__(gat_conv,
                         residual_sequential,
                         layers,
                         output_gat_levels,
                         residual_gat_levels,
                         att_heads,
                         gat_conv_args,
                         device)

    def forward(self, x):
        nodes, splits, fixations = x
        xy_mats, t_mats = get_xyt_pair_matrices(fixations, splits)
        xy_mats = [get_euclidean_encodings(xym) for xym in xy_mats]
        graphs = self.gats((nodes, splits, xy_mats, t_mats))
        return graphs


class IOU_XYTGazeGrapher(GazeGrapher):
    gat_convs = [CausalIOU_XYTGATConv.__name__,
                 CausalSummaryIOU_XYTGATConv.__name__]
    def __init__(self,
                 gat_conv=CausalIOU_XYTGATConv,
                 residual_sequential=MarkovResidualSequential,
                 layers=8,
                 output_gat_levels=None,
                 residual_gat_levels=None,
                 att_heads=8,
                 gat_conv_args={'channels': 256,
                                'xy_heads_ratio': 0.2,
                                't_heads_ratio': 0.1,
                                'iou_heads_ratio': 0.2,
                                'act_att': F.leaky_relu,
                                'act_gat': F.elu,
                                'norm': GraphNorm,
                                'dropout': 0.1,
                                'bias': True,
                                'w_init_f': lambda w: nn.init.xavier_uniform_(w)},
                 device='cpu'):
        assert gat_conv.__name__ in IOU_XYTGazeGrapher.gat_convs
        super().__init__(gat_conv,
                         residual_sequential,
                         layers,
                         output_gat_levels,
                         residual_gat_levels,
                         att_heads,
                         gat_conv_args,
                         device)

    def forward(self, x):
        nodes, splits, fixations, bboxes = x
        xy_mats, t_mats = get_xyt_pair_matrices(fixations, splits)
        iou_mats = get_bbox_iou_encodings(bboxes, splits)
        graphs = self.gats((nodes, splits, xy_mats, t_mats, iou_mats))
        return graphs


class IOU_EuclideanTGazeGrapher(GazeGrapher):
    gat_convs = [CausalIOU_EuclideanTGATConv.__name__,
                 CausalEuclideanSummaryIOU_TGATConv.__name__]
    def __init__(self,
                 gat_conv=CausalIOU_EuclideanTGATConv,
                 residual_sequential=MarkovResidualSequential,
                 layers=8,
                 output_gat_levels=None,
                 residual_gat_levels=None,
                 att_heads=8,
                 gat_conv_args={'channels': 256,
                                'xy_heads_ratio': 0.2,
                                't_heads_ratio': 0.1,
                                'iou_heads_ratio': 0.2,
                                'act_att': F.leaky_relu,
                                'act_gat': F.elu,
                                'norm': GraphNorm,
                                'dropout': 0.1,
                                'bias': True,
                                'w_init_f': lambda w: nn.init.xavier_uniform_(w)},
                 device='cpu'):
        assert gat_conv.__name__ in IOU_EuclideanTGazeGrapher.gat_convs
        super().__init__(gat_conv,
                         residual_sequential,
                         layers,
                         output_gat_levels,
                         residual_gat_levels,
                         att_heads,
                         gat_conv_args,
                         device)

    def forward(self, x):
        nodes, splits, fixations, bboxes = x
        xy_mats, t_mats = get_xyt_pair_matrices(fixations, splits)
        xy_mats = [get_euclidean_encodings(xym) for xym in xy_mats]
        iou_mats = get_bbox_iou_encodings(bboxes, splits)
        graphs = self.gats((nodes, splits, xy_mats, t_mats, iou_mats))
        return graphs