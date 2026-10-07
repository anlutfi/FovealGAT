import torch
from torch import nn


class MarkovResidualSequential(nn.Sequential):
    """Extension of nn.Sequential class to allow keeping residual tensors"""
    def __init__(self, output_levels, residual_levels, gat_conv, *args):
        """param:levels is an list of indices of layers which outputs are to be kept
        """
        super(MarkovResidualSequential, self).__init__(*args)
        self.output_lvls = output_levels
        self.residual_lvls = residual_levels
        self.gat_conv = gat_conv
                

    def add_conv(self,
                 name,
                 in_channels,
                 **kwargs):
        self.add_module(name,
                        self.gat_conv(in_channels,
                                      **kwargs))


    def forward(self, x):
        first_nodes, splits, *args = x
        nodes = first_nodes

        result = []
        for i, module in enumerate(self):
            residual = nodes
            nodes = module((nodes, splits, *args))
            if i > 0 and self.residual_lvls[i]:
                nodes = nodes + residual
            if i in self.output_lvls:
                result.append(nodes)
        return result


class DenseResidualSequential(nn.Sequential):
    """Extension of nn.Sequential class to allow keeping residual tensors"""
    def __init__(self, output_levels, residual_levels, gat_conv, *args):
        """param:levels is an list of indices of layers which outputs are to be kept
        """
        super(MarkovResidualSequential, self).__init__(*args)
        self.output_lvls = output_levels
        self.residual_lvls = residual_levels
        self.gat_conv = gat_conv
                

    def add_conv(self,
                    name,
                    in_channels,
                    **kwargs):
        self.add_module(name,
                        self.gat_conv(in_channels,
                                        **kwargs))


    def forward(self, x):
        first_nodes, splits, *args = x
        nodes = first_nodes
        residual = first_nodes

        result = []
        for i, module in enumerate(self):
            nodes = module((nodes, splits, *args))
            if i > 0 and self.residual_lvls[i]:
                nodes = nodes + residual
            if i in self.output_lvls:
                result.append(nodes)
        return result