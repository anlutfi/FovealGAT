from torch import nn


class ResidualSequential(nn.Sequential):
    """Extension of nn.Sequential class to allow keeping residual tensors"""
    def __init__(self, levels, *args):
        """param:levels is an list of indices of layers which outputs are to be kept
        """
        super(ResidualSequential, self).__init__(*args)
        self.levels = levels

    def forward(self, x):
        nodes, splits = x

        result = [nodes] if 0 in self.levels else []
        for i, module in enumerate(self, 1):
            nodes = module((nodes, splits))
            if i in self.levels:
                result.append(nodes)
        return result