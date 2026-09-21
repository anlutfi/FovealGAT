import torch
from torch import nn
from collections import OrderedDict

from model.tools import adjust_args, init_weights

class CausalReadout(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_features=[1024, 1024, 1024],
                 out_channels = 1,
                 hidden_act=nn.GELU,
                 pred_act=nn.Identity,
                 dropout=0.0,
                 norm=nn.BatchNorm1d,
                 bias=True,
                 w_init_f=lambda w: nn.init.kaiming_uniform_(w, nonlinearity='relu'),
                 device='cpu'):
        """Learns a causal graph's readout.
        param:hidden_features is either a single value or a list of values,
        determining the number of hidden layers to be created

        param:act, param:dropout, and param:norm are either lists OF THE SAME LEN as hidden_features,
        or single values to be repeated at each hidden layer
        """
        super(CausalReadout, self).__init__()

        self.device = device
        self.act = pred_act()

        (hidden_features,
         hidden_act,
         dropout,
         norm,
         bias) = adjust_args([hidden_features,
                              hidden_act,
                              dropout,
                              norm,
                              bias])

        self.mlp = nn.Sequential(OrderedDict([('fc1', nn.Conv1d(in_channels,
                                                                hidden_features[0],
                                                                1,
                                                                bias=bias[0],
                                                                device=device))]))

        for i, (feats, a, drop, n, b) in enumerate(zip(hidden_features,
                                                       hidden_act,
                                                       dropout,
                                                       norm,
                                                       bias),
                                                start=1):
            self.mlp.add_module('norm{}'.format(i), n(feats, device=device))
            self.mlp.add_module('act{}'.format(i), a())
            self.mlp.add_module('drop{}'.format(i), nn.Dropout(drop))
            if i < len(hidden_features):
                self.mlp.add_module('fc{}'.format(i + 1), nn.Conv1d(feats,
                                                                    feats,
                                                                    1,
                                                                    bias=b,
                                                                    device=device))
            else:
                self.mlp.add_module('prediction', nn.Conv1d(feats,
                                                            out_channels,
                                                            1,
                                                            bias=b,
                                                            device=device))

        self.mlp.apply(lambda m: init_weights(m, w_init_f))
        

    def forward(self, x):
        graphs, splits = x
        splits = [sum(splits[0:i + 1]) - 1 for i in range(len(splits))]
        sinks = torch.cat([graph[:, splits] for graph in graphs]).to(self.device)
        sinks = sinks.unsqueeze(0)
        predictions = self.mlp(sinks).squeeze(0).squeeze(0)
        return self.act(predictions)


        

