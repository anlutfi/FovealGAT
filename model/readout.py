import torch
from torch import nn
from collections import OrderedDict

from model.tools import adjust_args, init_weights

class CausalReadout(nn.Module):
    def __init__(self,
                 channels=[335, 512, 256, 128, 64],
                 out_channels = 1,
                 act=nn.GELU,
                 pred_act=nn.Identity,
                 dropout=0.0,
                 norm=nn.GroupNorm,
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

        (channels,
         act,
         dropout,
         norm,
         bias) = adjust_args([channels,
                              act,
                              dropout,
                              norm,
                              bias])

        self.mlp = nn.Sequential()
        
        for i in range(1, len(channels)):
            if i == 1:
                self.mlp.add_module('norm_input', norm[i - 1](num_groups=1, num_channels=channels[0], device=device))
            self.mlp.add_module('fc{}'.format(i - 1), nn.Conv1d(channels[i - 1],
                                                                channels[i],
                                                                1,
                                                                bias=bias[i - 1],
                                                                device=device))
            self.mlp.add_module('norm{}'.format(i), norm[i - 1](num_groups=1, num_channels=channels[i], device=device))
            self.mlp.add_module('act{}'.format(i), act[i - 1]())
            self.mlp.add_module('drop{}'.format(i), nn.Dropout(dropout[i - 1]))
            if i == len(channels) - 1:
                self.mlp.add_module('prediction', nn.Conv1d(channels[i],
                                                            out_channels,
                                                            1,
                                                            bias=bias[i],
                                                            device=device))

        self.mlp.apply(lambda m: init_weights(m, w_init_f))
        

    def forward(self, x):
        graphs, splits = x
        splits = [sum(splits[0:i + 1]) - 1 for i in range(len(splits))]
        sinks = torch.cat([graph[:, torch.tensor(splits, device=self.device)] for graph in graphs]).to(self.device)
        sinks = sinks.unsqueeze(0)
        predictions = self.mlp(sinks).squeeze(0).squeeze(0)
        return self.act(predictions)


        

