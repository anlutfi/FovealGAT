from torch import nn
import torch
import os
from feature_extraction.pretrained import pyramid_feature_extractor
from model.tools import init_weights, adjust_args


class GazePatchFeatureExtractor(nn.Module):
    """extracts a feature vector from a gaze patch image
    """
    def __init__(self,
                 tail_feat_ext=pyramid_feature_extractor,
                 state_dict_pth='../pretrained/pvig_ti_224_gelu.pt',
                 channels=[48, 128, 128, 128, 128],
                 act=nn.GELU,
                 norm=nn.BatchNorm2d,
                 bias=True,
                 w_init_f=lambda w: nn.init.kaiming_uniform_(w, nonlinearity='relu'),
                 device='cpu'):
        """param:tail_feat_ext is a pre-trained pyramid vision feature extractor
        this model expects a 31*31 gaze patch
        """
        channels, act, norm, bias = adjust_args([channels, act, norm, bias])
        super(GazePatchFeatureExtractor, self).__init__()
        self.device = device
        self.tail = tail_feat_ext(state_dict_pth, device=device)
        if state_dict_pth is None or not os.path.exists(state_dict_pth):
            self.tail.apply(lambda m: init_weights(m, w_init_f))

        self.head = nn.Sequential().to(device)
        self.head.add_module('norm_0', norm[0](channels[0],
                                               eps=1e-05,
                                               momentum=0.1,
                                               affine=bias[0],
                                               #bias=bias[0], version mismatch between GIVA servers and my own
                                               track_running_stats=True).to(device))
        self.head.add_module('act_0', act[0]().to(device))
                
        for i, chnn in enumerate(channels[1:], start=1):
            self.head.add_module('conv_{}'.format(i),
                                 nn.Conv2d(channels[i - 1],
                                           chnn,
                                           (3, 3) if i < 4 else (2, 2),
                                           bias=bias[i]).to(device))
            self.head.add_module('norm_{}'.format(i),
                                 norm[i](chnn,
                                         eps=1e-05,
                                         momentum=0.1,
                                         affine=bias[i],
                                         #bias=bias[i], version mismatch between GIVA servers and my own
                                         track_running_stats=True).to(device))
            self.head.add_module('act_{}'.format(i),
                                  act[i]().to(device))
                                  

        # tail is pre-trained, so no weight init
        self.head.apply(lambda m: init_weights(m, w_init_f))
        
                

    def forward(self, x):
        #pyramid feature extractor expects an RGB image
        _, c, _, _ = x.shape
        if c == 1:
            x = torch.tile(x, (1, 3, 1, 1))

        return self.head(self.tail(x))