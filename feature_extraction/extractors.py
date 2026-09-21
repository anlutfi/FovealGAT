from torch import nn
import torch
from feature_extraction.pretrained import pyramid_feature_extractor
from model.tools import init_weights


class GazePatchFeatureExtractor(nn.Module):
    """extracts a feature vector from a gaze patch image
    """
    def __init__(self,
                 tail_feat_ext=pyramid_feature_extractor,
                 in_channels=48,
                 out_channels=48,
                 act=nn.GELU,
                 bias=True,
                 w_init_f=lambda w: nn.init.kaiming_uniform_(w, nonlinearity='relu'),
                 device='cpu'):
        """param:tail_feat_ext is a pre-trained pyramid vision feature extractor
        this model expects a 31*31 gaze patch
        """
        super(GazePatchFeatureExtractor, self).__init__()
        self.device = device
        self.act = act()
        self.tail = tail_feat_ext()

        self.head = nn.Sequential(nn.BatchNorm2d(in_channels,
                                                 eps=1e-05,
                                                 momentum=0.1,
                                                 affine=True,
                                                 bias=bias,
                                                 track_running_stats=True),
                                  nn.Conv2d(in_channels,
                                            2 * in_channels,
                                            (3, 3),
                                            bias=bias),
                                  nn.BatchNorm2d(2 * in_channels,
                                                 eps=1e-05,
                                                 momentum=0.1,
                                                 affine=True,
                                                 bias=bias,
                                                 track_running_stats=True),
                                  self.act,
                                  nn.Conv2d(2 * in_channels,
                                            2 * in_channels,
                                            (3, 3),
                                            bias=bias),
                                  nn.BatchNorm2d(2 * in_channels,
                                                 eps=1e-05,
                                                 momentum=0.1,
                                                 affine=True,
                                                 bias=bias,
                                                 track_running_stats=True),
                                  self.act,
                                  nn.Conv2d(2 * in_channels,
                                            2 * in_channels,
                                            (3, 3),
                                            bias=bias),
                                  nn.BatchNorm2d(2 * in_channels,
                                                 eps=1e-05,
                                                 momentum=0.1,
                                                 affine=True,
                                                 bias=bias,
                                                 track_running_stats=True),
                                  self.act,
                                  nn.Conv2d(2 * in_channels,
                                            out_channels,
                                            (2, 2),
                                            bias=bias),
                                  nn.BatchNorm2d(out_channels,
                                                 eps=1e-05,
                                                 momentum=0.1,
                                                 affine=True,
                                                 bias=bias,
                                                 track_running_stats=True))

        self.head.apply(lambda m: init_weights(m, w_init_f))
        
                

    def forward(self, x):
        #pyramid feature extractor expects an RGB image
        _, c, _, _ = x.shape
        if c == 1:
            x = torch.tile(x, (1, 3, 1, 1))

        x = self.act(self.tail(x))
        return self.head(x)