import torch
from torch import nn
import os

class Stem(nn.Module):
    """ Image to Visual Embedding
    Overlap: https://arxiv.org/pdf/2106.13797.pdf
    took this straight out of GazeGNN codebase to eliminate the need to import their whole modules
    """
    def __init__(self, in_channels=3, out_channels=48, act='gelu'):
        from train.train_tools import get_act #to avoid circular import TODO make this not shit
        act_layer = get_act(act)
        super(Stem, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, out_channels//2, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels//2),
            act_layer(),
            nn.Conv2d(out_channels//2, out_channels, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            act_layer(),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        x = self.convs(x)
        return x


def pyramid_feature_extractor(pretrained_sd='./pretrained/pvig_ti_224_gelu.pt',
                              in_channels=3,
                              out_channels=48,
                              act='gelu',
                              device='cpu'):
    model = Stem(in_channels, out_channels, act)
    if pretrained_sd is not None and os.path.exists(pretrained_sd):
        sd = torch.load(pretrained_sd)
        model.load_state_dict(sd)
    return model.to(device)
