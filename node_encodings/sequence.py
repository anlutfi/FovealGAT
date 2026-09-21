import numpy as np
import torch

class GazeSinCosPosEncoding:
    def __init__(self, max_scanpath_len, feature_count):
        position = np.arange(max_scanpath_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, feature_count, 2) *
                        -(np.log(10000.0) / feature_count))
        
        encodings = np.zeros((max_scanpath_len, feature_count))
        encodings[:, 0::2] = np.sin(position * div_term)
        if feature_count % 2 == 0:
            encodings[:, 1::2] = np.cos(position * div_term)
        else:
            encodings[:, 1::2] = np.cos(position * div_term)[:,0:-1]

        self.max_scanpath_len = max_scanpath_len
        self.feature_count = feature_count
        self.encodings = torch.tensor(encodings)

    def get_encodings(self, scanpath_len):
        assert scanpath_len <= self.max_scanpath_len
        return self.encodings[:scanpath_len, :]

    def __call__(self, arg, adjust_dim=True):
        """arg is either a length of a list of lengths"""
        assert arg.__class__ in (int, list, tuple)
        if arg.__class__ == int:
            result = self.get_encodings(arg)
        else:
            result = torch.cat([self.get_encodings(l) for l in arg])

        if adjust_dim:
            result = result.unsqueeze(-1).unsqueeze(-1)

        return result
        
