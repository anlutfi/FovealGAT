from torch.nn.functional import pad
from math import ceil, floor

def pad_patch(patch, padding, pad_value=0):
    """adds zero-padding all around a numpy array so it becomes
    a square of param:padding x param:padding pixels.
    Centers the original array as best as it can"""
    _, h, w = patch.shape
    top = max(0, ceil((padding - h) / 2))
    bottom = max(0, floor((padding - h) / 2))
    
    left = max(0, ceil((padding - w) / 2))
    right = max(0, floor((padding - w) / 2))
    return pad(patch,
               (left, right, top, bottom),
               mode='constant',
               value=pad_value)