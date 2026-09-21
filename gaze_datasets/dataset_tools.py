import torch
import json
import os

def identity(x):
    return x 


def gaze_patch_dataset_collate(batch):
    """collate function to be used in torch.utils.data.DataLoader's
    that iterate through a ReflacxGazePatchDataset
    """
    lens = []
    fixations = []
    patches = []
    labels = []
    for item in batch:
        lens.append(len(item[0]))
        fixations.append(item[0])
        patches.append(item[1])
        labels.append(item[2].unsqueeze(0))
    fixations = torch.cat(fixations, dim=0)
    patches = torch.cat(patches, dim=0).unsqueeze(1)
    labels = torch.cat(labels, dim=0)
    return lens, fixations, patches, labels


def max_fixation_sequence_len(gaze_dataset_pth):
    """fetch largest fixation sequence in a dataset
    """
    fix_dir = os.path.sep.join([gaze_dataset_pth, 'fixations'])

    max_len = 0
    for rid in os.listdir(fix_dir):
        pth = os.path.sep.join([fix_dir, rid])
        with open(pth) as fp:
            max_len = max(max_len, len(json.load(fp)))

    return max_len