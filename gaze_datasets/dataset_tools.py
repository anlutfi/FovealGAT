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


def all_pairs(t):
    t = t.unsqueeze(-1)
    mat = torch.tile(t, (1, 1, t.shape[1]))
    matT = torch.transpose(mat, 2, 1)
    adjs = torch.cat((mat, matT), dim=0)
    return adjs


def get_xyt_pair_matrices(fixations, splits):
    fixations = fixations.squeeze(-1).squeeze(-1)
    ts, xs, ys = torch.split(fixations, (1, 1, 1), dim=-1)
    
    ts = ts.transpose(1, 0)
    xs = xs.transpose(1, 0)
    ys = ys.transpose(1, 0)

    ts = torch.split(ts, splits, dim=-1)
    xys = torch.cat((xs, ys), dim=0)
    xys = torch.split(xys, splits, dim=-1)

    xy_mats = [all_pairs(c) for c in xys]
    t_mats = [all_pairs(t) for t in ts]

    return xy_mats, t_mats

def get_euclidean_encodings(xy_mat, eps=1e-8):
    xy1, xy2 = torch.split(xy_mat, 2, dim=0)
    x1, y1 = torch.split(xy1, 1, dim=0)
    x2, y2 = torch.split(xy2, 1, dim=0)
    delta_x = torch.abs(x2 - x1)
    delta_y = torch.abs(y2 - y1)
    euc_dist = torch.sqrt(delta_x ** 2 + delta_y ** 2 + eps)
    return torch.cat([xy_mat, delta_x, delta_y, euc_dist], dim=0)


def break_iterables(l):
    result = []
    [(result.append(x) if x.__class__ not in (tuple, list) else [result.append(y) for y in x]) for x in l]
    return result


def delta_bb(u1, v1, u2, v2):
    d_u1v2 = torch.abs(u1 - v2)
    d_u2v1 = torch.abs(u2 - v1)
    return torch.min(torch.cat([d_u1v2, d_u2v1], dim=0), dim=0).values


def area_bb(l, r, t, b):
    return (r - l) * (b - t)


def get_bbox_iou_encodings(batch_bboxes, splits):
    batch_bboxes = batch_bboxes.squeeze(-1).squeeze(-1)
    batch_bboxes = torch.split(batch_bboxes, splits, dim=-1)
    iou_mats = [all_pairs(bbs) for bbs in batch_bboxes]
    return [get_bbox_iou(bbs) for bbs in iou_mats]


def get_bbox_iou(bboxes):
    l1, t1, r1, b1, l2, t2, r2, b2 = torch.split(bboxes, 1, dim=0)
    
    delta_lr = delta_bb(l1, r1, l2, r2)
    delta_tb = delta_bb(t1, b1, t2, b2)
    
    intersection = (delta_tb * delta_lr).unsqueeze(0)
    intersection[l2 >= r1] = 0.0
    intersection[l1 >= r2] = 0.0
    intersection[t2 >= b1] = 0.0
    intersection[t1 >= b2] = 0.0
    
    area_1 = area_bb(l1, r1, t1, b1)
    area_2 = area_bb(l2, r2, t2, b2)
    
    union = area_1 + area_2 - intersection
    return intersection / union          
