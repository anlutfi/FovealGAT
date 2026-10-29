import torch
import os
import json

from gaze_datasets.dataset_tools import identity, break_iterables
from gaze_datasets.gaze_patch_dataset import GazePatchDataset


class ReflacxGazePatchDataset(GazePatchDataset):
    """Loads REFLACX previously converted to a gaze_patch dataset
    """
    def __init__(self,
                 dataset_path,
                 target_label,
                 target_fixation_keys=('t', 'x', 'y'),
                 mode='train',
                 gaze_noise=0,
                 train_val_test_split=1.0,
                 transform=identity,
                 min_max_duration_pth='./reflacx_t_info.json'):
        """param:dataset_path is the directory containing the reflacx gaze_patch raw data
           param:target_labels is the label for a desired anomaly within those present in REFLACX to return as ground truths
           param:target_fixation_keys are the fixation's attributes to be returned alongside the gaze patches
           param:mode train or test
           param:transform which transformations are to be performed on each gaze patch
        """
        super().__init__(dataset_path,
                         target_label,
                         target_fixation_keys,
                         mode,
                         gaze_noise,
                         train_val_test_split,
                         transform,
                         min_max_duration_pth)
        self.init_labels()


    def init_labels(self):
        f = lambda labels, target_labels: labels[target_labels]
        super().init_labels(f)


    def __getitem__(self, i):
        #adjust index relative to current train/test split
        i = self.split_idxs[i]

        r_id = self.fixations[i].split('.')[0]
                
        with open(os.path.sep.join([self.fixations_path, self.fixations[i]])) as fp:
            fixations = json.load(fp)

        for i, fix in enumerate(fixations):
            fixations[i]['t'] = self.normalize_t(fix['t'])

        if self.target_fixation_keys is not None and self.target_fixation_keys != []:
            l = [[f[k] for k in self.target_fixation_keys] for f in fixations]
        else:
            l = [[f[k] for k in f] for f in fixations]

        fixations = [break_iterables(fs) for fs in l]
        fixations = torch.asarray(fixations)
        
        if self.mode == GazePatchDataset.train and self.gaze_noise > 0.0:
            ts, xs, ys, *tail = torch.split(fixations, 1, dim=-1)
            xs += torch.randn_like(xs) * self.gaze_noise
            ys += torch.randn_like(ys) * self.gaze_noise
            xs = torch.clamp(xs, 0.0, 1.0)
            ys = torch.clamp(ys, 0.0, 1.0)
            fixations = torch.cat([ts, xs, ys, *tail], dim=1)
                                    
        
        fixations = fixations.unsqueeze(-1).unsqueeze(-1)

        
        
        label = self.labels[i]

        gaze_patches_dir = os.path.sep.join([self.gaze_patches_path,
                                             self.gaze_patches[r_id]])
        l_gp = os.listdir(gaze_patches_dir)
        l_gp.sort(key=lambda x: int(x.split('.')[0]))
        gaze_patches = None
        for gaze_patch_pth in l_gp:
            gp = torch.load(os.path.sep.join([gaze_patches_dir, gaze_patch_pth]))
            gp = self.transform(gp)
            if gaze_patches is None:
                gaze_patches = gp
            else:
                gaze_patches = torch.cat((gaze_patches, gp), dim=0)

        return fixations, gaze_patches, label