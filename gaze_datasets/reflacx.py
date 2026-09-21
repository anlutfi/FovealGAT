import torch
import os
import json

from gaze_datasets.dataset_tools import identity
from gaze_datasets.gaze_patch_dataset import GazePatchDataset


class ReflacxGazePatchDataset(GazePatchDataset):
    """Loads REFLACX previously converted to a gaze_patch dataset
    """
    def __init__(self,
                 dataset_path,
                 target_label,
                 target_fixation_keys=('t', 'x', 'y'), #TODO test with zoom level(bounding box size)
                 mode='train',
                 train_test_split=1.0,
                 transform=identity):
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
                         train_test_split,
                         transform)
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
        
        if self.target_fixation_keys is not None and self.target_fixation_keys != []:
            fixations = torch.asarray([[f[k] for k in self.target_fixation_keys]
                                       for f in fixations])
        else:
            fixations = torch.asarray([[f[k] for k in f]
                                       for f in fixations])
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