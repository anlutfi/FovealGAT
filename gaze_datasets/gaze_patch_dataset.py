import torch
from torch.utils.data import Dataset
import os
import json

from gaze_datasets.dataset_tools import identity


class GazePatchDataset(Dataset):
    """Base gaze patch dataset to be extended
    """
    train = 'train'
    test = 'test'
    modes = (train, test)

    @staticmethod
    def load_sort_fnames(pth):
        result = os.listdir(pth)
        result.sort()
        return result
    

    def __init__(self,
                 dataset_path,
                 target_label,
                 target_fixation_keys=('t', 'x', 'y'), #TODO test with zoom level(bounding box size)
                 mode='train',
                 train_test_split=1.0,
                 transform=identity):
        """param:dataset_path is the directory containing the dataset gaze_patch raw data
           param:target_labels is a list of anomalies within those present in the dataset to return as ground truths
           param:target_fixation_keys are the fixation's attributes to be returned alongside the gaze patches
           param:mode train or test
           param:train_test_split a float between 0 and 1 denoting proportion of train subset. 
           param:transform which transformations are to be performed on each gaze patch

           This class automatically splits and shuffles data into stratified distribution according to param:train_test_split.
           If param:train_test_split is either 0.0 or 1.0, it does not shuffle.

           to keep the split consistent during the switch between trainning and testing,
           call GazePatchDataset.set_mode()
        """
        self.dataset_path = dataset_path
        self.fixations_path = os.path.sep.join([dataset_path, 'fixations'])
        self.gaze_patches_path = os.path.sep.join([dataset_path, 'gaze_patches'])
        self.labels_path = os.path.sep.join([dataset_path, 'labels'])
        self.target_label = target_label
        self.target_fixation_keys = target_fixation_keys
        self.train_test_split = train_test_split
        self.transform = transform
        self.fixations = GazePatchDataset.load_sort_fnames(self.fixations_path)

        gaze_patches_fns = GazePatchDataset.load_sort_fnames(self.gaze_patches_path)
        self.gaze_patches = {pth.split(os.path.sep)[-1]: pth
                             for pth in gaze_patches_fns}

        self.labels = None
        self.split_idxs = None
        self.train_split_idxs = torch.tensor([])
        self.test_split_idxs = torch.tensor([])

        self.mode = None
        self.set_mode(mode)
        

    def assign_split_idxs(self):
        self.split_idxs = (self.train_split_idxs
                           if self.mode == GazePatchDataset.train
                           else self.test_split_idxs)


    def set_mode(self, mode):
        assert mode.__class__ == str
        mode = mode.lower()
        assert mode in GazePatchDataset.modes
        self.mode = mode
        self.assign_split_idxs()


    def shuffle(self):
        if self.mode == GazePatchDataset.train and self.train_test_split == 1.0:
            self.train_split_idxs = torch.tensor(range(len(self.labels)))
            self.test_split_idxs = torch.tensor([])

        elif self.mode == GazePatchDataset.test and self.train_test_split == 0.0:
            self.train_split_idxs = torch.tensor([])
            self.test_split_idxs = torch.tensor(range(len(self.labels)))
            
        # if there is a non 100% split, stratify and shuffle dataset
        else:
            domain = torch.unique(self.labels)
            self.train_split_idxs = []
            self.test_split_idxs = []

            for v in domain:
                idxs = torch.where(self.labels == v)[0]
                idxs = idxs[torch.randperm(len(idxs))]
                cutoff = int(len(idxs) * self.train_test_split)

                self.train_split_idxs.append(idxs[:cutoff])
                self.test_split_idxs.append(idxs[cutoff:])

            self.train_split_idxs = torch.cat(self.train_split_idxs)
            self.test_split_idxs = torch.cat(self.test_split_idxs)
                
        self.assign_split_idxs()


    def init_labels(self, read_label_f):
        self.labels = []
        labels_fns = GazePatchDataset.load_sort_fnames(self.labels_path)
        for fname in labels_fns:
            with open(os.path.sep.join([self.labels_path, fname])) as fp:
                l = json.load(fp)
            self.labels.append(read_label_f(l, self.target_label))
        self.labels = torch.tensor(self.labels)
        self.shuffle()


    def __len__(self):
        if self.split_idxs is not None:
            return len(self.split_idxs)
        if self.mode == GazePatchDataset.train:
            return int(len(self.fixations) * self.train_test_split)
        return int(len(self.fixations) * (1 - self.train_test_split))


    def __getitem__(self, i):
        pass
                

