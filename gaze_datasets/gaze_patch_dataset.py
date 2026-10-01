import torch
from torch.utils.data import Dataset
import os
import json
from math import ceil # needed only in python 3.11
from math import log

from gaze_datasets.dataset_tools import identity


class GazePatchDataset(Dataset):
    """Base gaze patch dataset to be extended
    """
    train = 'train'
    test = 'test'
    val = 'val'
    modes = (train, test, val)

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
                 train_val_test_split=1.0,
                 transform=identity,
                 min_max_duration_pth='./dataset_t_info.json'):
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

        assert train_val_test_split.__class__ in (float, int, list)
        if train_val_test_split.__class__ in(float, int):
            assert int(train_val_test_split) == 1
            train_val_test_split = [1.0, 0.0, 0.0]
        else:
            assert len(train_val_test_split) == 3 
            assert ceil(sum(train_val_test_split)) == 1 ## only in python 3.11 that thinks that int(0.99999999999) == 0
        self.train_val_test_split = train_val_test_split
        self.transform = transform
        self.fixations = GazePatchDataset.load_sort_fnames(self.fixations_path)

        #normalizing fixation duration to 0 a 1 range
        #using log transform on min max t
        if os.path.exists(min_max_duration_pth):
            with open(min_max_duration_pth, 'r') as fp:
                minmax = json.load(fp)
            self.min_log_t = minmax['min_log']
            self.max_log_t = minmax['max_log']
        else: #find min and max t for all dataset
            self.min_log_t = float('inf')
            self.max_log_t = float('-inf')
            for fix_pth in self.fixations:
                pth = os.path.sep.join([self.fixations_path, fix_pth])
                with open(pth, 'r') as fp:
                    fixs = json.load(fp)
                for i, fix in enumerate(fixs):
                    if fix['t'] == 0: # TODO remove this when correcting mimic generation
                        continue
                    log_t = log(fix['t'])
                    self.min_log_t = min(self.min_log_t, log_t)
                    self.max_log_t = max(self.max_log_t, log_t)
            with open(min_max_duration_pth, 'w') as fp:
                json.dump({'min_log': self.min_log_t,
                           'max_log': self.max_log_t},
                          fp)
                                    
                        

        gaze_patches_fns = GazePatchDataset.load_sort_fnames(self.gaze_patches_path)
        self.gaze_patches = {pth.split(os.path.sep)[-1]: pth
                             for pth in gaze_patches_fns}

        self.labels = None
        self.split_idxs = None
        self.train_idxs = torch.tensor([])
        self.val_idxs = torch.tensor([])
        self.test_idxs = torch.tensor([])

        self.mode = None
        self.set_mode(mode)

        
    def normalize_t(self, t):
        return (log(t) - self.min_log_t) / (self.max_log_t - self.min_log_t)

    
    def assign_split_idxs(self):
        if self.mode == GazePatchDataset.train:
            self.split_idxs = self.train_idxs
        if self.mode == GazePatchDataset.val:
            self.split_idxs = self.val_idxs
        if self.mode == GazePatchDataset.test:
            self.split_idxs = self.test_idxs


    def load_splits_from_training(self, splits):
        self.train_idxs = splits['train']
        self.val_idxs = splits['val']
        self.test_idxs = splits['test']
        self.assign_split_idxs()                


    def set_mode(self, mode):
        assert mode.__class__ == str
        mode = mode.lower()
        assert mode in GazePatchDataset.modes
        self.mode = mode
        self.assign_split_idxs()


    def shuffle(self):
        if self.mode == GazePatchDataset.train and self.train_val_test_split[0] == 1.0:
            self.train_idxs = torch.tensor(range(len(self.labels)))
            self.val_idxs = torch.tensor([])
            self.test_idxs = torch.tensor([])

        elif self.mode == GazePatchDataset.test and self.train_val_test_split[1] == 1.0:
            self.train_idxs = torch.tensor([])
            self.val_idxs = torch.tensor(range(len(self.labels)))
            self.test_idxs = torch.tensor([])

        elif self.mode == GazePatchDataset.test and self.train_val_test_split[2] == 1.0:
            self.train_idxs = torch.tensor([])
            self.val_idxs = torch.tensor([])
            self.test_idxs = torch.tensor(range(len(self.labels)))
            
        # if there is a non 100% split, stratify and shuffle dataset
        else:
            domain = torch.unique(self.labels)
            self.train_idxs = []
            self.val_idxs = []
            self.test_idxs = []

            for v in domain:
                idxs = torch.where(self.labels == v)[0]
                idxs = idxs[torch.randperm(len(idxs))]
                cutoff_train = int(len(idxs) * self.train_val_test_split[0])
                cutoff_val = cutoff_train + int(len(idxs) * self.train_val_test_split[1])
                
                self.train_idxs.append(idxs[:cutoff_train])
                self.val_idxs.append(idxs[cutoff_train:cutoff_val])
                self.test_idxs.append(idxs[cutoff_val:])

            self.train_idxs = torch.cat(self.train_idxs)
            self.val_idxs = torch.cat(self.val_idxs)
            self.test_idxs = torch.cat(self.test_idxs)
                
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
            return int(len(self.fixations) * self.train_test_split[0])
        elif self.mode == GazePatchDataset.val:
            return int(len(self.fixations) * self.train_test_split[1])
        elif self.mode == GazePatchDataset.test:
            return int(len(self.fixations) * (self.train_test_split[2]))


    def __getitem__(self, i):
        pass
                

