import sys
import os
import json
import torch
from torch import nn

import only_gaze_confs

with open(sys.argv[1], 'r') as fp:
    args = json.load(fp)

desired_label = args['desired_label']
assert desired_label in ('Consolidation',
                         'Groundglass opacity',
                         'Atelectasis',
                         'Pulmonary edema',
                         'Pneumothorax',
                         'Enlarged cardiac silhouette')

criterion = nn.L1Loss
lr = 0.01
lr_pretrained = 0.003 # learning rate for pre-trained feature extractor tail
optimizer = lambda model: torch.optim.AdamW(model, lr=lr)
epochs = 100
store_best_model = True
store_last_model = True