import torch
from torch import nn
import os

try:
    from private_confs import base_pth # root dir path containing the whole project
except ModuleNotFoundError:
    print("warning: you should create a private_confs module with base_pth = 'path/to/my/project/root/dir'")
    print("setting base_pth to '.'")
    base_pth = '.'

# raw data paths
reflacx_dir = os.path.sep.join([base_pth, 'data/reflacx'])
mimic_dir = os.path.sep.join([base_pth, 'data/mimic/reflacx_imgs'])
mimic_part_dir = os.path.sep.join([base_pth, 'data/mimic_part_jpg'])

# reflacx metadata json file
full_meta_json = os.path.sep.join([base_pth, 'reflacx_lib/full_meta.json'])

# gaze path datasets paths
reflacx_gaze_patch_dataset_root = os.path.sep.join([base_pth, 'data/gaze_patch_datasets/reflacx'])
reflacx_gaze_dataset_pth = os.path.sep.join([base_pth, 'data/gaze_patch_datasets/reflacx/1_ang_res_256'])
mimic_gaze_patch_dataset_root = os.path.sep.join([base_pth, 'data/gaze_patch_datasets/mimic_eye'])
mimic_gaze_dataset_pth = os.path.sep.join([base_pth, 'data/gaze_patch_datasets/mimic_eye/0.037_0.035_pad31__resize256'])

dataset_type = 'reflacx'
assert dataset_type in ('reflacx', 'mimic')
train_val_test_split = [0.7, 0.2, 0.1]
transform = nn.Identity()
batch_size = 3
dl_num_workers = 4


from train.train_tools import choose_device
print('choosing least busy GPU...')
device = choose_device()
print('running on {}'.format(device))

#### model
import torch

desired_label = 'Atelectasis'
if dataset_type == 'reflacx':
    assert desired_label in ('Consolidation',
                             'Groundglass opacity',
                             'Atelectasis',
                             'Pulmonary edema',
                             'Pneumothorax',
                             'Enlarged cardiac silhouette')
elif dataset_type == 'mimic':
    assert desired_label in ('Normal',
                             'CHF',
                             'pneumonia')


criterion = nn.L1Loss
lr = 0.01
lr_pretrained = 0.003 # learning rate for pre-trained feature extractor tail
optimizer = torch.optim.AdamW
epochs = 100
store_best_model = True
store_last_model = True
stored_model_pth = os.path.sep.join([base_pth, 'models'])

#### feature extraction
from feature_extraction.pretrained import pyramid_feature_extractor
fe_tail_feat_ext = pyramid_feature_extractor
fe_state_dict_pth = os.path.sep.join([base_pth, 'GazeOnlyGAT', 'pretrained', 'pvig_ti_224_gelu.pt'])
fe_channels = [48, 128, 128, 128, 128]
fe_act = nn.GELU
fe_norm = nn.BatchNorm2d
fe_bias = True
fe_w_init_f = lambda w: nn.init.kaiming_uniform_(w, nonlinearity='relu')

##### node encodings
from gaze_datasets.dataset_tools import max_fixation_sequence_len
reflacx_max_len = max_fixation_sequence_len(reflacx_gaze_dataset_pth)
mimic_max_len = max_fixation_sequence_len(mimic_gaze_dataset_pth)
seq_encoding_feat_count = 16

#### Grapher
from torch.nn import functional as F
from torch_geometric.nn.norm import GraphNorm
import model.gat_conv as convs
in_channels = fe_channels[-1] + seq_encoding_feat_count + len(['x', 'y', 't'])
grph_gat_conv = convs.CausalGATConv
grph_layers = 8
grph_channels = 256
grph_residual_gat_levels = [False, True, False]
grph_output_gat_levels = None
grph_att_heads = [16, 8, 8, 8, 8, 8, 8, 8]
grph_xy_heads_ratio=0.2
grph_t_heads_ratio=0.1
grph_act_att = F.leaky_relu
grph_act_gat = F.elu
grph_norm = GraphNorm
grph_dropout = 0.15
grph_bias = True
grph_w_init_f = lambda w: nn.init.xavier_uniform_(w)

#### Readout
import model.readout as readouts
readout = readouts.CausalReadout
ro_channels = [-1, 512, 256, 128, 64]
ro_hidden_features = ro_channels[1:]
ro_out_channels = 1
ro_hidden_act = nn.GELU
ro_pred_act = nn.Identity
ro_dropout = 0.15
ro_norm = nn.GroupNorm
ro_bias = True
ro_w_init_f = lambda w: nn.init.kaiming_uniform_(w, nonlinearity='relu')