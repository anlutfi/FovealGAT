## ADJUST BEFORE RUNNING NOTEBOOKS AND TRAINING

# raw data paths
reflacx_dir = '.../data/reflacx'
mimic_dir = '.../data/mimic/reflacx_imgs'
mimic_part_dir = '.../data/mimic_part_jpg'

# reflacx metadata json file
full_meta_json = '.../reflacx_lib/full_meta.json'

# gaze path datasets paths
reflacx_gaze_patch_dataset_root = '.../gaze_patch_datasets/reflacx'
reflacx_gaze_dataset_pth = '.../gaze_patch_datasets/reflacx/1_ang_res_256'
mimic_gaze_patch_dataset_root = '.../gaze_patch_datasets/mimic_eye'
mimic_gaze_dataset_pth = '.../gaze_patch_datasets/mimic_eye/0.037_0.035_pad31__resize256'



import torch
from torch import nn


dataset_type = 'reflacx'
assert dataset_type in ('reflacx', 'mimic')
train_val_test_split = [0.7, 0.2, 0.1]
transform = nn.Identity()
batch_size = 4
dl_num_workers = 4


#### model
import torch
if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'

print('running on {}'.format(device))

desired_label = 'Atelectasis'
criterion = nn.L1Loss
lr = 0.01
optimizer = lambda model: torch.optim.AdamW(model, lr=lr)
epochs = 100
store_best_model = True
store_last_model = True
#stored_model_pth = '.'
stored_model_pth = '.'

#### feature extraction
from feature_extraction.pretrained import pyramid_feature_extractor
fe_tail_feat_ext=pyramid_feature_extractor
fe_in_channels=48
fe_out_channels=48
fe_act=nn.GELU
fe_bias=True
fe_w_init_f=lambda w: nn.init.kaiming_uniform_(w, nonlinearity='relu')

##### node encodings
from gaze_datasets.dataset_tools import max_fixation_sequence_len
reflacx_max_len = max_fixation_sequence_len(reflacx_gaze_dataset_pth)
mimic_max_len = max_fixation_sequence_len(mimic_gaze_dataset_pth)
seq_encoding_feat_count = 16

#### Grapher
from torch.nn import functional as F
from torch_geometric.nn.norm import GraphNorm
import model.GATConv as convs
grph_gat_conv = convs.CausalGATConv
grph_readout_gat_levels = None
grph_channels = [67, 134, 134, 134, 67]
grph_att_heads = 8
grph_act_att = F.leaky_relu
grph_act_gat = F.elu
grph_norm = GraphNorm
grph_dropout = [0.0, 0.1, 0.1, 0.1, 0.1]
grph_bias = True
grph_w_init_f = lambda w: nn.init.xavier_uniform_(w)

#### Readout
import model.readout as readouts
readout = readouts.CausalReadout
ro_in_channels = (sum(grph_channels)
                  if (grph_readout_gat_levels is None or len(grph_readout_gat_levels) == 0)
                  else sum([grph_channels[i] for i in grph_readout_gat_levels]))
ro_hidden_features = [1024, 1024, 1024]
ro_out_channels   = 1
ro_hidden_act = nn.GELU
ro_pred_act = nn.Identity
ro_dropout = 0.0
ro_norm = nn.BatchNorm1d
ro_bias = True
ro_w_init_f = lambda w: nn.init.kaiming_uniform_(w, nonlinearity='relu')



