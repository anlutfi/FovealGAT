# Replicating Publication Experiments

Parameters and dataset splits for models used for benchmarking in FovealGAT's publication are available in this folder. Here's how to access them.

### Dataset Splits


```python
import fovealgat_confs as conf
import torch
```


```python
pth = '/store/alutfi/dev/models/classification/xytreflacx_1_ang_res_256_Atelectasis_BCEWithLogitsLoss_AdamW/20261019_0117/session_0'
```

Fetch splits


```python
splits = torch.load('/'.join([pth, 'dataset_splits.pt']))
```


```python
{k: splits[k][:10] for k in splits}
```




    {'train': tensor([2398,  762, 1393, 2276,    8, 2954, 1481,  289, 2780, 2171]),
     'val': tensor([1776,  497, 1163, 1469, 2359, 2274,  942, 2781,  827,  353]),
     'test': tensor([2990, 2496, 2288, 2546, 2687,  743,  454, 2986, 2347, 1437])}



Load Splits


```python
from gaze_datasets.reflacx import ReflacxGazePatchDataset
gaze_dataset_pth = conf.reflacx_gaze_dataset_pth
desired_label = 'Atelectasis'
ds = ReflacxGazePatchDataset(gaze_dataset_pth, desired_label)
```


```python
ds.load_splits_from_training(splits)
```

### Check Model and Training Parameters


```python
model = torch.load('/'.join([pth, 'best_xytreflacx_1_ang_res_256_Atelectasis_BCEWithLogitsLoss_AdamW.pt']), map_location='cpu')
```


```python
model['train_args']
```




    {'sessions': 5,
     'epochs': 100,
     'lr': 0.0003,
     'pretrained_lr': 1e-05,
     'store_best_model': True,
     'stored_model_pth': '/store/alutfi/dev/models/classification',
     'loss': 'bce_with_logits',
     'optimizer': 'adamw',
     'dataset_pth': '/store/alutfi/dev/data/gaze_patch_datasets/reflacx/1_ang_res_256',
     'pretrained_sd': '',
     'gaze_noise': 0.0}




```python
model['model_args']
```




    {'train_val_test_split': [0.7, 0.2, 0.1],
     'transform': [['RandomHorizontalFlip', {'p': 0.5}],
      ['RandomRotation', {'degrees': 7}],
      ['ColorJitter', {'brightness': 0.05, 'contrast': 0.05}]],
     'batch_size': 5,
     'acc_steps': 8,
     'dl_num_workers': 9,
     'fe_feat_ext': 'GazePatchFeatureExtractor',
     'fe_channels': [48, 128, 128, 128, 128],
     'fe_act': 'gelu',
     'fe_norm': 'BatchNorm2d',
     'fe_bias': True,
     'fe_w_init_f': 'kaiming',
     'seq_encoding_feat_count': 16,
     'grph_gat_conv': 'CausalIOU_EuclideanTGATConv',
     'grph_res_seq': 'MarkovResidualSequential',
     'grph_layers': 8,
     'grph_channels': 384,
     'grph_residual_gat_levels': [False, True, True, True, True, True, True, True],
     'grph_output_gat_levels': None,
     'grph_att_heads': [16, 8, 8, 8, 8, 8, 8, 8],
     'grph_xy_heads_ratio': 0.2,
     'grph_t_heads_ratio': 0.1,
     'grph_iou_heads_ratio': 0.2,
     'self_edges': True,
     'mask_out_zeros': True,
     'grph_act_att': 'leaky_relu',
     'grph_act_gat': 'elu',
     'grph_norm': 'GraphNorm',
     'grph_dropout': 0.3,
     'grph_bias': True,
     'grph_w_init_f': 'xavier',
     'readout': 'CausalReadout',
     'ro_channels': [-1, 512, 256, 128, 64],
     'ro_out_channels': 1,
     'ro_hidden_act': 'gelu',
     'ro_pred_act': 'Identity',
     'ro_dropout': 0.3,
     'ro_norm': 'GroupNorm',
     'ro_bias': True,
     'ro_w_init_f': 'kaiming'}



### Shellscripts used for Training

Inside this folder there are the shell scripts used to training the pusblished models for regression and classification. When a REFLACX label is missing from classifications, use the one from regression because they share the same parameters.

the command line arguments are in the train_args and model_args subfolders
