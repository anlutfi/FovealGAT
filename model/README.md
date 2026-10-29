## FovealGAT: components overview


```python
import fovealgat_confs as conf
```

With the gaze patch features and encodings aggregated into graph nodes, we can feed it through our network. We'll go through each component.

Fetch sample batch


```python
import torch
from gaze_datasets.reflacx import ReflacxGazePatchDataset
from gaze_datasets.dataset_tools import gaze_patch_dataset_collate
from feature_extraction.extractors import GazePatchFeatureExtractor
from node_encodings.sequence import GazeSinCosPosEncoding

gaze_dataset_pth = conf.reflacx_gaze_dataset_pth
desired_label = 'Atelectasis'
ds = ReflacxGazePatchDataset(gaze_dataset_pth, desired_label,
                             target_fixation_keys=('t', 'x', 'y', 'top_left', 'bottom_right'))
splits, fixations, patches, labels = gaze_patch_dataset_collate((ds[33], ds[34], ds[0]))

feat_ext = GazePatchFeatureExtractor(conf.fe_tail_feat_ext,
                                     conf.fe_state_dict_pth,
                                     conf.fe_channels,
                                     conf.fe_act,
                                     conf.fe_norm,
                                     conf.fe_bias,
                                     conf.fe_w_init_f,
                                     conf.device)
feats = feat_ext(patches.to(conf.device))

max_len = conf.reflacx_max_len
seq_encoding_featcount = conf.seq_encoding_feat_count

seq_encoding = GazeSinCosPosEncoding(max_len, seq_encoding_featcount)
seqs = seq_encoding(splits, adjust_dim=True).to(conf.device)

fixations = fixations.to(conf.device)
fixations, bboxes = torch.split(fixations, (3, 4), dim=1)
nodes = torch.cat([fixations, seqs, feats], 1).float()
ns = nodes.squeeze(-1).squeeze(-1).permute((1, 0))
bboxes = bboxes.squeeze(-1).squeeze(-1).transpose(1, 0)
```

### GAT Layer
<a id="gat_layer"></a>

The first component is the Graph Attention Convolution Layer. As there are many choices, there is a seaparate [ReadMe](./gat_convs/README.md)

### GazeGrapher
<a id="grapher"></a>

GazeGrapher is a module that aggregates several GAT layers in sequence, performing all graph convolutions before graph readout and prediction.

PyTorch's native nn.Sequential does not keep residual tensors between its internal steps, so we need to extend it. Graph readout and predictions will need residual, intermediary, graphs


```python
from model.gat_convs.gat_conv import CausalGATConv
from model.gat_convs.xyt_gat_conv import CausalXYTGATConv, CausalEuclideanTGATConv
from model.gat_convs.iou_xyt_gat_conv import CausalIOU_XYTGATConv, CausalIOU_EuclideanTGATConv
```

We need input sizes for each of the graphs GAT Layers, and the residual layers' indices and GAT conv class of choice


```python
layers = 4
gat_channels = 256 # input size and 4 GAT convolution layers
residual_gat_levels = [False, True, True, True]
att_heads = [16, 8, 8, 8]
output_gat_levels = None
gat_conv = CausalGATConv
```


```python
from grapher import GazeGrapher, XYTGazeGrapher
```


```python
from torch import nn
from torch_geometric.nn.norm import GraphNorm
import torch.nn.functional as F
```

Semantic Attention Model


```python
grapher = GazeGrapher(layers=layers,
                      output_gat_levels=output_gat_levels,
                      residual_gat_levels=residual_gat_levels,
                      att_heads=att_heads,
                      gat_conv_args={'channels': gat_channels,
                                     'act_att': F.leaky_relu,
                                     'act_gat': F.elu,
                                     'norm': GraphNorm,
                                     'dropout': 0.1,
                                     'bias': True,
                                     'w_init_f': lambda w: nn.init.xavier_uniform_(w)},
                      device=conf.device)
grapher
```




    GazeGrapher(
      (gats): MarkovResidualSequential(
        (GAT_0): CausalGATConv(
          (att_scores): Conv2d(512, 16, kernel_size=(1, 1), stride=(1, 1))
          (node_update): Conv1d(4096, 256, kernel_size=(1,), stride=(1,))
          (norm): GraphNorm(256)
          (dropout): Dropout(p=0.1, inplace=False)
        )
        (GAT_1): CausalGATConv(
          (att_scores): Conv2d(512, 8, kernel_size=(1, 1), stride=(1, 1))
          (node_update): Conv1d(2048, 256, kernel_size=(1,), stride=(1,))
          (norm): GraphNorm(256)
          (dropout): Dropout(p=0.1, inplace=False)
        )
        (GAT_2): CausalGATConv(
          (att_scores): Conv2d(512, 8, kernel_size=(1, 1), stride=(1, 1))
          (node_update): Conv1d(2048, 256, kernel_size=(1,), stride=(1,))
          (norm): GraphNorm(256)
          (dropout): Dropout(p=0.1, inplace=False)
        )
        (GAT_3): CausalGATConv(
          (att_scores): Conv2d(512, 8, kernel_size=(1, 1), stride=(1, 1))
          (node_update): Conv1d(2048, 256, kernel_size=(1,), stride=(1,))
          (norm): GraphNorm(256)
          (dropout): Dropout(p=0.1, inplace=False)
        )
      )
    )




```python
expand_dims = nn.Conv1d(ns.shape[0], gat_channels, 1).to(conf.device)
ens = expand_dims(ns)
ens.shape
```




    torch.Size([256, 255])




```python
graphs = grapher((ens, splits)) # output is all selected residual graphs by residual_gat_levels
[g.shape for g in graphs]
```




    [torch.Size([256, 255]),
     torch.Size([256, 255]),
     torch.Size([256, 255]),
     torch.Size([256, 255])]



Semantic + Space + Duration (XYT) Models


```python
grapher = XYTGazeGrapher(layers=layers,
                         output_gat_levels=output_gat_levels,
                         residual_gat_levels=residual_gat_levels,
                         att_heads=att_heads,
                         gat_conv_args={'channels': gat_channels,
                                        'xy_heads_ratio': 0.2,
                                        't_heads_ratio': 0.1,
                                        'act_att': F.leaky_relu,
                                        'act_gat': F.elu,
                                        'norm': GraphNorm,
                                        'dropout': 0.1,
                                        'bias': True,
                                        'w_init_f': lambda w: nn.init.xavier_uniform_(w)},
                         device=conf.device)
grapher
```




    XYTGazeGrapher(
      (gats): MarkovResidualSequential(
        (GAT_0): CausalXYTGATConv(
          (att_scores): Conv2d(512, 16, kernel_size=(1, 1), stride=(1, 1))
          (node_update): Conv1d(5632, 256, kernel_size=(1,), stride=(1,))
          (norm): GraphNorm(256)
          (dropout): Dropout(p=0.1, inplace=False)
          (xy_scores): Conv2d(4, 4, kernel_size=(1, 1), stride=(1, 1))
          (t_scores): Conv2d(2, 2, kernel_size=(1, 1), stride=(1, 1))
        )
        (GAT_1): CausalXYTGATConv(
          (att_scores): Conv2d(512, 8, kernel_size=(1, 1), stride=(1, 1))
          (node_update): Conv1d(2816, 256, kernel_size=(1,), stride=(1,))
          (norm): GraphNorm(256)
          (dropout): Dropout(p=0.1, inplace=False)
          (xy_scores): Conv2d(4, 2, kernel_size=(1, 1), stride=(1, 1))
          (t_scores): Conv2d(2, 1, kernel_size=(1, 1), stride=(1, 1))
        )
        (GAT_2): CausalXYTGATConv(
          (att_scores): Conv2d(512, 8, kernel_size=(1, 1), stride=(1, 1))
          (node_update): Conv1d(2816, 256, kernel_size=(1,), stride=(1,))
          (norm): GraphNorm(256)
          (dropout): Dropout(p=0.1, inplace=False)
          (xy_scores): Conv2d(4, 2, kernel_size=(1, 1), stride=(1, 1))
          (t_scores): Conv2d(2, 1, kernel_size=(1, 1), stride=(1, 1))
        )
        (GAT_3): CausalXYTGATConv(
          (att_scores): Conv2d(512, 8, kernel_size=(1, 1), stride=(1, 1))
          (node_update): Conv1d(2816, 256, kernel_size=(1,), stride=(1,))
          (norm): GraphNorm(256)
          (dropout): Dropout(p=0.1, inplace=False)
          (xy_scores): Conv2d(4, 2, kernel_size=(1, 1), stride=(1, 1))
          (t_scores): Conv2d(2, 1, kernel_size=(1, 1), stride=(1, 1))
        )
      )
    )




```python
expand_dims = nn.Conv1d(ns.shape[0], gat_channels, 1).to(conf.device)
ens = expand_dims(ns)
ens.shape
```




    torch.Size([256, 255])




```python
graphs = grapher((ens, splits, fixations)) # output is all selected residual graphs by residual_gat_levels
[g.shape for g in graphs]
```




    [torch.Size([256, 255]),
     torch.Size([256, 255]),
     torch.Size([256, 255]),
     torch.Size([256, 255])]



Semantic + Space + Euclidean Measures + Duration (EuclideanT) Models


```python
from grapher import EuclideanTGazeGrapher
```


```python
grapher = EuclideanTGazeGrapher(layers=layers,
                                output_gat_levels=output_gat_levels,
                                residual_gat_levels=residual_gat_levels,
                                att_heads=att_heads,
                                gat_conv_args={'channels': gat_channels,
                                               'xy_heads_ratio': 0.2,
                                               't_heads_ratio': 0.1,
                                               'act_att': F.leaky_relu,
                                               'act_gat': F.elu,
                                               'norm': GraphNorm,
                                               'dropout': 0.1,
                                               'bias': True,
                                               'w_init_f': lambda w: nn.init.xavier_uniform_(w)},
                                device=conf.device)
grapher
```




    EuclideanTGazeGrapher(
      (gats): MarkovResidualSequential(
        (GAT_0): CausalEuclideanTGATConv(
          (att_scores): Conv2d(512, 16, kernel_size=(1, 1), stride=(1, 1))
          (node_update): Conv1d(5632, 256, kernel_size=(1,), stride=(1,))
          (norm): GraphNorm(256)
          (dropout): Dropout(p=0.1, inplace=False)
          (xy_scores): Conv2d(7, 4, kernel_size=(1, 1), stride=(1, 1))
          (t_scores): Conv2d(2, 2, kernel_size=(1, 1), stride=(1, 1))
        )
        (GAT_1): CausalEuclideanTGATConv(
          (att_scores): Conv2d(512, 8, kernel_size=(1, 1), stride=(1, 1))
          (node_update): Conv1d(2816, 256, kernel_size=(1,), stride=(1,))
          (norm): GraphNorm(256)
          (dropout): Dropout(p=0.1, inplace=False)
          (xy_scores): Conv2d(7, 2, kernel_size=(1, 1), stride=(1, 1))
          (t_scores): Conv2d(2, 1, kernel_size=(1, 1), stride=(1, 1))
        )
        (GAT_2): CausalEuclideanTGATConv(
          (att_scores): Conv2d(512, 8, kernel_size=(1, 1), stride=(1, 1))
          (node_update): Conv1d(2816, 256, kernel_size=(1,), stride=(1,))
          (norm): GraphNorm(256)
          (dropout): Dropout(p=0.1, inplace=False)
          (xy_scores): Conv2d(7, 2, kernel_size=(1, 1), stride=(1, 1))
          (t_scores): Conv2d(2, 1, kernel_size=(1, 1), stride=(1, 1))
        )
        (GAT_3): CausalEuclideanTGATConv(
          (att_scores): Conv2d(512, 8, kernel_size=(1, 1), stride=(1, 1))
          (node_update): Conv1d(2816, 256, kernel_size=(1,), stride=(1,))
          (norm): GraphNorm(256)
          (dropout): Dropout(p=0.1, inplace=False)
          (xy_scores): Conv2d(7, 2, kernel_size=(1, 1), stride=(1, 1))
          (t_scores): Conv2d(2, 1, kernel_size=(1, 1), stride=(1, 1))
        )
      )
    )




```python
expand_dims = nn.Conv1d(ns.shape[0], gat_channels, 1).to(conf.device)
ens = expand_dims(ns)
ens.shape
```




    torch.Size([256, 255])




```python
graphs = grapher((ens, splits, fixations)) # output is all selected residual graphs by residual_gat_levels
[g.shape for g in graphs]
```




    [torch.Size([256, 255]),
     torch.Size([256, 255]),
     torch.Size([256, 255]),
     torch.Size([256, 255])]



XYT + IOU Models


```python
import torch
from gaze_datasets.reflacx import ReflacxGazePatchDataset
from gaze_datasets.dataset_tools import gaze_patch_dataset_collate
from feature_extraction.extractors import GazePatchFeatureExtractor
from node_encodings.sequence import GazeSinCosPosEncoding

gaze_dataset_pth = conf.reflacx_gaze_dataset_pth
desired_label = 'Atelectasis'
ds = ReflacxGazePatchDataset(gaze_dataset_pth, desired_label, target_fixation_keys=('t', 'x', 'y', 'top_left', 'bottom_right'))
splits, fixations, patches, labels = gaze_patch_dataset_collate((ds[33], ds[34], ds[0]))

feat_ext = GazePatchFeatureExtractor(conf.fe_tail_feat_ext,
                                     conf.fe_state_dict_pth,
                                     conf.fe_channels,
                                     conf.fe_act,
                                     conf.fe_norm,
                                     conf.fe_bias,
                                     conf.fe_w_init_f,
                                     conf.device)
feats = feat_ext(patches.to(conf.device))

max_len = conf.reflacx_max_len
seq_encoding_featcount = conf.seq_encoding_feat_count

seq_encoding = GazeSinCosPosEncoding(max_len, seq_encoding_featcount)
seqs = seq_encoding(splits, adjust_dim=True).to(conf.device)
fixations = fixations.to(conf.device)
fixations, bboxes = torch.split(fixations, (3, 4), dim=1)

nodes = torch.cat([fixations, seqs, feats], 1).float()
ns = nodes.squeeze(-1).squeeze(-1).permute((1, 0))
bboxes = bboxes.squeeze(-1).squeeze(-1).transpose(1, 0)
ns.shape, bboxes.shape, sum([x.shape[1]for x in (seqs, fixations, feats)])
```




    (torch.Size([147, 255]), torch.Size([4, 255]), 147)




```python
from grapher import IOU_XYTGazeGrapher
```


```python
grapher = IOU_XYTGazeGrapher(gat_conv=CausalIOU_XYTGATConv,
                                layers=layers,
                                output_gat_levels=[layers - 1],
                                residual_gat_levels=residual_gat_levels,
                                att_heads=att_heads,
                                gat_conv_args={'channels': gat_channels,
                                               'xy_heads_ratio': 0.2,
                                               't_heads_ratio': 0.1,
                                               'iou_heads_ratio': 0.2,
                                               'act_att': F.leaky_relu,
                                               'act_gat': F.elu,
                                               'norm': GraphNorm,
                                               'dropout': 0.1,
                                               'bias': True,
                                               'w_init_f': lambda w: nn.init.xavier_uniform_(w)},
                                device=conf.device)
grapher
```




    IOU_XYTGazeGrapher(
      (gats): MarkovResidualSequential(
        (GAT_0): CausalIOU_XYTGATConv(
          (att_scores): Conv2d(512, 16, kernel_size=(1, 1), stride=(1, 1))
          (node_update): Conv1d(6656, 256, kernel_size=(1,), stride=(1,))
          (norm): GraphNorm(256)
          (dropout): Dropout(p=0.1, inplace=False)
          (xy_scores): Conv2d(4, 4, kernel_size=(1, 1), stride=(1, 1))
          (t_scores): Conv2d(2, 2, kernel_size=(1, 1), stride=(1, 1))
          (iou_scores): Conv2d(1, 4, kernel_size=(1, 1), stride=(1, 1))
        )
        (GAT_1): CausalIOU_XYTGATConv(
          (att_scores): Conv2d(512, 8, kernel_size=(1, 1), stride=(1, 1))
          (node_update): Conv1d(3328, 256, kernel_size=(1,), stride=(1,))
          (norm): GraphNorm(256)
          (dropout): Dropout(p=0.1, inplace=False)
          (xy_scores): Conv2d(4, 2, kernel_size=(1, 1), stride=(1, 1))
          (t_scores): Conv2d(2, 1, kernel_size=(1, 1), stride=(1, 1))
          (iou_scores): Conv2d(1, 2, kernel_size=(1, 1), stride=(1, 1))
        )
        (GAT_2): CausalIOU_XYTGATConv(
          (att_scores): Conv2d(512, 8, kernel_size=(1, 1), stride=(1, 1))
          (node_update): Conv1d(3328, 256, kernel_size=(1,), stride=(1,))
          (norm): GraphNorm(256)
          (dropout): Dropout(p=0.1, inplace=False)
          (xy_scores): Conv2d(4, 2, kernel_size=(1, 1), stride=(1, 1))
          (t_scores): Conv2d(2, 1, kernel_size=(1, 1), stride=(1, 1))
          (iou_scores): Conv2d(1, 2, kernel_size=(1, 1), stride=(1, 1))
        )
        (GAT_3): CausalIOU_XYTGATConv(
          (att_scores): Conv2d(512, 8, kernel_size=(1, 1), stride=(1, 1))
          (node_update): Conv1d(3328, 256, kernel_size=(1,), stride=(1,))
          (norm): GraphNorm(256)
          (dropout): Dropout(p=0.1, inplace=False)
          (xy_scores): Conv2d(4, 2, kernel_size=(1, 1), stride=(1, 1))
          (t_scores): Conv2d(2, 1, kernel_size=(1, 1), stride=(1, 1))
          (iou_scores): Conv2d(1, 2, kernel_size=(1, 1), stride=(1, 1))
        )
      )
    )




```python
expand_dims = nn.Conv1d(ns.shape[0], gat_channels, 1).to(conf.device)
ens = expand_dims(ns)
ens.shape
```




    torch.Size([256, 255])




```python
fixations.shape, bboxes.shape
```




    (torch.Size([255, 3, 1, 1]), torch.Size([4, 255]))




```python
graphs = grapher((ens, splits, fixations, bboxes)) # output is all selected residual graphs by residual_gat_levels
[g.shape for g in graphs]
```




    [torch.Size([256, 255])]



EuclideanT + IOU Models


```python
from grapher import IOU_EuclideanTGazeGrapher
```


```python
grapher = IOU_EuclideanTGazeGrapher(gat_conv=CausalIOU_EuclideanTGATConv,
                                layers=layers,
                                output_gat_levels=output_gat_levels,
                                residual_gat_levels=residual_gat_levels,
                                att_heads=att_heads,
                                gat_conv_args={'channels': gat_channels,
                                               'xy_heads_ratio': 0.2,
                                               't_heads_ratio': 0.1,
                                               'iou_heads_ratio': 0.2,
                                               'act_att': F.leaky_relu,
                                               'act_gat': F.elu,
                                               'norm': GraphNorm,
                                               'dropout': 0.1,
                                               'bias': True,
                                               'w_init_f': lambda w: nn.init.xavier_uniform_(w)},
                                device=conf.device)
grapher
```




    IOU_EuclideanTGazeGrapher(
      (gats): MarkovResidualSequential(
        (GAT_0): CausalIOU_EuclideanTGATConv(
          (att_scores): Conv2d(512, 16, kernel_size=(1, 1), stride=(1, 1))
          (node_update): Conv1d(6656, 256, kernel_size=(1,), stride=(1,))
          (norm): GraphNorm(256)
          (dropout): Dropout(p=0.1, inplace=False)
          (xy_scores): Conv2d(7, 4, kernel_size=(1, 1), stride=(1, 1))
          (t_scores): Conv2d(2, 2, kernel_size=(1, 1), stride=(1, 1))
          (iou_scores): Conv2d(1, 4, kernel_size=(1, 1), stride=(1, 1))
        )
        (GAT_1): CausalIOU_EuclideanTGATConv(
          (att_scores): Conv2d(512, 8, kernel_size=(1, 1), stride=(1, 1))
          (node_update): Conv1d(3328, 256, kernel_size=(1,), stride=(1,))
          (norm): GraphNorm(256)
          (dropout): Dropout(p=0.1, inplace=False)
          (xy_scores): Conv2d(7, 2, kernel_size=(1, 1), stride=(1, 1))
          (t_scores): Conv2d(2, 1, kernel_size=(1, 1), stride=(1, 1))
          (iou_scores): Conv2d(1, 2, kernel_size=(1, 1), stride=(1, 1))
        )
        (GAT_2): CausalIOU_EuclideanTGATConv(
          (att_scores): Conv2d(512, 8, kernel_size=(1, 1), stride=(1, 1))
          (node_update): Conv1d(3328, 256, kernel_size=(1,), stride=(1,))
          (norm): GraphNorm(256)
          (dropout): Dropout(p=0.1, inplace=False)
          (xy_scores): Conv2d(7, 2, kernel_size=(1, 1), stride=(1, 1))
          (t_scores): Conv2d(2, 1, kernel_size=(1, 1), stride=(1, 1))
          (iou_scores): Conv2d(1, 2, kernel_size=(1, 1), stride=(1, 1))
        )
        (GAT_3): CausalIOU_EuclideanTGATConv(
          (att_scores): Conv2d(512, 8, kernel_size=(1, 1), stride=(1, 1))
          (node_update): Conv1d(3328, 256, kernel_size=(1,), stride=(1,))
          (norm): GraphNorm(256)
          (dropout): Dropout(p=0.1, inplace=False)
          (xy_scores): Conv2d(7, 2, kernel_size=(1, 1), stride=(1, 1))
          (t_scores): Conv2d(2, 1, kernel_size=(1, 1), stride=(1, 1))
          (iou_scores): Conv2d(1, 2, kernel_size=(1, 1), stride=(1, 1))
        )
      )
    )




```python
expand_dims = nn.Conv1d(ns.shape[0], gat_channels, 1).to(conf.device)
ens = expand_dims(ns)
ens.shape
```




    torch.Size([256, 255])




```python
fixations.shape, bboxes.shape, splits, sum(splits)
```




    (torch.Size([255, 3, 1, 1]), torch.Size([4, 255]), [47, 155, 53], 255)




```python
graphs = grapher((ens, splits, fixations, bboxes)) # output is all selected residual graphs by residual_gat_levels
[g.shape for g in graphs]
```




    [torch.Size([256, 255]),
     torch.Size([256, 255]),
     torch.Size([256, 255]),
     torch.Size([256, 255])]



### Readout
<a id="readout"></a>

Since we are dealing with complete causal graphs --each node sees ALL of its predecessors and itself-- we developed a causal graph readout. All information is aggregated by the last node at each convolution. So, we take the last node of each graph selected by residual_gat_levels, concatenate them, and run them through linear layers, outputting the final predictions.


```python
from readout import CausalReadout
```


```python
sum([g.shape[0] for g in graphs])
```




    1024




```python
ro_channels=conf.ro_channels
ro_channels[0] = sum([g.shape[0] for g in graphs])
```


```python
causal_ro = CausalReadout(channels=ro_channels,
                          dropout=conf.ro_dropout,
                          device=conf.device)
```


```python
causal_ro
```




    CausalReadout(
      (act): Identity()
      (mlp): Sequential(
        (norm_input): GroupNorm(1, 1024, eps=1e-05, affine=True)
        (fc0): Conv1d(1024, 512, kernel_size=(1,), stride=(1,))
        (norm1): GroupNorm(1, 512, eps=1e-05, affine=True)
        (act1): GELU(approximate='none')
        (drop1): Dropout(p=0.15, inplace=False)
        (fc1): Conv1d(512, 256, kernel_size=(1,), stride=(1,))
        (norm2): GroupNorm(1, 256, eps=1e-05, affine=True)
        (act2): GELU(approximate='none')
        (drop2): Dropout(p=0.15, inplace=False)
        (fc2): Conv1d(256, 128, kernel_size=(1,), stride=(1,))
        (norm3): GroupNorm(1, 128, eps=1e-05, affine=True)
        (act3): GELU(approximate='none')
        (drop3): Dropout(p=0.15, inplace=False)
        (fc3): Conv1d(128, 64, kernel_size=(1,), stride=(1,))
        (norm4): GroupNorm(1, 64, eps=1e-05, affine=True)
        (act4): GELU(approximate='none')
        (drop4): Dropout(p=0.15, inplace=False)
        (prediction): Conv1d(64, 1, kernel_size=(1,), stride=(1,))
      )
    )




```python
predictions = causal_ro((graphs, splits))
predictions.shape, labels.shape
```




    (torch.Size([3]), torch.Size([3]))



#### Weighted Mean Readout (for ablation studies)


```python
from readout import WeightedMeanReadout
```


```python
[g.shape for g in graphs]
```




    [torch.Size([256, 255]),
     torch.Size([256, 255]),
     torch.Size([256, 255]),
     torch.Size([256, 255])]




```python
ro_channels[0] = graphs[0].shape[0]
ro_channels
```




    [256, 512, 256, 128, 64]




```python
ro = WeightedMeanReadout(ro_channels, device=conf.device)
```


```python
ro((graphs, splits)).shape
```




    torch.Size([3])



### Whole pipeline
<a id="full"></a>

Since we now have all individual components, we are ready to run the full model. with all possible GATConvs


```python
from model.foveal_gat import FovealGAT
```


```python
gat_conv_args={'channels': gat_channels,
               'att_heads': 8,
               'act_att': F.leaky_relu,
               'act_gat': F.elu,
               'norm': GraphNorm,
               'dropout': 0.1,
               'bias': True,
               'w_init_f': lambda w: nn.init.xavier_uniform_(w)}
```

Semantic Att


```python
model = FovealGAT(feat_extractor=feat_ext,
                    encoder=GazeSinCosPosEncoding(386, 16),
                    gat_conv=CausalGATConv,
                    gat_conv_args=gat_conv_args,
                    ro_hidden_features=ro_channels[1:],
                    device=conf.device)
```


```python
predictions = model((splits,
                     fixations.to(conf.device),
                     patches.to(conf.device),
                     labels.to(conf.device)))
predictions.shape, labels.shape
```




    (torch.Size([3]), torch.Size([3]))




```python
xyt_gat_conv_args={'channels': gat_channels,
                   'att_heads': 8,
                   'xy_heads_ratio': 0.2,
                   't_heads_ratio': 0.1,
                   'act_att': F.leaky_relu,
                   'act_gat': F.elu,
                   'norm': GraphNorm,
                   'dropout': 0.1,
                   'bias': True,
                   'w_init_f': lambda w: nn.init.xavier_uniform_(w)}
```

XYT Att


```python
model = FovealGAT(feat_extractor=feat_ext,
                    encoder=GazeSinCosPosEncoding(386, 16),
                    gat_conv=CausalXYTGATConv,
                    gat_conv_args=xyt_gat_conv_args,
                    ro_hidden_features=ro_channels[1:],
                    device=conf.device)
```


```python
predictions = model((splits,
                     fixations.to(conf.device),
                     patches.to(conf.device),
                     labels.to(conf.device)))
predictions.shape, labels.shape
```




    (torch.Size([3]), torch.Size([3]))




```python
euc_gat_conv_args={'channels': gat_channels,
                   'att_heads': 8,
                   'xy_heads_ratio': 0.2,
                   't_heads_ratio': 0.1,
                   'act_att': F.leaky_relu,
                   'act_gat': F.elu,
                   'norm': GraphNorm,
                   'dropout': 0.1,
                   'bias': True,
                   'w_init_f': lambda w: nn.init.xavier_uniform_(w)}
```

EuclideanT Att


```python
model = FovealGAT(feat_extractor=feat_ext,
                    encoder=GazeSinCosPosEncoding(386, 16),
                    gat_conv=CausalEuclideanTGATConv,
                    gat_conv_args=euc_gat_conv_args,
                    ro_hidden_features=ro_channels[1:],
                    device=conf.device)
```


```python
predictions = model((splits,
                     fixations.to(conf.device),
                     patches.to(conf.device),
                     labels.to(conf.device)))
predictions.shape, labels.shape
```




    (torch.Size([3]), torch.Size([3]))




```python
euc_gat_conv_args={'channels': gat_channels,
                   'att_heads': 8,
                   'xy_heads_ratio': 0.2,
                   't_heads_ratio': 0.1,
                   'iou_heads_ratio': 0.2,
                   'act_att': F.leaky_relu,
                   'act_gat': F.elu,
                   'norm': GraphNorm,
                   'dropout': 0.1,
                   'bias': True,
                   'w_init_f': lambda w: nn.init.xavier_uniform_(w)}
```

Iou_EuclideanT Att


```python
import torch
from gaze_datasets.reflacx import ReflacxGazePatchDataset
from gaze_datasets.dataset_tools import gaze_patch_dataset_collate
from feature_extraction.extractors import GazePatchFeatureExtractor
from node_encodings.sequence import GazeSinCosPosEncoding

gaze_dataset_pth = conf.reflacx_gaze_dataset_pth
desired_label = 'Atelectasis'
ds = ReflacxGazePatchDataset(gaze_dataset_pth, desired_label, target_fixation_keys=('t', 'x', 'y', 'top_left', 'bottom_right'))
splits, fixations, patches, labels = gaze_patch_dataset_collate((ds[33], ds[34], ds[0]))

feat_ext = GazePatchFeatureExtractor(conf.fe_tail_feat_ext,
                                     conf.fe_state_dict_pth,
                                     conf.fe_channels,
                                     conf.fe_act,
                                     conf.fe_norm,
                                     conf.fe_bias,
                                     conf.fe_w_init_f,
                                     conf.device)
feats = feat_ext(patches.to(conf.device))

max_len = conf.reflacx_max_len
seq_encoding_featcount = conf.seq_encoding_feat_count

seq_encoding = GazeSinCosPosEncoding(max_len, seq_encoding_featcount)
seqs = seq_encoding(splits, adjust_dim=True).to(conf.device)
fixations = fixations.to(conf.device)
#fixations, bboxes = torch.split(fixations, (3, 4), dim=1)

nodes = torch.cat([fixations, seqs, feats], 1).float()
ns = nodes.squeeze(-1).squeeze(-1).permute((1, 0))
bboxes = bboxes.squeeze(-1).squeeze(-1).transpose(1, 0)
ns.shape, bboxes.shape, sum([x.shape[1]for x in (seqs, fixations, feats)])
```




    (torch.Size([151, 255]), torch.Size([255, 4]), 151)




```python
conf.device
```




    'cuda:0'




```python
torch.cuda.empty_cache()
mem = torch.cuda.memory_allocated()
```


```python
model = FovealGAT(feat_extractor=feat_ext,
                  encoder=GazeSinCosPosEncoding(386, 16),
                  gat_conv=CausalIOU_EuclideanTGATConv,
                  output_gat_levels=[7],
                  gat_conv_args=euc_gat_conv_args,
                  readout=WeightedMeanReadout,
                  ro_hidden_features=ro_channels[1:],
                  device=conf.device)
```


```python
predictions = model((splits,
                     fixations.to(conf.device),
                     patches.to(conf.device),
                     labels.to(conf.device)))
predictions.shape, labels.shape
```




    (torch.Size([3]), torch.Size([3]))


