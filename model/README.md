### GazeOnlyGAT: components overview


```python
import only_gaze_confs as conf
```

With the gaze patch features and encodings aggregated into graph nodes, we can feed it through our network. We'll go through each component.


```python
import torch
if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'

print(device)
```

Fetch sample batch


```python
from gaze_datasets.reflacx import ReflacxGazePatchDataset
from gaze_datasets.dataset_tools import gaze_patch_dataset_collate
from feature_extraction.extractors import GazePatchFeatureExtractor
from node_encodings.sequence import GazeSinCosPosEncoding

gaze_dataset_pth = conf.reflacx_gaze_dataset_pth
desired_label = 'Atelectasis'
ds = ReflacxGazePatchDataset(gaze_dataset_pth, desired_label)
splits, fixations, patches, labels = gaze_patch_dataset_collate((ds[33], ds[34], ds[0]))

feat_ext = GazePatchFeatureExtractor().to(device)
feats = feat_ext(patches.to(device))

max_len = 386
seq_encoding_featcount = 16
seq_encoding = GazeSinCosPosEncoding(max_len, seq_encoding_featcount)
seqs = seq_encoding(splits, adjust_dim=True).to(device)
fixations = fixations.to(device)

nodes = torch.cat([fixations, seqs, feats], 1).float()
nodes.shape, sum([x.shape[1]for x in (seqs, fixations, feats)])
```


```python
ns = nodes.squeeze(-1).squeeze(-1).permute((1, 0))
```

#### GAT Layer
<a id="gat_layer"></a>

The first component is the Graph Attention Convolution Layer.


```python
from model.GATConv import CausalGATConv
```

set number of attention heads for the layer


```python
att_heads = 4
```


```python
gcnn = CausalGATConv(ns.shape[0],
               ns.shape[0],
               att_heads,
               device=device)
```


```python
conv_nodes = gcnn((ns, splits))
conv_nodes.shape
```

#### GazeGrapher
<a id="grapher"></a>

GazeGrapher is a module that aggregates several GAT layers in sequence, performing all graph convolutions before graph readout and prediction.

PyTorch's native nn.Sequential does not keep residual tensors between its internal steps, so we need to extend it. Graph readout and predictions will need residual, intermediary, graphs


```python
from residual_sequential import ResidualSequential
```

We need input sizes for each of the graphs GAT Layers, and the residual layers' indices and GAT conv class of choice


```python
channels = [67, 134, 134, 134, 67] # input size and 4 GAT convolution layers
residual_gat_levels = [1, 2, 4]
gat_conv = CausalGATConv
```


```python
from grapher import GazeGrapher
```


```python
grapher = GazeGrapher(channels=channels,
                      readout_gat_levels=residual_gat_levels,
                      device=device)
grapher
```


```python
graphs = grapher((ns, splits)) # output is all selected residual graphs by residual_gat_levels
[g.shape for g in graphs]
```

#### Readout
<a id="readout"></a>

Since we are dealing with complete causal graphs --each node sees ALL of its predecessors and itself-- we developed a causal graph readout. All information is aggregated by the last node at each convolution. So, we take the last node of each graph selected by residual_gat_levels, concatenate them, and run them through linear layers, outputting the final predictions.


```python
from readout import CausalReadout
```


```python
causal_ro = CausalReadout(sum([g.shape[0] for g in graphs]),
                          device=device)
```


```python
predictions = causal_ro((graphs, splits))
predictions.shape, labels.shape
```

#### Whole pipeline
<a id="full"></a>

Since we now have all individual components, we are ready to run the full model.


```python
from gaze_only_gat import GazeOnlyGAT
```


```python
model = GazeOnlyGAT(device=device)
```


```python
predictions = model((splits, fixations.to(device), patches.to(device), labels.to(device)))
predictions.shape, labels.shape
```
