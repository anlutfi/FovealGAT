### Adding encodings for sequence position, [X, Y], and fixation duration

With 1X1 feature vectors representing each fixation gaze patch, we now need to add encodings for the fixation (X, Y) position within the CXR and time duration.

regular sine cosine positional encoding


```python
from sequence import GazeSinCosPosEncoding
```


```python
seq_encoding_featcount = 16
```


```python
gazedataset_pth = '/media/antonio/ubuntail/Dropbox/education/phd/thesis/data/gaze_patch_datasets/reflacx/1_ang_res_256'
```


```python
# get longest gaze sequence in dataset
import json
import os

fix_dir = os.path.sep.join([gazedataset_pth, 'fixations'])

max_len = 0
for rid in os.listdir(fix_dir):
    pth = os.path.sep.join([fix_dir, rid])
    with open(pth) as fp:
        max_len = max(max_len, len(json.load(fp)))

max_len
```




    386




```python
import torch
if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'

print(device)
```

Load sample data


```python
from gaze_datasets.reflacx import ReflacxGazePatchDataset
from gaze_datasets.dataset_tools import gaze_patch_dataset_collate
from feature_extraction.extractors import GazePatchFeatureExtractor

gaze_dataset_pth = '/media/antonio/ubuntail/Dropbox/education/phd/thesis/data/gaze_patch_datasets/reflacx/1_ang_res_256'
desired_label = 'Atelectasis'
ds = ReflacxGazePatchDataset(gaze_dataset_pth, desired_label)
fs, gsp, labels = ds[33]
splits, fixations, patches, labels = gaze_patch_dataset_collate((ds[33], ds[34], ds[0]))

feat_ext = GazePatchFeatureExtractor().to(device)
feats = feat_ext(patches.to(device))
```


```python
seq_encoding = GazeSinCosPosEncoding(max_len, seq_encoding_featcount)
```

gaze sequences (graphs) are batched, so sequential positional encoding should be per batch item, so we need to pass the batch splits


```python
seqs = seq_encoding(splits, adjust_dim=True).to(device)
splits, seqs.shape
```


```python
fixations = fixations.to(device)
fixations.shape
```


```python
feats.shape
```


```python
fixations.device
```

inserting encodings into gaze patch feature vector effectively turns it into the graph nodes


```python
nodes = torch.cat([fixations, seqs, feats], 1).float()
nodes.shape, sum([x.shape[1]for x in (seqs, fixations, feats)])
```
