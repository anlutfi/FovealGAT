## Adding encodings for sequence position, [X, Y], and fixation duration


```python
import fovealgat_confs as conf
```

With 1X1 feature vectors representing each fixation gaze patch, we now need to add encodings for the fixation (X, Y) position within the CXR and time duration.

regular sine cosine positional encoding


```python
from sequence import GazeSinCosPosEncoding
```


```python
seq_encoding_featcount = 16
```


```python
gaze_dataset_pth = conf.reflacx_gaze_dataset_pth
```


```python
# get longest gaze sequence in dataset
import torch
from gaze_datasets.dataset_tools import max_fixation_sequence_len

max_len = max_fixation_sequence_len(gaze_dataset_pth)
max_len
```




    386



Load sample data


```python
from gaze_datasets.reflacx import ReflacxGazePatchDataset
from gaze_datasets.dataset_tools import gaze_patch_dataset_collate
from feature_extraction.extractors import GazePatchFeatureExtractor

desired_label = 'Atelectasis'
ds = ReflacxGazePatchDataset(gaze_dataset_pth, desired_label)
fs, gsp, labels = ds[33]
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
```


```python
seq_encoding = GazeSinCosPosEncoding(max_len, seq_encoding_featcount)
```

gaze sequences (graphs) are batched, so sequential positional encoding should be per batch item, so we need to pass the batch splits


```python
seqs = seq_encoding(splits, adjust_dim=True).to(conf.device)
splits, seqs.shape
```




    ([47, 155, 53], torch.Size([255, 16, 1, 1]))




```python
fixations = fixations.to(conf.device)
fixations.shape
```




    torch.Size([255, 3, 1, 1])




```python
feats.shape
```




    torch.Size([255, 128, 1, 1])



inserting encodings into gaze patch feature vector effectively turns it into the graph nodes


```python
nodes = torch.cat([fixations, seqs, feats], 1).float()
nodes.shape, sum([x.shape[1]for x in (seqs, fixations, feats)])
```




    (torch.Size([255, 147, 1, 1]), 147)


