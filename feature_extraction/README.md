## Feature Extraction


```python
import only_gaze_confs as conf
```


```python
gazedataset_pth = conf.reflacx_gaze_dataset_pth
desired_label = 'Atelectasis'
```


```python
from gaze_datasets.reflacx import ReflacxGazePatchDataset
ds = ReflacxGazePatchDataset(gazedataset_pth, desired_label)
fs, gsp, labels = ds[33]
from gaze_datasets.dataset_tools import gaze_patch_dataset_collate
ls, fs, ps, labels = gaze_patch_dataset_collate((ds[33], ds[34], ds[0]))
ls, sum(ls), fs.shape, ps.shape, labels.shape
```


```python
from feature_extraction.extractors import GazePatchFeatureExtractor
```


```python
feat_ext = GazePatchFeatureExtractor(conf.fe_tail_feat_ext,
                                     conf.fe_channels,
                                     conf.fe_act,
                                     conf.fe_norm,
                                     conf.fe_bias,
                                     conf.fe_w_init_f,
                                     conf.device)
```

The pyramid feature extractor (tail) does not reduce the input to a 1X1 feature vector. So we add extra Conv2d to further reduce each gaze patch (head)


```python
feat_ext
```


```python
feats = feat_ext(ps.to(conf.device))
feats.shape
```
