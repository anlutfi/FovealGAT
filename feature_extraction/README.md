## Feature Extraction


```python
import fovealgat_confs as conf
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




    ([47, 155, 53],
     255,
     torch.Size([255, 3, 1, 1]),
     torch.Size([255, 1, 31, 31]),
     torch.Size([3]))




```python
from feature_extraction.extractors import GazePatchFeatureExtractor
```


```python
feat_ext = GazePatchFeatureExtractor(conf.fe_tail_feat_ext,
                                     conf.fe_state_dict_pth,
                                     conf.fe_channels,
                                     conf.fe_act,
                                     conf.fe_norm,
                                     conf.fe_bias,
                                     conf.fe_w_init_f,
                                     conf.device)
```


```python
feat_ext
```




    GazePatchFeatureExtractor(
      (tail): Stem(
        (convs): Sequential(
          (0): Conv2d(3, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
          (1): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): GELU(approximate='none')
          (3): Conv2d(24, 48, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
          (4): BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (5): GELU(approximate='none')
          (6): Conv2d(48, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (7): BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (head): Sequential(
        (norm_0): BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (act_0): GELU(approximate='none')
        (conv_1): Conv2d(48, 128, kernel_size=(3, 3), stride=(1, 1))
        (norm_1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (act_1): GELU(approximate='none')
        (conv_2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1))
        (norm_2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (act_2): GELU(approximate='none')
        (conv_3): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1))
        (norm_3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (act_3): GELU(approximate='none')
        (conv_4): Conv2d(128, 128, kernel_size=(2, 2), stride=(1, 1))
        (norm_4): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (act_4): GELU(approximate='none')
      )
    )



The pyramid feature extractor (tail) does not reduce the input to a 1X1 feature vector. So we add extra Conv2d to further reduce each gaze patch (head)


```python
feats = feat_ext(ps.to(conf.device))
feats.shape
```




    torch.Size([255, 128, 1, 1])



### ConvCoord feature extraction

This feature extractor adds channels to the foveal patch for XY position within the original image and tixation duration (T)


```python
from extractors import GazePatchCoordConvFeatureExtractor
```


```python
from gaze_datasets.reflacx import ReflacxGazePatchDataset
ds = ReflacxGazePatchDataset(gazedataset_pth, desired_label)
fs, gsp, labels = ds[33]
from gaze_datasets.dataset_tools import gaze_patch_dataset_collate
ls, fs, ps, labels = gaze_patch_dataset_collate((ds[33], ds[34], ds[0]))
ls, sum(ls), fs.shape, ps.shape, labels.shape
```




    ([47, 155, 53],
     255,
     torch.Size([255, 3, 1, 1]),
     torch.Size([255, 1, 31, 31]),
     torch.Size([3]))




```python
ccfe = GazePatchCoordConvFeatureExtractor()
ccfe
```




    GazePatchCoordConvFeatureExtractor(
      (head): Sequential(
        (conv_0): Conv2d(4, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        (norm_0): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (act_0): GELU(approximate='none')
        (conv_1): Conv2d(24, 48, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        (norm_1): BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (act_1): GELU(approximate='none')
        (conv_2): Conv2d(48, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (norm_2): BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (act_2): GELU(approximate='none')
        (conv_3): Conv2d(48, 128, kernel_size=(3, 3), stride=(1, 1))
        (norm_3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (act_3): GELU(approximate='none')
        (conv_4): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1))
        (norm_4): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (act_4): GELU(approximate='none')
        (conv_5): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1))
        (norm_5): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (act_5): GELU(approximate='none')
        (conv_6): Conv2d(128, 128, kernel_size=(2, 2), stride=(1, 1))
        (norm_6): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (act_6): GELU(approximate='none')
      )
    )




```python
ccfe((ps, fs)).shape
```




    torch.Size([255, 128, 1, 1])


