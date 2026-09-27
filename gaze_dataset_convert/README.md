## Converting CXR datasets to Gaze Patch Datasets

In order to run GazeOnlyGAT, we first need to convert regular CXR + Gaze datasets into Gaze Patch datasets. Each image needs to be transformed in a sequence of crops corresponding to the region of each gaze fixation


```python
import dataset_convert_confs as conf
```

### -REFLACX

In order to run this notebook, you need to import the Metadata class from [the reflacx_lib repo](https://github.com/anlutfi/reflacx_lib)


```python
reflacx_dir = conf.reflacx_dir
mimic_dir = conf.mimic_dir
full_meta_json = conf.full_meta_json
```


```python
from metadata import Metadata
```


```python
mtdt = Metadata(reflacx_dir,
                mimic_dir,
                full_meta_json,
                valid_img_only=True,
                valid_fixations_only=True)
```

How big should each fixation patch be?
REFLACX provides each fixation with their angular resolutions (pixels/degree). Tipically, in gaze heatmap generation, each fixation is seen as a gaussian centered on the gaze's xy coordinates, with standard deviation of the angular resolution.

So, we need to choose how many angular resolutions to each size we see in each fixation. Foveal vision is about 1 to 2 degrees.


```python
qtt_ang_res = 1
```

Pretrained feature extractors are expecting images rescaled to a specific resolution. pyramid vision expects a 224 image. However, this 224 image is a center-crop of a 256 pixel scaled image. Since center-crop does not change the size of features as resizing does, the resizing factor has to be 256


```python
img_resize = 256
```

set resulting dataset path


```python
dataset_path = conf.reflacx_gaze_patch_dataset_root
```

the function below does the whole operation. it returns the largest sizes for gaze patches


```python
from reflacx import make_REFLACX_gazepatch_dataset
```

setting next flag to false wont recreate the dataset in case this notebook is beeing run in sequence


```python
make_dataset = False
```


```python
if make_dataset:
    bb = make_REFLACX_gazepatch_dataset(qtt_ang_res, img_resize, mtdt, dataset_path, pad=False)
    bb
```


```python
if not make_dataset:
    bb = [31]
```

largest bb dims: 31 31, so all gaze patches will need to be padded to be 31 x 31.


```python
pad = max(bb)
pad
```


```python
import os
```


```python
#TODO un-hardcode
new_dataset_root = os.path.sep.join([dataset_path, '{}_ang_res_{}'.format(qtt_ang_res, img_resize)])
gaze_path_dir = os.path.sep.join([new_dataset_root, 'gaze_patches'])
```


```python
from reflacx import pad_gaze_patches
```


```python
if make_dataset:
    pad_gaze_patches(pad, gaze_path_dir)
```

### -MIMIC Eye

MIMIC does not provide an angular resolution for each fixation, so we will use the average angular resolution from REFLACX


```python
fixations_dir = os.path.sep.join([new_dataset_root, 'fixations'])
mimic_dir = conf.mimic_part_dir
```


```python
import json
import os
```


```python
sum_delta_x = 0
sum_delta_y = 0
count = 0
for sample in os.listdir(fixations_dir):
    with open(os.path.sep.join([fixations_dir, sample])) as fp:
        fixations = json.load(fp)
    for fixation in fixations:
        count += 1
        sum_delta_x += fixation['bottom_right'][0] - fixation['top_left'][0]
        sum_delta_y += fixation['bottom_right'][1] - fixation['top_left'][1]

avg_delta_x = sum_delta_x / (2 * count)
avg_delta_y = sum_delta_y / (2 * count)

avg_delta_x, avg_delta_y

```

pyramid vision expects xrays in 256 rescaling


```python
img_resize = 256
```

set output directory for new dataset


```python
out_path = conf.mimic_gaze_patch_dataset_root
```

set padding to be the same as in REFLACX gaze patch dataset


```python
padding = 31
```


```python
from mimic_eye import make_MIMIC_EYE_gazepatch_dataset
```

setting next flag to false wont recreate the dataset in case this notebook is beeing run in sequence


```python
make_dataset = False
```


```python
if make_dataset:
    make_MIMIC_EYE_gazepatch_dataset(mimic_dir,
                                     out_path,
                                     avg_delta_x,
                                     avg_delta_y,
                                     img_resize,
                                     padding)
```
