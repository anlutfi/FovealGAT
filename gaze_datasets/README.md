### Loading Gaze Patch Datasets

After conversion, we need to load the newly created gaze patch datasets into pytorch datasets.

#### REFLACX


```python
from reflacx import ReflacxGazePatchDataset
```


```python
gaze_dataset_pth = '/media/antonio/ubuntail/Dropbox/education/phd/thesis/data/gaze_patch_datasets/reflacx/1_ang_res_256'
desired_label = 'Atelectasis'
```


```python
ds = ReflacxGazePatchDataset(gaze_dataset_pth, desired_label)
```


```python
fs, gsp, labels = ds[33]
```


```python
import matplotlib.pyplot as plt
```


```python
plt.imshow(gsp[13])
```

to make a data loader, we first need a collate function, as the gaze patch sequences vary in size


```python
from dataset_tools import gaze_patch_dataset_collate
```


```python
ls, fs, ps, labels = gaze_patch_dataset_collate((ds[33], ds[34], ds[0]))
```


```python
ls, sum(ls)
```


```python
fs.shape, ps.shape, labels.shape
```

now for the DataLoader itself


```python
from torch.utils.data import DataLoader
```


```python
loader = DataLoader(ds,
                    batch_size=6,
                    collate_fn=gaze_patch_dataset_collate,
                    num_workers=4,
                    pin_memory=True)
```


```python
b = None
for batch in loader:
    b = batch
    break
```


```python
ls, fs, ps, labels = b
```


```python
ls, sum(ls)
```


```python
fs.shape, ps.shape, labels.shape
```

#### MIMIC EYE


```python
from matplotlib import pyplot as plt
```


```python
from mimic_eye import MimicEyeGazePatchDataset
```


```python
gaze_dataset_pth = '/media/antonio/ubuntail/Dropbox/education/phd/thesis/data/gaze_patch_datasets/mimic_eye/0.037_0.035_pad31__resize256'
desired_label = 'pneumonia'
```


```python
ds = MimicEyeGazePatchDataset(gaze_dataset_pth, desired_label)
```


```python
fs, gsp, labels = ds[33]
```


```python
fs.shape, gsp.shape, labels.shape
```


```python
plt.imshow(gsp[8])
```

now for the data loader, identical to the reflacx case


```python
from torch.utils.data import DataLoader

from dataset_tools import gaze_patch_dataset_collate
```


```python
loader = DataLoader(ds,
                    batch_size=6,
                    collate_fn=gaze_patch_dataset_collate,
                    num_workers=4,
                    pin_memory=True)
```


```python
b = None
for batch in loader:
    b = batch
    break
```


```python
ls, fs, ps, labels = b
```


```python
ls, sum(ls)
```


```python
fs.shape, ps.shape, labels.shape
```
