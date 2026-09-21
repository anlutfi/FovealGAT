## GazeOnlyGat: Chest X-Rays (CXRs) as a sequence of gaze fixations

To the best of our knowledge, all multimodal machine learning models that use visual gaze data, use it to enrich the image. This, however, is not how our brains build the image's representation. We capture and interpret it via a sequence of eye fixations and movements.

So, in the interest of mimicking human reasoning, it seems logical to represent an image as a sequence of smaller patches, each captured by a fixation.

This work experiments with this concept in the specific context of chest x-rays.

### 1. Converting CXRs datasets to a gaze sequence

In order to run GazeOnlyGAT, we first need to convert regular CXR + Gaze datasets into Gaze Patch datasets. Each image needs to be transformed in a sequence of crops corresponding to the region of each gaze fixation.
[Dataset Conversion ReadMe](./gaze_dataset_convert/readme.html)

### 2. Loading new datasets to torch dataset classes

After conversion, we need to load the newly created gaze patch datasets into pytorch datasets. [Pytorch Datasets ReadMe](./gaze_datasets/readme.html)

### 3. Feature Extraction

We're using pretrained pyramid vision feature extraction, similar to gazegnn and vit. however, since we are working with small gaze patches, compared to the whole image, we need to limit the convolutions. Failing to do so, will try to extract features that need a larger size than the gazepatch can provide. [Feature Extraction ReadMe](./feature_extraction/readme.html)

### 4. Adding encodings for sequence position, [X, Y], and fixation duration (t)

With 1X1 feature vectors representing each fixation gaze patch, we now need to add encodings for the fixation's (X, Y) position within the CXR and time duration. [Encodings ReadMe](./node_encodings/readme.html)

### 5. GazeOnlyGAT

With the gaze patch features and encodings aggregated into graph nodes, we can feed it through our network. We'll go through each component. [GazeOnlyGAT ReadMe](./model/readme.html)

#### 5.1 GAT Layer

The first component is the Graph Attention Convolution Layer. [GAT Layer ReadMe](./model/readme.html#gat_layer)

#### 5.2 GazeGrapher

GazeGrapher is a module that aggregates several GAT layers in sequence, performing all graph convolutions before graph readout and prediction. [GazeGrapher ReadMe](./model/readme.html#grapher)

#### 5.3 Readout

Since we are dealing with complete causal graphs --each node sees ALL of its predecessors and itself-- we developed a causal graph readout. All information is aggregated by the last node at each convolution. So, we take the last node of each selected residual graph, concatenate them, and run them through a linear layer. [Readout ReadMe](./model/readme.html#readout)

#### 5.4 Whole pipeline

Since we now have all individual components, we are ready to run the full model. [Full Model ReadMe](./model/readme.html#full)

### 6. Gradient Test

We can now test the model's capacity of proper backpropagation


```python
import torch
if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'

print(device)
```

fetch sample data batch


```python
from gaze_datasets.reflacx import ReflacxGazePatchDataset
from gaze_datasets.dataset_tools import gaze_patch_dataset_collate

gaze_dataset_pth = '/media/antonio/ubuntail/Dropbox/education/phd/thesis/data/gaze_patch_datasets/reflacx/1_ang_res_256'
desired_label = 'Atelectasis'
ds = ReflacxGazePatchDataset(gaze_dataset_pth, desired_label)
fs, gsp, labels = ds[33]
splits, fixations, patches, labels = gaze_patch_dataset_collate((ds[33], ds[34], ds[0]))
```


```python
from torch import nn
```


```python
from model.gaze_only_gat import GazeOnlyGAT
from test_tools import mem_str
```


```python
model = GazeOnlyGAT(device=device)
```


```python
criterion = nn.L1Loss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

model.train()
optimizer.zero_grad()

h1 = model((splits, fixations.to(device), patches.to(device), labels.to(device)))

loss = criterion(h1, labels.to(device))

loss.backward()

for name, param in model.named_parameters():
    if param.requires_grad and param.grad is not None:
        print(f"{name} Grad Mean: {param.grad.mean().item()}")
print()

optimizer.step()

print('max: {}'.format(mem_str(torch.cuda.max_memory_allocated(device))))
print('curr: {}\n'.format(mem_str(torch.cuda.memory_allocated(device))))
```


```python
for name, _ in model.named_parameters():
    print(name)
```

### 7. Trainning
