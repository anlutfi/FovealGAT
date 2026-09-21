import pandas as pd
import os
from glob import glob
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import json

from gaze_dataset_convert.conv_dataset_tools import pad_patch

def make_MIMIC_EYE_gazepatch_dataset(mimic_path,
                                     out_path,
                                     x_norm_radius,
                                     y_norm_radius,
                                     img_resize,
                                     padding=None):
    """Converts MIMIC Eye dataset into a gaze patch dataset
    param:mimic_path path with MIMIC Eye raw data
    param:out_path path to the generated gaze path dataset
    param:x_norm_radius half of the patch horizontal size, in proportion to whole image's width
    param:y_norm_radius half of the patch vetical size, in proportion to whole image's height
    param:img_resize resized image's size vefore cropping the patches
    param:padding final patch size, to be achieved with zero padding
    """
    print('Loading MIMIC fixations csv')
    csv = pd.read_csv(os.path.join(mimic_path, "gaze","fixations.csv"))

    out_dir_leaf = '{:.3f}_{:.3f}_pad{}__resize{}'.format(x_norm_radius,
                                                          y_norm_radius,
                                                          padding,
                                                          img_resize)
    out_path = os.path.sep.join([out_path, out_dir_leaf])

    fixations_path = os.path.sep.join([out_path, 'fixations'])
    gaze_patches_path = os.path.sep.join([out_path, 'gaze_patches'])
    labels_path = os.path.sep.join([out_path, 'labels'])

    print("creating output file structure")
    for pth in [fixations_path, gaze_patches_path, labels_path]:
        if not os.path.exists(pth):
            os.makedirs(pth)

    d_id_paths = []
    for label in ["CHF", "Normal", "pneumonia"]:
        d_id_paths.extend(glob(os.path.join(mimic_path, 'train', label, "*.jpg")))

    #resize = transforms.Resize(img_resize)
    T = transforms.Compose([transforms.ToTensor(),
                            transforms.Resize(img_resize)])

    for i, d_id_path in enumerate(d_id_paths):
        d_id = d_id_path.split("/")[-1].split(".jpg")[0]
        print('{}/{} -- generating gaze patches for DICOM-ID: {}'.format(i + 1, len(d_id_paths), d_id))
        with open(d_id_path, 'rb') as fp:
            img = Image.open(fp)
            img = np.array(img)

        d_id_gaze_patches_path = os.path.sep.join([gaze_patches_path, d_id])
        if not os.path.exists(d_id_gaze_patches_path):
            os.makedirs(d_id_gaze_patches_path)
        
        orig_h, orig_w = img.shape
        #timg = torch.from_numpy(np.expand_dims(np.expand_dims(img, axis=0), axis=0))
        img = T(img).float()
        #img = resize(timg).numpy()[0, 0, :, :]
        _, h, w = img.shape
        minpx, maxpx = torch.min(img), torch.max(img)
        img -= minpx
        if minpx != maxpx: # prevent unlikely div/0
            img /= (maxpx - minpx)


        px_radius_x = int(x_norm_radius * w)
        px_radius_y = int(y_norm_radius * h)

        label = d_id_path.split("/")[-2]

        fixations = []
        gazes = csv.loc[csv["DICOM_ID"] == d_id]
        for i in range(len(gazes)):
            if i == 0:
                t = gazes.iloc[i]["Time (in secs)"]
            else:
                t = gazes.iloc[i]["Time (in secs)"] - gazes.iloc[i-1]["Time (in secs)"]
            x = gazes.iloc[i]["X_ORIGINAL"]
            y = gazes.iloc[i]["Y_ORIGINAL"]

            norm_x = x / orig_w
            norm_y = y / orig_h               

            x = int(norm_x * w)
            y = int(norm_y * h)
                            

            top_left = max(0, x - px_radius_x), max(0, y - px_radius_y)
            bottom_right = min(w, x + px_radius_x), min(h, y + px_radius_y)
            #patch = np.copy(img[top_left[1]: bottom_right[1], top_left[0]: bottom_right[0]])
            patch = img[:,
                        top_left[1]: bottom_right[1],
                        top_left[0]: bottom_right[0]].clone()


            if padding is not None and padding > 0:
                patch = pad_patch(patch, padding)

            torch.save(patch,
                       os.path.sep.join([d_id_gaze_patches_path, '{}.pt'.format(i)]),
                       )
        
            fixations.append({'x': norm_x,
                            'y': norm_y,
                            't': t,
                            'top_left': (max(0.0, norm_x - x_norm_radius),
                                         max(0.0, norm_y - y_norm_radius)),
                            'bottom_right': (min(1.0, norm_x + x_norm_radius),
                                             min(1.0, norm_y + y_norm_radius))})

        with open(os.path.sep.join([fixations_path, '{}.json'.format(d_id)]), 'w') as fp:
            json.dump(fixations, fp)

        with open(os.path.sep.join([labels_path, '{}.json'.format(d_id)]), 'w') as fp:
            json.dump({'label': label}, fp)
