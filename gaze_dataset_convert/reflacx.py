import json
import os
from torchvision import transforms
import torch
from math import isnan
import numbers
import pandas as pd
import pydicom

from gaze_dataset_convert.conv_dataset_tools import pad_patch

label_domain = ['Airway wall thickening',
                'Atelectasis',
                'Consolidation',
                'Emphysema',
                'Enlarged cardiac silhouette',
                'Fibrosis',
                'Fracture',
                'Groundglass opacity',
                'Mass',
                'Nodule',
                'Pleural effusion',
                'Pleural thickening',
                'Pneumothorax',
                'Pulmonary edema',
                'Wide mediastinum',
                'Acute fracture',
                'Enlarged hilum',
                'Hiatal hernia',
                'High lung volume / emphysema',
                'Interstitial lung disease',
                'Lung nodule or mass',
                'Pleural abnormality',
                'Other']

def make_REFLACX_gazepatch_dataset(qtt_ang_res,
                                   img_resize,
                                   reflacx_dir,
                                   mimic_dir,
                                   dataset_path,
                                   pad=True,
                                   label_domain=label_domain):
    """generates new gazepatch dataset from original REFLACX.
    takes each reflacx sample and generates one image with param:qtt_ang_res angular resolutions of radius for each gaze fixation. saves these images in gaze_patches folder
    a list of fixations' x, y, delta_t and bouding box coords are saved in fixations folder
    ground truth is saved in labels folder. Original REFLACX attributes 0 to 5.0 values for the probability of each anomaly. This conversion regularizes this interval to [0, 1]

    before cropping each gaze patch, original CXR will be resizes to param:img_resize
    """
    dataset_path = os.path.sep.join([dataset_path, '{}_ang_res_{}'.format(qtt_ang_res, img_resize)])
    fixations_path = os.path.sep.join([dataset_path, 'fixations'])
    gaze_patches_path = os.path.sep.join([dataset_path, 'gaze_patches'])
    labels_path = os.path.sep.join([dataset_path, 'labels'])
    if not os.path.exists(fixations_path):
        os.makedirs(fixations_path)
    if not os.path.exists(gaze_patches_path):
        os.makedirs(gaze_patches_path)
    if not os.path.exists(labels_path):
        os.makedirs(labels_path)
    max_bbox_x = 0
    max_bbox_y = 0
    T = transforms.Compose([transforms.ToTensor(),
                            transforms.Resize(img_resize)])

    last_did = None
    cxr_img = None
    mtdt = [pd.read_csv(os.path.sep.join([reflacx_dir, 'metadata_phase_{}.csv'.format(i)])).sort_values(by='dicom_id') for i in range(1, 4)]
    for m in mtdt:
        for i, row in m.iterrows():
            did = row['dicom_id']
            rid = row['id']
            labels = {l: row[l] for l in label_domain if l in row}
            if last_did is None or did != last_did:
                print('dicom-id: {}'.format(did))
                try:
                    cxr_img = pydicom.dcmread(os.path.sep.join([mimic_dir, '{}.dcm'.format(did)])).pixel_array
                except FileNotFoundError:
                    print('skipping rid - {}: dicom image file missing'.format(rid))
                    continue
                orig_h, orig_w = cxr_img.shape
                cxr_img = T(cxr_img).float()
                _, h, w = cxr_img.shape
                minpx, maxpx = torch.min(cxr_img), torch.max(cxr_img)
                cxr_img -= minpx
                if minpx != maxpx: # prevent unlikely div/0
                    cxr_img /= (maxpx - minpx)

                last_did = did
                
            print('\treflacx-id: {}'.format(rid))
            

            out_path = os.path.sep.join([fixations_path, '{}.json'.format(rid)])
            out_gaze_path = os.path.sep.join([gaze_patches_path, rid])
            if not os.path.exists(out_gaze_path):
                os.makedirs(out_gaze_path)

            

            try:
                fixations = pd.read_csv(os.path.sep.join([reflacx_dir, rid, 'fixations.csv']))
            except FileNotFoundError:
                print('skipping rid - {}: fixations file missing'.format(rid))
                continue
            
            short_fixs = []
            for i, fixation in fixations.iterrows():
                t = fixation['timestamp_end_fixation'] - fixation['timestamp_start_fixation']
                x = int(fixation['x_position'])
                y = int(fixation['y_position'])

                if (not 0 <= x < orig_w) or (not 0 <= y < orig_h):
                    continue

                x_ang_res = fixation['angular_resolution_x_pixels_per_degree']
                y_ang_res = fixation['angular_resolution_y_pixels_per_degree']
            
                top_left = (max(0,
                                int(x - (qtt_ang_res * x_ang_res))),
                            max(0,
                                int(y - (qtt_ang_res * y_ang_res))))
                bottom_right = (min(orig_w,
                                    int(x + 1 + (qtt_ang_res * x_ang_res))),
                                min(orig_h,
                                    int(y + 1 + (qtt_ang_res * y_ang_res))))

                
                #normalize position and bounding box coords to [0, 1]
                x /= orig_w
                y /= orig_h

                top_left = (top_left[0] / orig_w, top_left[1] / orig_h)
                bottom_right = (bottom_right[0] / orig_w, bottom_right[1] / orig_h)

                short_fixs.append({'x': x,
                                'y': y,
                                't': t,
                                'top_left': top_left,
                                'bottom_right': bottom_right})

                #get gaze patch from resized image
                top_left = (int(top_left[0] * w), int(top_left[1] * h))
                bottom_right = (int(bottom_right[0] * w), int(bottom_right[1] * h))
                torch.save(cxr_img[:,
                                top_left[1]:bottom_right[1],
                                top_left[0]:bottom_right[0]],
                        os.path.sep.join([out_gaze_path, '{}.pt'.format(i)]))

                #get largest bounding box for padding
                max_bbox_x = max(max_bbox_x, bottom_right[0] - top_left[0])
                max_bbox_y = max(max_bbox_y, bottom_right[1] - top_left[1])

            with open(out_path, 'w') as fp:
                json.dump(short_fixs, fp)

            for k in labels:
                if isinstance(labels[k], numbers.Number) and not isnan(labels[k]):
                    labels[k] /= 5
            with open(os.path.sep.join([labels_path, '{}.json'.format(rid)]), 'w') as fp:
                json.dump(labels, fp)

    if pad:
        pad_sz = max(max_bbox_x, max_bbox_y)
        print('\n\npadding gaze patches to {}x{}'.format(pad_sz, pad_sz))
        pad_gaze_patches(pad_sz, gaze_patches_path)
    return max_bbox_x, max_bbox_y

def pad_gaze_patches(pad_sz, patches_path):
    for r_id in os.listdir(patches_path):
        pth = os.path.sep.join([patches_path, r_id])
        for i in os.listdir(pth):
            ptpth = os.path.sep.join([pth, i])
            patch = torch.load(ptpth)
            padded = pad_patch(patch, pad_sz)
            torch.save(padded, ptpth)