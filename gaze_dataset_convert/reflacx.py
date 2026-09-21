import json
import os
from torchvision import transforms
import torch
from math import isnan
import numbers

from gaze_dataset_convert.conv_dataset_tools import pad_patch

def make_REFLACX_gazepatch_dataset(qtt_ang_res,
                                   img_resize,
                                   mtdt,
                                   dataset_path,
                                   pad=True):
    """generates new gazepatch dataset from original REFLACX.
    takes each reflacx sample and generates one image with param:qtt_ang_res angular resolutions of radius for each gaze fixation. saves these images in gaze_patches folder
    a list of fixations' x, y, delta_t and bouding box coords are saved in fixations folder
    ground truth is saved in labels folder. Original REFLACX attributes 0 to 5.0 values for the probability of each anomaly. This conversion regularizes this interval to [0, 1]

    before cropping each gaze patch, original CXR will be resizes to param:img_resize

    param:mtdt is a Metadata object from reflacx_lib
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

    for d_id in mtdt.list_dicom_ids():
        print('dicom-id: {}'.format(d_id))
        cxr_img = None
        for r_id in mtdt.list_reflacx_ids(d_id):
            print('\treflacx-id: {}'.format(r_id))
            try:
                sample = mtdt.get_sample_r(r_id)
            except KeyError:
                print('could not load sample {} (dicom-id:{})'.format(r_id, d_id))
                continue

            out_path = os.path.sep.join([fixations_path, '{}.json'.format(r_id)])
            out_gaze_path = os.path.sep.join([gaze_patches_path, r_id])
            if not os.path.exists(out_gaze_path):
                os.makedirs(out_gaze_path)

            #load and resize cxr
            if cxr_img is None:
                cxr_img = sample.get_dicom_img()
                orig_h, orig_w = cxr_img.shape
                cxr_img = T(cxr_img).float()
                _, h, w = cxr_img.shape
                minpx, maxpx = torch.min(cxr_img), torch.max(cxr_img)
                cxr_img -= minpx
                if minpx != maxpx: # prevent unlikely div/0
                    cxr_img /= (maxpx - minpx)

            fixations = sample.get_fixations()
            short_fixs = []
            for i, fixation in enumerate(fixations):
                t = fixation['timestamp_end_fixation'] - fixation['timestamp_start_fixation']
                x = int(fixation['x_position'])
                y = int(fixation['y_position'])
                
                # exclude fixations outside of xray borders
                if (not 0 <= x < orig_w) or (not 0 <= y < orig_h):
                    continue

                top_left = (max(0,
                                int(x - (qtt_ang_res * fixation['angular_resolution_x_pixels_per_degree']))),
                            max(0,
                                int(y - (qtt_ang_res * fixation['angular_resolution_y_pixels_per_degree']))))
                bottom_right = (min(orig_w,
                                    int(x + 1 + (qtt_ang_res * fixation['angular_resolution_x_pixels_per_degree']))),
                                min(orig_h,
                                    int(y + 1 + (qtt_ang_res * fixation['angular_resolution_y_pixels_per_degree']))))

                
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

            labels = sample.get_labels()
            for k in labels:
                if isinstance(labels[k], numbers.Number) and not isnan(labels[k]):
                    labels[k] /= 5
            with open(os.path.sep.join([labels_path, '{}.json'.format(r_id)]), 'w') as fp:
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