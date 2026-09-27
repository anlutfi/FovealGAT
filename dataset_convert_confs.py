import os

try:
    from private_confs import base_pth # root dir path containing the whole project
except ModuleNotFoundError:
    print("warning: you should create a private_confs module with base_pth = 'path/to/my/project/root/dir'")
    print("setting base_pth to '.'")
    base_pth = '.'

# raw data paths
reflacx_dir = os.path.sep.join([base_pth, 'data/reflacx'])
mimic_dir = os.path.sep.join([base_pth, 'data/mimic/reflacx_imgs'])
mimic_part_dir = os.path.sep.join([base_pth, 'data/mimic_part_jpg'])

# reflacx metadata json file
full_meta_json = os.path.sep.join([base_pth, 'reflacx_lib/full_meta.json'])

# gaze path datasets paths
reflacx_gaze_patch_dataset_root = os.path.sep.join([base_pth, 'data/gaze_patch_datasets/reflacx'])
reflacx_gaze_dataset_pth = os.path.sep.join([base_pth, 'data/gaze_patch_datasets/reflacx/1_ang_res_256'])
mimic_gaze_patch_dataset_root = os.path.sep.join([base_pth, 'data/gaze_patch_datasets/mimic_eye'])
mimic_gaze_dataset_pth = os.path.sep.join([base_pth, 'data/gaze_patch_datasets/mimic_eye/0.037_0.035_pad31__resize256'])