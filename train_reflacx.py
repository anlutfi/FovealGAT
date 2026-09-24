import sys
import os
import torch
import json

if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'

with open(sys.argv[1], 'r') as fp:
    args = json.load(fp)

from feature_extraction.extractors import GazePatchFeatureExtractor
feat_ext = GazePatchFeatureExtractor(args['fe_tail_feat_ext'],
                                     args['fe_channels'],
                                     args['fe_act'],
                                     args['fe_norm'],
                                     args['fe_bias'],
                                     args['fe_w_init_f'],
                                     device)

from node_encodings.sequence import GazeSinCosPosEncoding
encoder = GazeSinCosPosEncoding((args['reflacx_max_len']
                                 if args['dataset_type'] == 'reflacx'
                                 else args['mimic_max_len']),
                                args['seq_encoding_feat_count'])

from model.gaze_only_gat import GazeOnlyGAT
model = GazeOnlyGAT(feat_ext,
                    encoder,
                    args['grph_output_gat_levels'],
                    args['grph_residual_gat_levels'],
                    args['grph_gat_conv'],
                    args['readout'],
                    args['grph_channels'],
                    args['grph_att_heads'],
                    args['grph_act_att'],
                    args['grph_act_gat'],
                    args['grph_norm'],
                    args['grph_dropout'],
                    args['grph_bias'],
                    args['ro_hidden_features'],
                    args['ro_out_channels'],
                    args['ro_hidden_act'],
                    args['ro_pred_act'],
                    args['ro_dropout'],
                    args['ro_norm'],
                    args['ro_bias'],
                    args['grph_w_init_f'],
                    args['ro_w_init_f'],
                    args['device'])