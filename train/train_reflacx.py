if __name__ == '__main__':

    import sys
    import os
    import json
    import torch
    from torch import nn
    import torchvision.transforms as T

    from train_tools import get_loss, get_optim, choose_device

    import torch
    
    device = choose_device()
    print('running on {}'.format(device))

    desired_label = sys.argv[1]
    with open(sys.argv[2], 'r') as fp:
        args = json.load(fp)

    assert desired_label in ('Consolidation',
                            'Groundglass opacity',
                            'Atelectasis',
                            'Pulmonary edema',
                            'Pneumothorax',
                            'Enlarged cardiac silhouette')

    sessions = args['sessions']
    epochs = args['epochs']
    lr = args['lr']
    pretrained_lr = args['pretrained_lr'] if 'pretrained_lr' in args else None
    store_best_model = args['store_best_model']
    stored_model_pth = args['stored_model_pth']

    criterion_clss = get_loss(args['loss'])
    optimizer_clss = get_optim(args['optimizer'])

    dataset_pth = args['dataset_pth']

    train_val_test_split = [0.7, 0.2, 0.1]
    transform = T.Compose([T.RandomHorizontalFlip(p=0.5), #TODO controversial
                           T.RandomRotation(degrees=7),
                           T.ColorJitter(brightness=0.05, contrast=0.05)])
    batch_size = 5
    acc_steps = 8
    dl_num_workers = 4

    #### feature extraction
    from feature_extraction.pretrained import pyramid_feature_extractor
    fe_tail_feat_ext = pyramid_feature_extractor
    fe_channels = [48, 128, 128, 128, 128]
    fe_act = nn.GELU
    fe_norm = nn.BatchNorm2d
    fe_bias = True
    fe_w_init_f = lambda w: nn.init.kaiming_uniform_(w, nonlinearity='relu')

    ##### node encodings
    from gaze_datasets.dataset_tools import max_fixation_sequence_len
    max_len = max_fixation_sequence_len(dataset_pth)
    seq_encoding_feat_count = 16

    #### Grapher
    from torch.nn import functional as F
    from torch_geometric.nn.norm import GraphNorm
    import model.gat_conv as convs
    from model.xyt_gat_conv import CausalXYTGATConv, CausalEuclideanTGATConv
    from model.residual_sequential import MarkovResidualSequential, DenseResidualSequential
        
    in_channels = fe_channels[-1] + seq_encoding_feat_count + len(['x', 'y', 't'])
    grph_gat_conv = CausalEuclideanTGATConv
    grph_res_seq = MarkovResidualSequential
    grph_layers = 8
    grph_channels = 512
    grph_residual_gat_levels = [False, True, True, True, True, True, True, True]
    grph_output_gat_levels = None
    grph_att_heads = [16, 8, 8, 8, 8, 8, 8, 8]
    grph_xy_heads_ratio=0.2
    grph_t_heads_ratio=0.1
    grph_act_att = F.leaky_relu
    grph_act_gat = F.elu
    grph_norm = GraphNorm
    grph_dropout = 0.3
    grph_bias = True
    grph_w_init_f = lambda w: nn.init.xavier_uniform_(w)

    gat_conv_args={'channels': grph_channels,
                   'xy_heads_ratio': grph_xy_heads_ratio,
                   't_heads_ratio': grph_t_heads_ratio,
                   'act_att': grph_act_att,
                   'act_gat': grph_act_gat,
                   'norm': grph_norm,
                   'dropout': grph_dropout,
                   'bias': grph_bias,
                   'w_init_f': grph_w_init_f}

    #### Readout
    import model.readout as readouts
    readout = readouts.CausalReadout
    ro_channels = [-1, 512, 256, 128, 64]
    ro_hidden_features = ro_channels[1:]
    ro_out_channels = 1
    ro_hidden_act = nn.GELU
    ro_pred_act = nn.Identity
    ro_dropout = 0.3
    ro_norm = nn.GroupNorm
    ro_bias = True
    ro_w_init_f = lambda w: nn.init.kaiming_uniform_(w, nonlinearity='relu')

    from feature_extraction.extractors import GazePatchFeatureExtractor
    from node_encodings.sequence import GazeSinCosPosEncoding
    from model.gaze_only_gat import GazeOnlyGAT
    from gaze_datasets import reflacx
    from torch.utils.data import DataLoader
    from datetime import datetime
    from gaze_datasets.dataset_tools import gaze_patch_dataset_collate

    def save_model_checkpoint(epoch,
                              model,
                              optimizer,
                              val_loss,
                              args_dump,
                              output_pth_head,
                              session_dir,
                              best=False):
        checkpoint = {'epoch': epoch,
                      'model': model.state_dict(),
                      'optimizer': optimizer.state_dict(),
                      'val_loss': val_loss,
                      'args_dump': args_dump}
        
        
        mname = '_'.join(['best' if best else 'last', output_pth_head])
        mname = '.'.join([mname, 'pt'])
        mname = mname.replace(' ', '_')
        torch.save(checkpoint, os.path.sep.join([session_dir, mname]))
    
    
    encoder = GazeSinCosPosEncoding(max_len, seq_encoding_feat_count)

    ds_name = dataset_pth.split(os.path.sep)[-1]
    output_pth_head = '_'.join(['xytreflacx',
                                ds_name,
                                desired_label,
                                criterion_clss.__name__,
                                optimizer_clss.__name__])
    with open('./train/train_reflacx.py', 'r', encoding='utf-8') as fp:
        args_dump = fp.read()
    train_time = datetime.now().strftime("%Y%m%d_%H%M")
    for session in range(sessions):
        session_dir = os.path.sep.join([stored_model_pth,
                                        output_pth_head,
                                        train_time,
                                        'session_{}'.format(session)])
        os.makedirs(session_dir, exist_ok=True)
        print('REFLACX -- {} -- Beginning training session {} of {}\n'.format(desired_label,
                                                                              session + 1,
                                                                              sessions))
        
        feat_ext = GazePatchFeatureExtractor(fe_tail_feat_ext,
                                             args['pretrained_sd'],
                                            fe_channels,
                                            fe_act,
                                            fe_norm,
                                            fe_bias,
                                            fe_w_init_f,
                                            device)
        model = GazeOnlyGAT(feat_ext,
                            encoder,
                            grph_gat_conv,
                            grph_res_seq,
                            grph_layers,
                            grph_output_gat_levels,
                            grph_residual_gat_levels,
                            grph_att_heads,
                            gat_conv_args,
                            readout,
                            ro_hidden_features,
                            ro_out_channels,
                            ro_hidden_act,
                            ro_pred_act,
                            ro_dropout,
                            ro_norm,
                            ro_bias,
                            ro_w_init_f,
                            device)
        
        dataset = reflacx.ReflacxGazePatchDataset(dataset_pth,
                                                desired_label,
                                                mode='train',
                                                train_val_test_split=train_val_test_split,
                                                transform=transform)

        criterion = criterion_clss()
        if pretrained_lr is not None:
            optimizer = optimizer_clss([{'params': [p for n, p in model.named_parameters()
                                                    if 'feat_ext.tail' not in n],
                                        'lr': lr},
                                        {'params': [p for n, p in model.named_parameters()
                                                    if 'feat_ext.tail' in n],
                                        'lr': pretrained_lr}])
        else:
            optimizer = optimizer_clss(model.parameters(), lr=lr)

        min_val_loss = torch.inf
        stats = {'train': [],
                'eval': [],
                'test': [],
                'epoch_train': {},
                'epoch_val': {},
                'session_test': {}}
        
        # TRAIN & VAL
        for epoch in range(epochs):
            
            # Train
            print('REFLACX -- {} -- Session: {}  --  Epoch: {} of {}'.format(desired_label,
                                                                             session + 1,
                                                                             epoch + 1, epochs))
            model.train()
            dataset.set_mode('train')
            loader = DataLoader(dataset,
                                batch_size=batch_size,
                                collate_fn=gaze_patch_dataset_collate,
                                shuffle=True,
                                num_workers=dl_num_workers,
                                pin_memory=True)
            running_loss = 0.0
            sample_count = 0
            acc_graphs = 0
            optimizer.zero_grad()
            for i, (splits, fixations, patches, labels) in enumerate(loader):
                h = model((splits,
                        fixations.to(device),
                        patches.to(device),
                        labels.to(device)))
                acc_graphs += len(splits)
                
                loss = criterion(h, labels.to(device)) / len(splits)
                
                stats['train'].append({'epoch': epoch,
                                    'predictions': h,
                                    'labels': labels,
                                    'loss': loss.item() * len(splits)})
                running_loss += loss * len(splits)
                sample_count += len(splits)
                
                loss.backward()
                if i != 0 and (acc_graphs >= batch_size * acc_steps or i == len(loader) - 1):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    acc_graphs = 0

            stats['epoch_train'][epoch] = {'total_loss': running_loss,
                                           'avg_loss': running_loss / sample_count}
            
            print('epoch: {}  total train loss: {}'.format(epoch + 1, running_loss))
            print('epoch: {}    avg train loss: {}'.format(epoch + 1, running_loss / sample_count))

            # Val
            model.eval()
            dataset.set_mode('val')
            loader = DataLoader(dataset,
                                batch_size=batch_size,
                                collate_fn=gaze_patch_dataset_collate,
                                num_workers=dl_num_workers,
                                pin_memory=True)
            running_loss = 0.0
            sample_count = 0
            with torch.no_grad():
                for splits, fixations, patches, labels in loader:
                    h = model((splits,
                            fixations.to(device),
                            patches.to(device),
                            labels.to(device)))
                    loss = criterion(h, labels.to(device))
                    stats['eval'].append({'epoch': epoch,
                                        'predictions': h,
                                        'labels': labels,
                                        'loss': loss.item() * len(splits)})
                    running_loss += loss * len(splits)
                    sample_count += len(splits)
            avg_val_loss = running_loss / sample_count
            if epoch == epochs - 1:
                #TODO run test split
                save_model_checkpoint(epoch + 1,
                                      model,
                                      optimizer,
                                      avg_val_loss,
                                      args_dump,
                                      output_pth_head,
                                      session_dir,
                                      best=avg_val_loss < min_val_loss)
            elif avg_val_loss < min_val_loss:
                min_val_loss = avg_val_loss
                if store_best_model:
                    save_model_checkpoint(epoch + 1,
                                        model,
                                        optimizer,
                                        min_val_loss,
                                        args_dump,
                                        output_pth_head,
                                        session_dir,
                                        best=True)
                    print('epoch: {} best val loss so far, saved model checkpoint'.format(epoch + 1))

            stats['epoch_val'][epoch] = {'total_loss': running_loss,
                                         'avg_loss': running_loss / sample_count}
            print("epoch: {}   total val loss: {:.4f}".format(epoch + 1, running_loss))
            print("epoch: {}     avg val loss: {:.4f}".format(epoch + 1, avg_val_loss))
            print()

        # TEST for current session
        model.eval()
        test_loss = 0.0
        test_samples = 0

        dataset.set_mode('test')
        loader = DataLoader(dataset,
                            batch_size=batch_size,
                            collate_fn=gaze_patch_dataset_collate,
                            num_workers=dl_num_workers,
                            pin_memory=True)

        with torch.no_grad():
            
            for splits, fixations, patches, labels in loader:
                h = model((splits,
                        fixations.to(device),
                        patches.to(device),
                        labels.to(device)))
                loss = criterion(h, labels.to(device))

                stats['test'].append({'session': session,
                                    'predictions': h,
                                    'labels': labels,
                                    'loss': loss.item() * len(splits)})
                test_loss += loss.item() * len(splits)
                test_samples += len(splits)
            stats['session_test'][session] = {'total_loss': test_loss,
                                              'avg_loss': test_loss / test_samples}
            print("session: {}   total test loss: {:.4f}".format(session + 1, test_loss))
            print("session: {}     avg test loss: {:.4f}".format(session + 1, test_loss / test_samples))
            print()

        stats_pth = os.path.sep.join([session_dir, 'stats.pt'])
        torch.save(stats, stats_pth)
            
        