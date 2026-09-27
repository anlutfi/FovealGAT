if __name__ == '__main__':

    import sys
    import os
    import json
    import torch
    from torch import nn

    from train_tools import get_loss, get_optim

    import torch
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'

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
    transform = nn.Identity()
    batch_size = 2
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
    in_channels = fe_channels[-1] + seq_encoding_feat_count + len(['x', 'y', 't'])
    grph_gat_conv = convs.CausalGATConv
    grph_channels = [in_channels, 256, 256, 256, 256]
    grph_residual_gat_levels = [False, True, False]
    grph_output_gat_levels = None
    grph_att_heads = [16, 8, 8, 8, 8]
    grph_act_att = F.leaky_relu
    grph_act_gat = F.elu
    grph_norm = GraphNorm
    grph_dropout = 0.15
    grph_bias = True
    grph_w_init_f = lambda w: nn.init.xavier_uniform_(w)

    #### Readout
    import model.readout as readouts
    readout = readouts.CausalReadout
    ro_channels = [-1, 512, 256, 128, 64]
    ro_hidden_features = ro_channels[1:]
    ro_out_channels = 1
    ro_hidden_act = nn.GELU
    ro_pred_act = nn.Identity
    ro_dropout = 0.15
    ro_norm = nn.BatchNorm1d
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
    output_pth_head = '_'.join(['reflacx',
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
        print('Beginning training session {} of {}\n'.format(session + 1, sessions))
        
        feat_ext = GazePatchFeatureExtractor(fe_tail_feat_ext,
                                            fe_channels,
                                            fe_act,
                                            fe_norm,
                                            fe_bias,
                                            fe_w_init_f,
                                            device)
        model = GazeOnlyGAT(feat_ext,
                            encoder,
                            grph_output_gat_levels,
                            grph_residual_gat_levels,
                            grph_gat_conv,
                            readout,
                            grph_channels,
                            grph_att_heads,
                            grph_act_att,
                            grph_act_gat,
                            grph_norm,
                            grph_dropout,
                            grph_bias,
                            ro_hidden_features,
                            ro_out_channels,
                            ro_hidden_act,
                            ro_pred_act,
                            ro_dropout,
                            ro_norm,
                            ro_bias,
                            grph_w_init_f,
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
            print('Session: {}  --  Epoch: {}'.format(session + 1, epoch + 1))
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
            for i, (splits, fixations, patches, labels) in enumerate(loader):
                optimizer.zero_grad()
                h = model((splits,
                        fixations.to(device),
                        patches.to(device),
                        labels.to(device)))
                loss = criterion(h, labels.to(device))

                stats['train'].append({'epoch': epoch,
                                    'predictions': h,
                                    'labels': labels,
                                    'loss': loss.item() * len(splits)})
                running_loss += loss * len(splits)
                sample_count += len(splits)
                
                loss.backward()
                optimizer.step()
            stats['epoch_train'] = {'total_loss': running_loss,
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
                    print('epoch: {} best val loss so far, saved model checkpoint'.format(epoch))

            stats['epoch_val'] = {'total_loss': running_loss,
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
            stats['session_test'] = {'total_loss': test_loss,
                                     'avg_loss': test_loss / test_samples}
            print("session: {}   total test loss: {:.4f}".format(session + 1, test_loss))
            print("session: {}     avg test loss: {:.4f}".format(session + 1, test_loss / test_samples))
            print()

        stats_pth = os.path.sep.join([session_dir, 'stats.pt'])
        torch.save(stats, stats_pth)
            
        