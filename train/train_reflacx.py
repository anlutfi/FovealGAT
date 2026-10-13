if __name__ == '__main__':

    import sys
    import os
    import json
    import torch
    from torch import nn
    import torchvision.transforms as T

    from train_tools import get_loss, get_optim, get_sched, choose_device, ArgParser

    import torch
    
    device = choose_device()
    print('running on {}'.format(device))

    desired_label = sys.argv[1]
    with open(sys.argv[2], 'r') as fp:
        train_args = json.load(fp)
    with open(sys.argv[3], 'r') as fp:
        model_args_d = json.load(fp)
        model_args = ArgParser(model_args_d)
        

    assert desired_label in ('Consolidation',
                            'Groundglass opacity',
                            'Atelectasis',
                            'Pulmonary edema',
                            'Pneumothorax',
                            'Enlarged cardiac silhouette')

    sessions = train_args['sessions']
    epochs = train_args['epochs']
    lr = train_args['lr']
    pretrained_lr = train_args['pretrained_lr'] if 'pretrained_lr' in train_args else None
    weight_decay = train_args['weight_decay'] if 'weight_decay' in train_args else 0.0
    store_best_model = train_args['store_best_model']
    stored_model_pth = train_args['stored_model_pth']

    criterion_clss = get_loss(train_args['loss'])
    optimizer_clss = get_optim(train_args['optimizer'])
    scheduler_clss = get_sched(train_args['scheduler'])

    if scheduler_clss is not None:
        epoch_lr_restart = train_args['epoch_lr_restart']
        lr_min = train_args['lr_min']
    else:
        epoch_lr_restart = None
        lr_min = None

    dataset_pth = train_args['dataset_pth']
    gaze_noise = train_args['gaze_noise']

    #dataloader
    train_val_test_split = model_args('train_val_test_split')
    transform = model_args('transform')
    batch_size = model_args('batch_size')
    acc_steps = model_args('acc_steps')
    dl_num_workers = model_args('dl_num_workers')

    #feature extractor
    fe_tail_feat_ext = model_args('fe_tail_feat_ext')
    fe_channels = model_args('fe_channels')
    fe_act = model_args('fe_act')
    fe_norm = model_args('fe_norm')
    fe_bias = model_args('fe_bias')
    fe_w_init_f = model_args('fe_w_init_f')
    
    ##### node encodings
    from gaze_datasets.dataset_tools import max_fixation_sequence_len
    max_len = max_fixation_sequence_len(dataset_pth)
    seq_encoding_feat_count = model_args('seq_encoding_feat_count')
    
    #### grapher
    in_channels = fe_channels[-1] + seq_encoding_feat_count + len(['x', 'y', 't'])
    grph_gat_conv = model_args('grph_gat_conv')
    grph_res_seq = model_args('grph_res_seq')
    grph_layers = model_args('grph_layers')
    grph_channels = model_args('grph_channels')
    grph_residual_gat_levels = model_args('grph_residual_gat_levels')
    grph_output_gat_levels = model_args('grph_output_gat_levels')
    grph_att_heads = model_args('grph_att_heads')
    grph_xy_heads_ratio = model_args('grph_xy_heads_ratio')
    grph_t_heads_ratio = model_args('grph_t_heads_ratio')
    grph_iou_heads_ratio = model_args('grph_iou_heads_ratio')
    grph_self_edges = model_args("grph_self_edges")
    grph_mask_out_zeros = model_args("grph_mask_out_zeros")
    grph_act_att = model_args('grph_act_att')
    grph_act_gat = model_args('grph_act_gat')
    grph_norm = model_args('grph_norm')
    grph_dropout = model_args('grph_dropout')
    grph_bias = model_args('grph_bias')
    grph_w_init_f = model_args('grph_w_init_f')

    gat_conv_args={'channels': grph_channels,
                   'act_att': grph_act_att,
                   'act_gat': grph_act_gat,
                   'norm': grph_norm,
                   'dropout': grph_dropout,
                   'bias': grph_bias,
                   'w_init_f': grph_w_init_f}
    if grph_xy_heads_ratio is not None:
        gat_conv_args['xy_heads_ratio'] = grph_xy_heads_ratio
    if grph_t_heads_ratio is not None:
        gat_conv_args['t_heads_ratio'] = grph_t_heads_ratio
        
    if grph_iou_heads_ratio is not None and grph_iou_heads_ratio > 0:
        gat_conv_args['iou_heads_ratio'] = grph_iou_heads_ratio
        gat_conv_args['self_edges'] = grph_self_edges
        gat_conv_args['mask_out_zeros'] = grph_mask_out_zeros
        iou = True
    else:
        iou = False
    
    readout = model_args('readout')
    ro_channels = model_args('ro_channels')
    ro_out_channels = model_args('ro_out_channels')
    ro_hidden_act = model_args('ro_hidden_act')
    ro_pred_act = model_args('ro_pred_act')
    ro_dropout = model_args('ro_dropout')
    ro_norm = model_args('ro_norm')
    ro_bias = model_args('ro_bias')
    ro_w_init_f = model_args('ro_w_init_f')

    ro_hidden_features = ro_channels[1:]

    from model.foveal_gat import FovealGAT
        
       
    from feature_extraction.extractors import GazePatchFeatureExtractor
    from node_encodings.sequence import GazeSinCosPosEncoding
    from gaze_datasets import reflacx
    from torch.utils.data import DataLoader
    from datetime import datetime
    from gaze_datasets.dataset_tools import gaze_patch_dataset_collate

    def save_model_checkpoint(epoch,
                              model,
                              optimizer,
                              val_loss,
                              args_dump,
                              train_args,
                              model_args_d,
                              output_pth_head,
                              session_dir,
                              best=False):
        checkpoint = {'epoch': epoch,
                      'model': model.state_dict(),
                      'optimizer': optimizer.state_dict(),
                      'val_loss': val_loss,
                      'script_dump': args_dump,
                      'train_args': train_args,
                      'model_args': model_args_d}
        
        
        mname = '_'.join(['best' if best else 'last', output_pth_head])
        mname = '.'.join([mname, 'pt'])
        mname = mname.replace(' ', '_')
        mpath = os.path.sep.join([session_dir, mname])
        torch.save(checkpoint, mpath)
        return mpath
    
    
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
        last_mpath = None
        best_mpath = None
        session_dir = os.path.sep.join([stored_model_pth,
                                        output_pth_head,
                                        train_time,
                                        'session_{}'.format(session)])
        os.makedirs(session_dir, exist_ok=True)
        print('REFLACX -- {} -- Beginning training session {} of {}\n'.format(desired_label,
                                                                              session + 1,
                                                                              sessions))
        
        feat_ext = GazePatchFeatureExtractor(fe_tail_feat_ext,
                                             train_args['pretrained_sd'],
                                             fe_channels,
                                             fe_act,
                                             fe_norm,
                                             fe_bias,
                                             fe_w_init_f,
                                             device)
        model = FovealGAT(feat_ext,
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
        
        tfks = (('x', 'y', 't', 'top_left', 'bottom_right')
                if iou
                else ('x', 'y', 't'))
        dataset = reflacx.ReflacxGazePatchDataset(dataset_pth,
                                                  desired_label,
                                                  target_fixation_keys=tfks,
                                                  mode='train',
                                                  gaze_noise=gaze_noise,
                                                  train_val_test_split=train_val_test_split,
                                                  transform=transform)

        criterion = criterion_clss()
        if pretrained_lr is not None:
            optimizer = optimizer_clss([{'params': [p for n, p in model.named_parameters()
                                                    if 'feat_ext.tail' not in n],
                                        'lr': lr},
                                        {'params': [p for n, p in model.named_parameters()
                                                    if 'feat_ext.tail' in n],
                                        'lr': pretrained_lr}],
                                       weight_decay=weight_decay)
        else:
            optimizer = optimizer_clss(model.parameters(), lr=lr, weight_decay=weight_decay)

        scheduler = None
        if scheduler_clss is not None:
            scheduler = scheduler_clss(optimizer,
                                       T_0=epoch_lr_restart,
                                       T_mult=1,
                                       eta_min=lr_min)

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
            print("epoch: {}   total val loss: {:.4f}".format(epoch + 1, running_loss))
            print("epoch: {}     avg val loss: {:.4f}".format(epoch + 1, avg_val_loss))
            if epoch == epochs - 1:
                last_mpath = save_model_checkpoint(epoch + 1,
                                                   model,
                                                   optimizer,
                                                   avg_val_loss,
                                                   args_dump,
                                                   train_args,
                                                   model_args_d,
                                                   output_pth_head,
                                                   session_dir,
                                                   best=avg_val_loss < min_val_loss)
                if avg_val_loss < min_val_loss:
                    best_mpath = last_mpath
            elif avg_val_loss < min_val_loss:
                min_val_loss = avg_val_loss
                if store_best_model:
                    best_mpath = save_model_checkpoint(epoch + 1,
                                                       model,
                                                       optimizer,
                                                       min_val_loss,
                                                       args_dump,
                                                       train_args,
                                                       model_args_d,
                                                       output_pth_head,
                                                       session_dir,
                                                       best=True)
                    print('epoch: {} best val loss so far, saved model checkpoint'.format(epoch + 1))

            stats['epoch_val'][epoch] = {'total_loss': running_loss,
                                         'avg_loss': running_loss / sample_count}
            print()
            if scheduler is None:
                print('no lr scheduler found. continuing...')
            else:
                print('adjusting lr...')
                scheduler.step()

        # TEST for current session (model with best val losses during training)
        best_checkpoint = torch.load(best_mpath)
        best_model_sd = best_checkpoint['model']
        model.load_state_dict(best_model_sd)
        model.to(device)
        model.eval()
        test_loss = 0.0
        test_samples = 0

        stats['dataset_split'] = {'train': dataset.train_idxs,
                                  'val': dataset.val_idxs,
                                  'test': dataset.test_idxs}
        
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
            
        