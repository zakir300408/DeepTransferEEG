# -*- coding: utf-8 -*-
# @Time    : 2023/01/11
# @Author  : Siyang Li
# @File    : dnn.py
import numpy as np
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

from utils.network import backbone_net
from utils.LogRecord import LogRecord
from utils.dataloader import read_mi_combine_tar
from utils.utils import fix_random_seed, cal_acc_comb, data_loader

import gc
import sys


def train_target(args):
    X_src, y_src, X_tar, y_tar = read_mi_combine_tar(args)
    # prepare per‐session boundaries for CustomEpoch
    if args.data == 'CustomEpoch':
        df_meta = pd.read_csv('./data/CustomEpoch/meta.csv')
        counts = df_meta['n_trials'].values
        idts = args.idt if isinstance(args.idt, (list,tuple)) else [args.idt]
        bounds = []
        start = 0
        for i in idts:
            length = counts[i]
            bounds.append((start, start + length))
            start += length
        args.tar_bounds = bounds

    print('X_src, y_src, X_tar, y_tar:', X_src.shape, y_src.shape, X_tar.shape, y_tar.shape)
    # dynamic trial count per subject (CustomEpoch subjects vary in trial number)
    args.trial_num = y_tar.shape[0]
    dset_loaders = data_loader(X_src, y_src, X_tar, y_tar, args)

    # grab EA‐aligned target tensors for per‐session splits
    Xt_tensor, Yt_tensor = dset_loaders["target"].dataset.tensors
    if args.data_env != 'local':
        Xt_tensor, Yt_tensor = Xt_tensor.cpu(), Yt_tensor.cpu()
    args.Xt_aligned = Xt_tensor.numpy()
    args.Yt_aligned = Yt_tensor.numpy()

    netF, netC = backbone_net(args, return_type='xy')
    if args.data_env != 'local':
        netF, netC = netF.cuda(), netC.cuda()
    base_network = nn.Sequential(netF, netC)

    criterion = nn.CrossEntropyLoss()

    optimizer_f = optim.Adam(netF.parameters(), lr=args.lr)
    optimizer_c = optim.Adam(netC.parameters(), lr=args.lr)

    max_iter = args.max_epoch * len(dset_loaders["source"])
    interval_iter = max_iter // args.max_epoch
    args.max_iter = max_iter
    iter_num = 0
    base_network.train()

    while iter_num < max_iter:
        try:
            inputs_source, labels_source = next(iter_source)
        except:
            iter_source = iter(dset_loaders["source"])
            inputs_source, labels_source = next(iter_source)

        if inputs_source.size(0) == 1:
            continue

        iter_num += 1

        features_source, outputs_source = base_network(inputs_source)

        classifier_loss = criterion(outputs_source, labels_source)

        optimizer_f.zero_grad()
        optimizer_c.zero_grad()
        classifier_loss.backward()
        optimizer_f.step()
        optimizer_c.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            base_network.eval()

            acc_t_te, _ = cal_acc_comb(dset_loaders["Target"], base_network, args=args)
            args.log.record(f"Task: {args.task_str}, Iter:{int(iter_num//len(dset_loaders['source']))}/{args.max_epoch}; Acc = {acc_t_te:.2f}%")
            print(f"Task: {args.task_str}, Iter:{int(iter_num//len(dset_loaders['source']))}/{args.max_epoch}; Acc = {acc_t_te:.2f}%")

            # per‐session breakdown
            for idx, (s, e) in enumerate(getattr(args, 'tar_bounds', [])):
                if e <= s:
                    args.log.record(f"  Session {idx} skipped (no trials)")
                    continue
                X_seg = args.Xt_aligned[s:e]
                y_seg = args.Yt_aligned[s:e]

                # build loader
                ts = torch.from_numpy(X_seg).float()
                if 'EEGNet' in args.backbone:
                    # already in N,C,H,W from EA
                    ts = ts
                ys = torch.from_numpy(y_seg).long()
                if args.data_env != 'local':
                    ts, ys = ts.cuda(), ys.cuda()
                loader = DataLoader(TensorDataset(ts, ys),
                                    batch_size=args.batch_size*3,
                                    shuffle=False)
                acc_sess, _ = cal_acc_comb(loader, base_network, args=args)
                args.log.record(f"  Session {idx} Acc = {acc_sess:.2f}%")
                print(f"  Session {idx} Acc = {acc_sess:.2f}%")

            base_network.train()

    print('Test Acc = {:.2f}%'.format(acc_t_te))

    print('saving model...')
    # ensure output directory exists
    save_run_dir = os.path.join('.', 'runs', args.data_name)
    os.makedirs(save_run_dir, exist_ok=True)
    if args.align:
        torch.save(base_network.state_dict(),
                   os.path.join(save_run_dir, f"{args.backbone}_S{args.idt}_seed{args.SEED}.ckpt"))
    else:
        torch.save(base_network.state_dict(),
                   os.path.join(save_run_dir, f"{args.backbone}_S{args.idt}_seed{args.SEED}_noEA.ckpt"))

    gc.collect()
    if args.data_env != 'local':
        torch.cuda.empty_cache()

    return acc_t_te


if __name__ == '__main__':
    # include your CustomEpoch
    data_name_list = ['CustomEpoch']

    # For CustomEpoch, read each session filename as a “subject”
    subject_names = None
    if 'CustomEpoch' in data_name_list:
        df_meta = pd.read_csv('./data/CustomEpoch/meta.csv')
        # keep each .mat filename in order
        subject_names = df_meta['file'].tolist()

    # initialize results container
    dct = pd.DataFrame()

    for data_name in data_name_list:
        # If CustomEpoch, set N from per-session list
        if data_name == 'CustomEpoch':
            N = len(subject_names)

        if data_name == 'BNCI2014001': 
            paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 9, 22, 2, 1001, 250, 144, 248
        elif data_name == 'BNCI2014002':
            paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 14, 15, 2, 2561, 512, 100, 640
        elif data_name == 'BNCI2015001':
            paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 12, 13, 2, 2561, 512, 200, 640
        elif data_name == 'BNCI2014001-4':
            paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 9, 22, 4, 1001, 250, 288, 248
        elif data_name == 'CustomEpoch':
            # adjust these values to match your CustomEpoch dataset
            paradigm        = 'MI'
            N               = len(subject_names)  # sessions count
            chn             = 31
            class_num       = 2                 # your labels are 0/1
            time_sample_num = 1600              # number of time-samples per trial
            sample_rate     = 200               # (or your actual sampling rate)
            trial_num       = 1000              # not used for FC size
            feature_deep_dim= 400               # F2*(time_sample_num/(4*8)) = 8*(1600/32)
        else:
            raise ValueError(f"Unknown data_name {data_name}")

        args = argparse.Namespace(
            feature_deep_dim=feature_deep_dim, trial_num=trial_num,
            time_sample_num=time_sample_num, sample_rate=sample_rate,
            N=N, chn=chn, class_num=class_num, paradigm=paradigm, data_name=data_name
        )

        args.method = 'EEGNet'
        args.backbone = 'EEGNet'

        # enable EA instead of pure source‐only training
        args.align = True

        # learning rate
        args.lr = 0.001

        # train batch size
        args.batch_size = 32

        # training epochs
        args.max_epoch = 20

        # detect device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        args.data_env = 'gpu' if device.type=='cuda' else 'local'
        args.device = device

        total_acc = []

        # train multiple randomly initialized models
        for s in [4, 5, 6, 7, 8, 9]:
            args.SEED = s

            fix_random_seed(args.SEED)
            torch.backends.cudnn.deterministic = True

            args.data = data_name
            print(args.data)
            print(args.method)
            print(args.SEED)
            print(args)

            args.local_dir = './data/' + str(data_name) + '/'
            args.result_dir = './logs/'
            my_log = LogRecord(args)
            my_log.log_init()
            my_log.record('=' * 50 + '\n' + os.path.basename(__file__) + '\n' + '=' * 50)
            args.log = my_log

            clean = [f.replace('.mat', '') for f in subject_names]
            prefixes = sorted({name.split('_')[0] for name in clean})
            sub_acc = []
            for prefix in prefixes:
                # all session‐indices with this prefix become the target
                idts = [i for i, name in enumerate(clean) if name.split('_')[0] == prefix]
                args.idt = idts
                target_str = prefix
                source_str = 'Except_' + '_'.join([p for p in prefixes if p != prefix])
                args.task_str = source_str + '_2_' + target_str
                info_str = '\n===== Transfer to ' + target_str + ' ====='
                print(info_str); my_log.record(info_str)
                acc = train_target(args)
                sub_acc.append(acc)
            sub_acc_all = np.array(sub_acc)

            print('Sub acc: ', np.round(sub_acc_all, 3))
            print('Avg acc: ', np.round(np.mean(sub_acc_all), 3))
            total_acc.append(sub_acc_all)

            acc_sub_str = str(np.round(sub_acc_all, 3).tolist())
            acc_mean_str = str(np.round(np.mean(sub_acc_all), 3).tolist())
            args.log.record("\n==========================================")
            args.log.record(acc_sub_str)
            args.log.record(acc_mean_str)

        args.log.record('\n' + '#' * 20 + 'final results' + '#' * 20)

        print(str(total_acc))

        args.log.record(str(total_acc))

        subject_mean = np.round(np.average(total_acc, axis=0), 5)
        total_mean = np.round(np.average(np.average(total_acc)), 5)
        total_std = np.round(np.std(np.average(total_acc, axis=1)), 5)

        print(subject_mean)
        print(total_mean)
        print(total_std)

        args.log.record(str(subject_mean))
        args.log.record(str(total_mean))
        args.log.record(str(total_std))

        result_dct = {'dataset': data_name, 'avg': total_mean, 'std': total_std}
        for i in range(len(subject_mean)):
            result_dct['s' + str(i)] = subject_mean[i]

        # use pd.concat instead of deprecated DataFrame.append
        dct = pd.concat([dct, pd.DataFrame([result_dct])], ignore_index=True)

    # save results to csv
    dct.to_csv('./logs/' + str(args.method) + ".csv")