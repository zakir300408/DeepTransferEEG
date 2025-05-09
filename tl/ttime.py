# -*- coding: utf-8 -*-
# @Time    : 2023/07/07
# @Author  : Siyang Li
# @File    : ttime.py
import numpy as np
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import csv
from utils.network import backbone_net
from utils.LogRecord import LogRecord
from utils.dataloader import read_mi_combine_tar
from utils.utils import fix_random_seed, cal_acc_comb, data_loader, cal_auc_comb, cal_score_online
from utils.alg_utils import EA, EA_online
from scipy.linalg import fractional_matrix_power
from utils.loss import Entropy
from sklearn.metrics import roc_auc_score, accuracy_score

import gc
import sys
import time
from torch.utils.data import DataLoader, TensorDataset


def TTIME(loader, model, args, balanced=True):
    # "T-TIME: Test-Time Information Maximization Ensemble for Plug-and-Play BCIs"
    # IEEE Transactions on Biomedical Engineering
    # Note that the ensemble experiment is separately implemented in ttime_ensemble.py, using recorded test prediction.

    if balanced == False and args.data_name == 'BNCI2014001-4':
        print('ERROR, imbalanced multi-class not implemented')
        sys.exit(0)

    y_true = []
    y_pred = []

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # initialize test reference matrix for Incremental EA
    if args.align:
        R = 0

    if not balanced:
        zk_arrs = np.zeros(2)
        c = 4

    iter_test = iter(loader)

    # loop through test data stream one by one
    for i in range(len(loader)):
        #################### Phase 1: target label prediction ####################
        model.eval()
        data = next(iter_test)
        inputs, labels = data[0], data[1]
        inputs = inputs.reshape(1, 1, inputs.shape[-2], inputs.shape[-1]).to(args.device)

        # accumulate test data
        if i == 0:
            data_cum = inputs.float().cpu()
        else:
            data_cum = torch.cat((data_cum, inputs.float().cpu()), 0)

        # Incremental EA
        if args.align:
            start_time = time.time()

            if i == 0:
                sample_test = data_cum.reshape(args.chn, args.time_sample_num)
            else:
                sample_test = data_cum[i].reshape(args.chn, args.time_sample_num)
            # update reference matrix
            R = EA_online(sample_test, R, i)

            sqrtRefEA = fractional_matrix_power(R, -0.5)
            # transform current test sample
            sample_test = np.dot(sqrtRefEA, sample_test)

            EA_time = time.time()
            if args.calc_time:
                print('sample ', str(i), ', pre-inference IEA finished time in ms:', np.round((EA_time - start_time) * 1000, 3))
            sample_test = sample_test.reshape(1, 1, args.chn, args.time_sample_num)
        else:
            sample_test = data_cum[i].numpy()
            sample_test = sample_test.reshape(1, 1, sample_test.shape[1], sample_test.shape[2])

        sample_test = torch.from_numpy(sample_test).to(torch.float32).to(args.device)

        _, outputs = model(sample_test)

        softmax_out = nn.Softmax(dim=1)(outputs)

        outputs = outputs.float().cpu()
        labels = labels.float().cpu()
        _, predict = torch.max(outputs, 1)

        y_pred.append(softmax_out.detach().cpu().numpy())
        y_true.append(labels.item())

        #################### Phase 2: target model update ####################
        model.train()
        # sliding batch
        if (i + 1) >= args.test_batch and (i + 1) % args.stride == 0:
            if args.align:
                batch_test = np.copy(data_cum[i - args.test_batch + 1:i + 1])
                # transform test batch
                batch_test = np.dot(sqrtRefEA, batch_test)
                batch_test = np.transpose(batch_test, (1, 2, 0, 3))
            else:
                batch_test = data_cum[i - args.test_batch + 1:i + 1].numpy()
                batch_test = batch_test.reshape(args.test_batch, 1, batch_test.shape[2], batch_test.shape[3])

            batch_test = torch.from_numpy(batch_test).to(torch.float32).to(args.device)

            start_time = time.time()
            for step in range(args.steps):

                _, outputs = model(batch_test)
                outputs = outputs.float().cpu()

                args.epsilon = 1e-5
                softmax_out = nn.Softmax(dim=1)(outputs / args.t)
                # Conditional Entropy Minimization loss
                CEM_loss = torch.mean(Entropy(softmax_out))
                msoftmax = softmax_out.mean(dim=0)

                if balanced:
                    # Marginal Distribution Regularization loss
                    MDR_loss = torch.sum(msoftmax * torch.log(msoftmax + args.epsilon))
                    loss = CEM_loss + MDR_loss
                else:
                    # Adaptive Marginal Distribution Regularization
                    qk = torch.zeros((args.class_num, )).to(torch.float32)
                    for k in range(args.class_num):
                        qk[k] = msoftmax[k] / (c + zk_arrs[k])
                    sum_qk = torch.sum(qk)
                    normed_qk = qk / sum_qk
                    AMDR_loss = torch.sum(normed_qk * torch.log(normed_qk + args.epsilon))
                    loss = CEM_loss + AMDR_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            TTA_time = time.time()
            if args.calc_time:
                print('sample ', str(i), ', post-inference model update finished in ms:', np.round((TTA_time - start_time) * 1000, 3))

            if not balanced:
                if i + 1 == args.test_batch:
                    args.pred_thresh = 0.7
                    pl = torch.max(softmax_out, 1)[1]
                    for l in range(args.test_batch):
                        if pl[l] == 0:
                            if softmax_out[l][0] > args.pred_thresh:
                                zk_arrs[0] += 1
                        elif pl[l] == 1:
                            if softmax_out[l][1] > args.pred_thresh:
                                zk_arrs[1] += 1
                        else:
                            print('ERROR in pseudo labeling!')
                            sys.exit(0)
                else:
                    # update confident prediction ids for current test sample
                    pl = torch.max(softmax_out, 1)[1]
                    if pl[-1] == 0:
                        if softmax_out[-1][0] > args.pred_thresh:
                            zk_arrs[0] += 1
                    elif pl[-1] == 1:
                        if softmax_out[-1][1] > args.pred_thresh:
                            zk_arrs[1] += 1
                    else:
                        print('ERROR in pseudo labeling!')

        model.eval()

    if balanced:
        _, predict = torch.max(torch.from_numpy(np.array(y_pred)).to(torch.float32).reshape(-1, args.class_num), 1)
        pred = torch.squeeze(predict).float()
        score = accuracy_score(y_true, pred)
        if args.data_name == 'BNCI2014001-4':
            y_pred = np.array(y_pred).reshape(-1, )  # multiclass
        else:
            y_pred = np.array(y_pred).reshape(-1, args.class_num)[:, 1]  # binary
    else:
        predict = torch.from_numpy(np.array(y_pred)).to(torch.float32).reshape(-1, args.class_num)
        y_pred = np.array(predict).reshape(-1, args.class_num)[:, 1]  # binary
        score = roc_auc_score(y_true, y_pred)

    return score * 100, y_pred


def train_target(args):
    if not args.align:
        extra_string = '_noEA'
    else:
        extra_string = ''
    # make sure save directory exists
    os.makedirs(os.path.join('.', 'runs', args.data_name), exist_ok=True)

    # load source/target
    X_src, y_src, X_tar, y_tar = read_mi_combine_tar(args)

    # build per-session bounds for CustomEpoch
    if args.data == 'CustomEpoch':
        df_meta     = pd.read_csv('./data/CustomEpoch/meta.csv')
        counts      = df_meta['n_trials'].values
        idts        = args.idt if isinstance(args.idt, (list, tuple)) else [args.idt]
        # compute local bounds relative to X_tar (only this subject’s sessions)
        sel_counts    = counts[idts]
        local_starts  = np.concatenate(([0], np.cumsum(sel_counts)[:-1]))
        local_ends    = np.cumsum(sel_counts)
        args.tar_bounds = [(local_starts[k], local_ends[k]) for k in range(len(idts))]

    dset_loaders = data_loader(X_src, y_src, X_tar, y_tar, args)

    # move backbone and classifier to configured device
    netF, netC = backbone_net(args, return_type='xy')
    netF, netC = netF.to(args.device), netC.to(args.device)
    base_network = nn.Sequential(netF, netC).to(args.device)

    if args.max_epoch == 0:
        if args.align:
            if args.data_env != 'local':
                base_network.load_state_dict(torch.load('./runs/' + str(args.data_name) + '/' + str(args.backbone) +
                    '_S' + str(args.idt) + '_seed' + str(args.SEED) + extra_string + '.ckpt'))
            else:
                base_network.load_state_dict(torch.load('./runs/' + str(args.data_name) + '/' + str(args.backbone) +
                    '_S' + str(args.idt) + '_seed' + str(args.SEED) + extra_string + '.ckpt', map_location=torch.device('cpu')))
    else:
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
            # send source batch to device
            inputs_source, labels_source = inputs_source.to(args.device), labels_source.to(args.device)

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

                if args.balanced:
                    acc_t_te, _ = cal_acc_comb(dset_loaders["Target"], base_network, args=args)
                    log_str = 'Task: {}, Iter:{}/{}; Offline-EA Acc = {:.2f}%'.format(args.task_str, int(iter_num // len(dset_loaders["source"])), int(max_iter // len(dset_loaders["source"])), acc_t_te)
                else:
                    acc_t_te, _ = cal_auc_comb(dset_loaders["Target-Imbalanced"], base_network, args=args)
                    log_str = 'Task: {}, Iter:{}/{}; Offline-EA AUC = {:.2f}%'.format(args.task_str, int(iter_num // len(dset_loaders["source"])), int(max_iter // len(dset_loaders["source"])), acc_t_te)
                args.log.record(log_str)
                print(log_str)

                base_network.train()

        print('saving model...')
        torch.save(base_network.state_dict(),
                   './runs/' + str(args.data_name) + '/' + str(args.backbone) + '_S' + str(
                       args.idt) + '_seed' + str(args.SEED) + extra_string + '.ckpt')


    base_network.eval()

    # global pre-TTA IEA
    score = cal_score_online(dset_loaders["Target-Online"], base_network, args=args)
    pre_score = score
    # per-session Pre-TTA IEA AUC
    pre_scores = []
    if args.data == 'CustomEpoch':
        for idx,(s,e) in enumerate(getattr(args,'tar_bounds',[])):
            if e<=s: continue
            ts = torch.from_numpy(X_tar[s:e]).float()
            if 'EEGNet' in args.backbone:
                ts = ts.unsqueeze(3).permute(0,3,1,2)
            ys = torch.from_numpy(y_tar[s:e]).long()
            if args.data_env!='local':
                ts,ys = ts.cuda(), ys.cuda()
            loader = DataLoader(TensorDataset(ts,ys), batch_size=1, shuffle=False)
            score_s = cal_score_online(loader, base_network, args=args)
            args.log.record(f"  Pre-TTA IEA Session {idx} AUC = {score_s:.2f}%")
            print(f"  Pre-TTA IEA Session {idx} AUC = {score_s:.2f}%")
            pre_scores.append(score_s)
        pre_score = float(np.mean(pre_scores)) if pre_scores else 0.0
    else:
        pre_score = cal_score_online(dset_loaders["Target-Online"], base_network, args=args)

    if args.balanced:
        log_str = 'Task: {}, Pre-TTA IEA Acc = {:.2f}%'.format(args.task_str, pre_score)
    else:
        log_str = 'Task: {}, Pre-TTA IEA AUC = {:.2f}%'.format(args.task_str, pre_score)
    args.log.record(log_str)
    print(log_str)

    print('executing TTA per session...')
    sess_accs = []
    sess_test_accs = []
    for idx, (s, e) in enumerate(getattr(args, 'tar_bounds', [])):
        if e <= s:
            args.log.record(f"  Session {idx} skipped for TTA (no trials)")
            continue
        # build per-session tensors
        ts = torch.from_numpy(X_tar[s:e]).float()
        ys = torch.from_numpy(y_tar[s:e]).long()
        # restrict to first 20 trials only
        max_tta = 20
        ts_tta = ts[:max_tta]
        ys_tta = ys[:max_tta]
        ts_rem = ts[max_tta:]
        ys_rem = ys[max_tta:]
        if 'EEGNet' in args.backbone:
            ts_tta = ts_tta.unsqueeze(3).permute(0, 3, 1, 2)
            ts_rem = ts_rem.unsqueeze(3).permute(0, 3, 1, 2)
        if args.data_env != 'local':
            ts_tta, ys_tta = ts_tta.cuda(), ys_tta.cuda()
            ts_rem, ys_rem = ts_rem.cuda(), ys_rem.cuda()
        loader = DataLoader(TensorDataset(ts_tta, ys_tta), batch_size=1, shuffle=False)
        # run TTA on this (trimmed) session
        acc_sess, _ = TTIME(loader, base_network, args=args, balanced=args.balanced)
        args.log.record(f"  Session {idx} TTA {'Acc' if args.balanced else 'AUC'} = {acc_sess:.2f}%")
        print(f"  Session {idx} TTA {'Acc' if args.balanced else 'AUC'} = {acc_sess:.2f}%")
        sess_accs.append(acc_sess)
        # test on remaining trials
        if len(ts_rem) > 0:
            loader_rem = DataLoader(TensorDataset(ts_rem, ys_rem), batch_size=1, shuffle=False)
            if args.balanced:
                acc_rem, _ = cal_acc_comb(loader_rem, base_network, args=args)
            else:
                acc_rem = cal_score_online(loader_rem, base_network, args=args)
            args.log.record(f"  Session {idx} post-TTA test {'Acc' if args.balanced else 'AUC'} = {acc_rem:.2f}%")
            print(f"  Session {idx} post-TTA test {'Acc' if args.balanced else 'AUC'} = {acc_rem:.2f}%")
            sess_test_accs.append(acc_rem)
    # overall average across sessions
    acc_t_te = float(np.mean(sess_accs)) if sess_accs else 0.0
    acc_test = float(np.mean(sess_test_accs)) if sess_test_accs else 0.0
    log_str = f"Task: {args.task_str}, Overall TTA {'Acc' if args.balanced else 'AUC'} = {acc_t_te:.2f}%"
    args.log.record(log_str)
    print(log_str)
    args.log.record(f"Task: {args.task_str}, Overall post-TTA test {'Acc' if args.balanced else 'AUC'} = {acc_test:.2f}%")
    print(f"Overall post-TTA test {'Acc' if args.balanced else 'AUC'} = {acc_test:.2f}%")

    torch.save(base_network.state_dict(), './runs/' + str(args.data_name) + '/' + str(args.backbone) + '_S' + str(args.idt) + '_seed' + str(
        args.SEED) + extra_string + '_adapted' + '.ckpt')

    gc.collect()
    if args.data_env != 'local':
        torch.cuda.empty_cache()

    return acc_t_te, pre_score, acc_test


if __name__ == '__main__':

    data_name_list = ['CustomEpoch']

    # load session filenames and extract prefixes as in dnn.py
    df_meta = pd.read_csv('./data/CustomEpoch/meta.csv')
    # files list from df_meta (‘file’ column, full filenames)
    files = df_meta['file'].tolist()
    prefixes = sorted({f.split('_')[0] for f in files})
    subject_names = prefixes

    # prepare result columns for each prefix
    sess_cols = [f's{i}' for i in range(len(subject_names))]
    dct = pd.DataFrame(columns=['dataset','avg','std'] + sess_cols)

    for data_name in data_name_list:
        # N: number of subjects, chn: number of channels
        if data_name == 'BNCI2014001':
            paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 9, 22, 2, 1001, 250, 144, 248
        elif data_name == 'BNCI2014002':
            paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 14, 15, 2, 2561, 512, 100, 640
        elif data_name == 'BNCI2015001':
            paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 12, 13, 2, 2561, 512, 200, 640
        elif data_name == 'CustomEpoch':
            paradigm = 'MI'
            N = len(subject_names)  # number of unique prefixes/sessions
            chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 31, 2, 1600, 200, 1000, 400
        else:
            raise ValueError(f"Unknown data_name {data_name}")

        # whether to use pretrained model
        # if source models have not been trained, set use_pretrained_model to False to train them
        # alternatively, run dnn.py to train source models, in seperating the steps
        use_pretrained_model = True
        if use_pretrained_model:
            # no training
            max_epoch = 0
        else:
            # training epochs
            max_epoch = 20

        # learning rate
        lr = 0.001

        # test batch size
        test_batch = 20

        # update step
        steps = 10

        # update stride
        stride = 1

        # whether to use EA
        align = True

        # temperature rescaling, for test entropy calculation
        t = 2

        # whether to test balanced or imbalanced (2:1) target subject
        balanced = False

        # whether to record running time
        calc_time = False

        args = argparse.Namespace(feature_deep_dim=feature_deep_dim, align=align, lr=lr, t=t, max_epoch=max_epoch,
                                  trial_num=trial_num, time_sample_num=time_sample_num, sample_rate=sample_rate,
                                  N=N, chn=chn, class_num=class_num, stride=stride, steps=steps, calc_time=calc_time,
                                  paradigm=paradigm, test_batch=test_batch, data_name=data_name, balanced=balanced)

        args.method = 'T-TIME'
        args.backbone = 'EEGNet'

        # train batch size
        args.batch_size = 32

        # GPU device id
        # detect device and default to GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        args.device = device
        args.data_env = 'gpu' if torch.cuda.is_available() else 'local'
        total_acc = []

        # update multiple models, independently, from the source models
        for s in [1, 42]:
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

            # log device usage
            print(f"Using device: {args.device}")
            args.log.record(f"Using device: {args.device}")

            sub_acc_all = np.zeros(N)
            pre_acc_all = np.zeros(N)
            post_acc_all = np.zeros(N)

            for idt in range(N):
                # collect all session indices belonging to this subject prefix
                target_str = subject_names[idt]
                idts = [i for i, fn in enumerate(files) if fn.split('_')[0] == target_str]
                args.idt = idts
                # use prefix names
                others = subject_names.copy()
                others.pop(idt)
                source_str = 'Except_' + '_'.join(others)
                args.task_str = source_str + '_2_' + target_str
                info_str = '\n========================== Transfer to ' + target_str + ' =========================='
                print(info_str)
                my_log.record(info_str)

                tta_acc, pre_acc, post_acc = train_target(args)
                sub_acc_all[idt] = tta_acc
                pre_acc_all[idt] = pre_acc
                post_acc_all[idt] = post_acc

            print('Sub acc: ', np.round(sub_acc_all, 3))
            print('Avg acc: ', np.round(np.mean(sub_acc_all), 3))
            print('Sub pre-TTA IEA Acc: ', np.round(pre_acc_all, 3))
            print('Avg pre-TTA IEA Acc: ', np.round(np.mean(pre_acc_all), 3))
            print('Sub post-TTA test AUC: ', np.round(post_acc_all, 3))
            print('Avg post-TTA test AUC: ', np.round(np.mean(post_acc_all), 3))

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

        # build result_dct for this dataset
        result_dct = {'dataset': data_name, 'avg': total_mean, 'std': total_std}
        for i, v in enumerate(subject_mean):
            result_dct[f's{i}'] = v

        dct = pd.concat([dct, pd.DataFrame([result_dct])], ignore_index=True)

    dct.to_csv('./logs/' + str(args.method) + ".csv")