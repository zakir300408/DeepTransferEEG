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
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve
from ttime_ensemble import SML                # << add this import

import gc
import sys
import time
from torch.utils.data import DataLoader, TensorDataset

# Add once: prepare_loader helper to avoid recreating DataLoader inline
def prepare_loader(data, targets, batch_size):
    return DataLoader(TensorDataset(data, targets), batch_size=batch_size, shuffle=False)

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

    # Initialize data_cum before the loop
    data_cum = None
    # loop through test data stream one by one
    for i in range(len(loader)):
        #################### Phase 1: target label prediction ####################
        model.eval()
        data = next(iter_test)
        inputs, labels = data[0], data[1]
        inputs = inputs.reshape(1, 1, inputs.shape[-2], inputs.shape[-1]).to(args.device)

        # accumulate test data
        if data_cum is None:
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
        # binary case: calibrate threshold via Youden’s J
        if args.class_num == 2:
            # extract positive-class scores
            y_scores = np.array(y_pred).reshape(-1, args.class_num)[:, 1]
            # compute ROC curve
            fpr, tpr, thresholds = roc_curve(y_true, y_scores)
            # Youden’s J statistic
            optimal_idx = np.argmax(tpr - fpr)
            best_thresh = thresholds[optimal_idx]
            # apply best threshold
            preds = (y_scores > best_thresh).astype(int)
            score = accuracy_score(y_true, preds)
        else:
            # multiclass: default argmax
            _, predict = torch.max(torch.from_numpy(np.array(y_pred))
                               .to(torch.float32).reshape(-1, args.class_num), 1)
            preds = torch.squeeze(predict).float().numpy().astype(int)
            score = accuracy_score(y_true, preds)
        # retain y_pred formatting
        if args.data_name == 'BNCI2014001-4':
            y_pred = np.array(y_pred).reshape(-1,)  # multiclass
        else:
            y_pred = y_scores
    else:
        predict = torch.from_numpy(np.array(y_pred)).to(torch.float32).reshape(-1, args.class_num)
        y_pred = np.array(predict).reshape(-1, args.class_num)[:, 1]  # binary
        score = roc_auc_score(y_true, y_pred)

    # after loop ends, ensure sqrtRefEA exists when align=True
    if args.align:
        return score * 100, y_pred, sqrtRefEA
    else:
        return score * 100, y_pred, None


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
        # compute local bounds relative to X_tar
        sel_counts    = counts[idts]
        local_starts  = np.concatenate(([0], np.cumsum(sel_counts)[:-1]))
        local_ends    = np.cumsum(sel_counts)
        args.tar_bounds = [(local_starts[k], local_ends[k]) for k in range(len(idts))]
        # map each bound to its actual session filename
        file_list = df_meta['file'].tolist()
        args.session_names = [file_list[i] for i in idts]

    dset_loaders = data_loader(X_src, y_src, X_tar, y_tar, args)

    # move backbone and classifier to configured device
    netF, netC = backbone_net(args, return_type='xy')
    netF, netC = netF.to(args.device), netC.to(args.device)
    base_network = nn.Sequential(netF, netC).to(args.device)

    # helper for per‐session summary
    def _log_session_summary(ttas, posts):
        print('Subject per-session TTA Accuracies:', np.round(ttas, 3))
        args.log.record(f"Subject per-session TTA Accuracies: {np.round(ttas, 3)}")
        print('Subject per-session Post-TTA Accuracies:', np.round(posts, 3))
        args.log.record(f"Subject per-session Post-TTA Accuracies: {np.round(posts, 3)}")

    if args.max_epoch == 0:
        # generate correct id string for loading pretrained model
        if isinstance(args.idt, (list, tuple)):
            idt_str = '_'.join(map(str, args.idt))
        else:
            idt_str = str(args.idt)
        if args.data_env != 'local':
            base_network.load_state_dict(torch.load(
                f'./runs/{args.data_name}/{args.backbone}_S{idt_str}_seed{args.SEED}{extra_string}_best.ckpt'))
        else:
            base_network.load_state_dict(torch.load(
                f'./runs/{args.data_name}/{args.backbone}_S{idt_str}_seed{args.SEED}{extra_string}_best.ckpt',
                map_location=torch.device('cpu')))
        # perform a quick evaluation and print results
        if args.balanced:
            acc_val, _ = cal_acc_comb(dset_loaders["Target"], base_network, args=args)
            metric = "Acc"
        else:
            acc_val = cal_score_online(dset_loaders["Target-Imbalanced"], base_network, args=args)
            metric = "AUC"
        print(f"Task: {args.task_str}, Pre-TTA IEA {metric} = {acc_val:.2f}%")
        args.log.record(f"Task: {args.task_str}, Pre-TTA IEA {metric} = {acc_val:.2f}%")

        # after loading best model, save Pre-TTA probabilities
        loader_pre = DataLoader(
            TensorDataset(torch.from_numpy(X_tar).float(),
                          torch.from_numpy(y_tar).long()),
            batch_size=1, shuffle=False
        )
        pre_probs = []
        base_network.eval()
        for x_pre, _ in loader_pre:
            # reshape to (1,1,chn,time) for EEGNet
            x_in = x_pre
            if 'EEGNet' in args.backbone:
                x_in = x_in.unsqueeze(3).permute(0,3,1,2)
            with torch.no_grad():
                _, out_pre = base_network(x_in.to(args.device))
                prob_pre = nn.Softmax(dim=1)(out_pre).cpu().numpy().squeeze()
            pre_probs.append(prob_pre)
        pre_probs = np.stack(pre_probs)  # shape (n_trials, class_num)
        np.savetxt(
            os.path.join(args.result_dir,
                         f"{args.data_name}_T-TIME_seed_{args.SEED}_pre_probs.csv"),
            pre_probs, delimiter=",",
        )

        # ─── now use helper instead of inline loop ─────────────────────────
        avg_tta, avg_post = run_chunked_tta(base_network, X_tar, y_tar, args, idt_str, extra_string)
        # define & save final adapted model
        best_ckpt_adapted = f'./runs/{args.data_name}/{args.backbone}_S{idt_str}_seed{args.SEED}{extra_string}_adapted.ckpt'
        torch.save(base_network.state_dict(), best_ckpt_adapted)

        # ─── dump per-seed predictions ─────────────────────────────────────────
        loader_all = DataLoader(
            TensorDataset(torch.from_numpy(X_tar).float(),
                          torch.from_numpy(y_tar).long()),
            batch_size=1, shuffle=False
        )
        _, y_pred_full, _ = TTIME(loader_all, base_network,
                                  args=args, balanced=args.balanced)
        # For CustomEpoch, output predictions one row per session.
        # write per-session predictions to CSV (handles varying lengths)
        out_path = os.path.join(args.result_dir,
                                f"{args.data_name}_T-TIME_seed_{args.SEED}_pred.csv")
        with open(out_path, 'w') as fw:
            if args.data_name == 'CustomEpoch':
                for start, end in args.tar_bounds:
                    row = y_pred_full[start:end]
                    fw.write(','.join(map(str, row.tolist())) + '\n')
            else:
                # flat array case
                fw.write(','.join(map(str, y_pred_full.tolist())) + '\n')
        return avg_tta, acc_val, avg_post

    else:
        criterion = nn.CrossEntropyLoss()
        optimizer_f = optim.Adam(netF.parameters(), lr=args.lr)
        optimizer_c = optim.Adam(netC.parameters(), lr=args.lr)

        max_iter = args.max_epoch * len(dset_loaders["source"])
        interval_iter = max_iter // args.max_epoch
        args.max_iter = max_iter
        iter_num = 0
        base_network.train()

        best_acc = 0.0
        # define best‐model path with corrected session id string
        if isinstance(args.idt, (list, tuple)):
            idt_str = '_'.join(map(str, args.idt))
        else:
            idt_str = str(args.idt)
        best_ckpt = f'./runs/{args.data_name}/{args.backbone}_S{idt_str}_seed{args.SEED}{extra_string}_best.ckpt'
        # define adapted‐model path
        best_ckpt_adapted = f'./runs/{args.data_name}/{args.backbone}_S{idt_str}_seed{args.SEED}{extra_string}_adapted.ckpt'

        iter_source = iter(dset_loaders["source"])
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
                # Added print statement for progress reporting
                epoch_num = iter_num // interval_iter if iter_num < max_iter else args.max_epoch
                print(f"Epoch {epoch_num}/{args.max_epoch} - Iteration {iter_num}/{max_iter} completed.")

                base_network.eval()
                if args.balanced:
                    acc_t_te, _ = cal_acc_comb(dset_loaders["Target"], base_network, args=args)
                else:
                    acc_t_te, _ = cal_auc_comb(dset_loaders["Target-Imbalanced"], base_network, args=args)
                print(f"Validation Accuracy: {acc_t_te:.2f}%")
                # if this is the best so far, save it
                if acc_t_te > best_acc:
                    best_acc = acc_t_te
                    torch.save(base_network.state_dict(), best_ckpt)
                base_network.train()

        # load the best model (not the last one) before any further evaluation
        base_network.load_state_dict(torch.load(best_ckpt, map_location=args.device))

        # global pre-TTA IEA
        pre_score = cal_score_online(dset_loaders["Target-Online"], base_network, args=args)
        metric = 'Acc' if args.balanced else 'AUC'
        print(f"Task: {args.task_str}, Pre-TTA IEA {metric} = {pre_score:.2f}%")
        args.log.record(f"Task: {args.task_str}, Pre-TTA IEA {metric} = {pre_score:.2f}%")

        # after loading best model, save Pre-TTA probabilities
        loader_pre = DataLoader(
            TensorDataset(torch.from_numpy(X_tar).float(),
                          torch.from_numpy(y_tar).long()),
            batch_size=1, shuffle=False
        )
        pre_probs = []
        base_network.eval()
        for x_pre, _ in loader_pre:
            # reshape to (1,1,chn,time) for EEGNet
            x_in = x_pre
            if 'EEGNet' in args.backbone:
                x_in = x_in.unsqueeze(3).permute(0,3,1,2)
            with torch.no_grad():
                _, out_pre = base_network(x_in.to(args.device))
                prob_pre = nn.Softmax(dim=1)(out_pre).cpu().numpy().squeeze()
            pre_probs.append(prob_pre)
        pre_probs = np.stack(pre_probs)  # shape (n_trials, class_num)
        np.savetxt(
            os.path.join(args.result_dir,
                         f"{args.data_name}_T-TIME_seed_{args.SEED}_pre_probs.csv"),
            pre_probs, delimiter=",",
        )

        # ─── run chunked adaptation...
        avg_tta, avg_post = run_chunked_tta(base_network, X_tar, y_tar, args, idt_str, extra_string)
        # save final adapted model
        best_ckpt_adapted = f'./runs/{args.data_name}/{args.backbone}_S{idt_str}_seed{args.SEED}{extra_string}_adapted.ckpt'
        torch.save(base_network.state_dict(), best_ckpt_adapted)

        # after final adaptation, save Post-TTA probabilities
        loader_post = DataLoader(
            TensorDataset(torch.from_numpy(X_tar).float(),
                          torch.from_numpy(y_tar).long()),
            batch_size=1, shuffle=False
        )
        post_probs = []
        base_network.eval()
        for x_post, _ in loader_post:
            # reshape to (1,1,chn,time) for EEGNet
            x_in = x_post
            if 'EEGNet' in args.backbone:
                x_in = x_in.unsqueeze(3).permute(0,3,1,2)
            with torch.no_grad():
                _, out_post = base_network(x_in.to(args.device))
                prob_post = nn.Softmax(dim=1)(out_post).cpu().numpy().squeeze()
            post_probs.append(prob_post)
        post_probs = np.stack(post_probs)
        np.savetxt(
            os.path.join(args.result_dir,
                         f"{args.data_name}_T-TIME_seed_{args.SEED}_post_probs.csv"),
            post_probs, delimiter=",",
        )

        # ─── dump per-seed predictions ─────────────────────────────────────────
        loader_all = DataLoader(
            TensorDataset(torch.from_numpy(X_tar).float(),
                          torch.from_numpy(y_tar).long()),
            batch_size=1, shuffle=False
        )
        _, y_pred_full, _ = TTIME(loader_all, base_network,
                                  args=args, balanced=args.balanced)
        # For CustomEpoch, output predictions one row per session.
        # write per-session predictions to CSV (handles varying lengths)
        out_path = os.path.join(args.result_dir,
                                f"{args.data_name}_T-TIME_seed_{args.SEED}_pred.csv")
        with open(out_path, 'w') as fw:
            if args.data_name == 'CustomEpoch':
                for start, end in args.tar_bounds:
                    row = y_pred_full[start:end]
                    fw.write(','.join(map(str, row.tolist())) + '\n')
            else:
                fw.write(','.join(map(str, y_pred_full.tolist())) + '\n')
        return avg_tta, pre_score, avg_post


# ─── Helper: run chunked TTA per session ──────────────────────────────────────
def run_chunked_tta(base_network, X_tar, y_tar, args, idt_str, extra_string):
    print('Executing per-session TTA with chunked batches...')
    sess_tta_accs, sess_post_accs = [], []
    for idx, (s, e) in enumerate(getattr(args, 'tar_bounds', [])):
        sess_name = args.session_names[idx]
        ts = torch.from_numpy(X_tar[s:e]).float()
        ys = torch.from_numpy(y_tar[s:e]).long()
        if 'EEGNet' in args.backbone:
            ts = ts.unsqueeze(3).permute(0,3,1,2)
        if args.data_env != 'local':
            ts, ys = ts.cuda(), ys.cuda()
        max_tta = getattr(args, 'max_tta', 20)
        num_rounds = min(3, len(ts) // max_tta)
        round_tta, round_post = [], []
        for r in range(num_rounds):
            start, end = r * max_tta, (r + 1) * max_tta
            loader_tta = prepare_loader(ts[start:end], ys[start:end], batch_size=1)

            # get TTA accuracy and per-trial probabilities
            acc_tta, y_pred_tta, sqrtRefEA = TTIME(loader_tta, base_network, args=args, balanced=args.balanced)
            # ensure 2D probs array
            if y_pred_tta.ndim == 1:
                probs2d = np.stack([1 - y_pred_tta, y_pred_tta], axis=1)
            else:
                probs2d = y_pred_tta
            preds_tta = np.argmax(probs2d, axis=1)
            true_tta = ys[start:end].cpu().numpy()
            # print each trial as "[idx: class1_prob : class0_prob, true, pred]"
            if args.print_trial_details:
                for j, (p0, p1, t, pr) in enumerate(zip(probs2d[:,0], probs2d[:,1], true_tta, preds_tta)):
                    print(f"[{start+j}: {p1:.4f} : {p0:.4f}, {t}, {pr}]")
                    args.log.record(f"[{start+j}: {p1:.4f} : {p0:.4f}, {t}, {pr}]")

            # align full test set with latest EA matrix
            aligned_ts = ts.squeeze(1).cpu().numpy()
            aligned = np.einsum('ij,njt->nit', sqrtRefEA, aligned_ts)
            ts_al = torch.from_numpy(aligned).unsqueeze(1).to(ts.dtype)
            if args.data_env != 'local':
                ts_al = ts_al.cuda()
            loader_post = prepare_loader(ts_al, ys, batch_size=1)

            # compute Post-TTA per-trial probabilities and accuracy
            probs_post = []; labels_post = []
            for x_post, y_post in loader_post:
                x_in = x_post.to(args.device)
                base_network.eval()
                with torch.no_grad():
                    _, out_post = base_network(x_in)
                    prob_post = nn.Softmax(dim=1)(out_post).detach().cpu().numpy()
                probs_post.append(prob_post)
                labels_post.append(y_post.item())
            probs_post_arr = np.vstack(probs_post)
            if args.balanced:
                preds_post = np.argmax(probs_post_arr, axis=1)
                post_acc = accuracy_score(labels_post, preds_post) * 100
            else:
                post_acc = roc_auc_score(labels_post, probs_post_arr[:,1]) * 100
            labels_post_arr = np.array(labels_post)
            preds_post = np.argmax(probs_post_arr, axis=1)
            combined_post = np.stack([labels_post_arr, preds_post], axis=1)
            # print each post-TTA trial
            if args.print_trial_details:
                for j, (p0, p1, t, pr) in enumerate(zip(probs_post_arr[:,0], probs_post_arr[:,1], labels_post_arr, preds_post)):
                    print(f"[{start+j}: {p1:.4f} : {p0:.4f}, {t}, {pr}]")
                    args.log.record(f"[{start+j}: {p1:.4f} : {p0:.4f}, {t}, {pr}]")

            print(f"  Session {sess_name} Round {r+1} TTA = {acc_tta:.2f}%, Post-TTA = {post_acc:.2f}%")
            args.log.record(f"  Session {sess_name} Round {r+1} TTA = {acc_tta:.2f}%, Post-TTA = {post_acc:.2f}%")
            # save this round’s adapted model
            torch.save(
                base_network.state_dict(),
                f'./runs/{args.data_name}/{args.backbone}_S{idt_str}_seed{args.SEED}{extra_string}_adapted_r{r+1}.ckpt'
            )
            round_tta.append(acc_tta); round_post.append(post_acc)
        sess_acc = float(np.mean(round_tta)) if round_tta else 0.0
        sess_post = float(np.mean(round_post)) if round_post else 0.0
        print(f"Session {sess_name} Avg TTA = {sess_acc:.2f}%, Avg Post-TTA = {sess_post:.2f}%")
        args.log.record(f"Session {sess_name} Avg TTA = {sess_acc:.2f}%, Avg Post-TTA = {sess_post:.2f}%")
        sess_tta_accs.append(sess_acc); sess_post_accs.append(sess_post)
    avg_tta = float(np.mean(sess_tta_accs)) if sess_tta_accs else 0.0
    avg_post = float(np.mean(sess_post_accs)) if sess_post_accs else 0.0
    print(f"Overall Avg TTA = {avg_tta:.2f}%, Overall Avg Post-TTA = {avg_post:.2f}%")
    args.log.record(f"Overall Avg TTA = {avg_tta:.2f}%, Overall Avg Post-TTA = {avg_post:.2f}%")
    print('Subject per-session TTA Accuracies:', np.round(sess_tta_accs,3))
    args.log.record(f"Subject per-session TTA Accuracies: {np.round(sess_tta_accs,3)}")
    print('Subject per-session Post-TTA Accuracies:', np.round(sess_post_accs,3))
    args.log.record(f"Subject per-session Post-TTA Accuracies: {np.round(sess_post_accs,3)}")
    return avg_tta, avg_post


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
            chn, class_num, time_sample_num, sample_rate = 31, 2, 1515, 200
            # F2 * (time_sample_num // 32)
            feature_deep_dim = 1504
            # use actual total trials across all sessions
            import pandas as _pd
            trial_num = int(_pd.read_csv('./data/CustomEpoch/meta.csv')['n_trials'].sum())
        else:
            raise ValueError(f"Unknown data_name {data_name}")

        # whether to use pretrained model
        # if source models have not been trained, set use_pretrained_model to False to train them
        # alternatively, run dnn.py to train source models, in seperating the steps
        use_pretrained_model = False
        if use_pretrained_model:
            # no training
            max_epoch = 0
        else:
            # training epochs
            max_epoch = 50

        # learning rate
        lr = 0.0005
        # test batch size
        test_batch = 12

        # update step
        steps = 7

        # update stride
        stride = 1


        # whether to use EA
        align = True

        # temperature rescaling, for test entropy calculation
        t = 1.8

        # whether to test balanced or imbalanced (2:1) target subject
        balanced = True

        # whether to record running time
        calc_time = False

        args = argparse.Namespace(feature_deep_dim=feature_deep_dim, align=align, lr=lr, t=t, max_epoch=max_epoch,
                                  trial_num=trial_num, time_sample_num=time_sample_num, sample_rate=sample_rate,
                                  N=N, chn=chn, class_num=class_num, stride=stride, steps=steps, calc_time=calc_time,
                                  paradigm=paradigm, test_batch=test_batch, data_name=data_name, balanced=balanced)
        # control trial-level detail printing
        args.print_trial_details = False

        args.method = 'T-TIME'
        args.backbone = 'EEGNet'

        # train batch size
        args.batch_size = 64

        # GPU device id
        # detect device and default to GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        args.device = device
        args.data_env = 'gpu' if torch.cuda.is_available() else 'local'
        
        # Make sure args.data is set correctly before initializing logging
        args.data = data_name
        
        # Initialize logging before starting subject processing
        args.local_dir = './data/' + str(data_name) + '/'
        args.result_dir = './logs/'
        my_log = LogRecord(args)
        my_log.log_init()
        my_log.record('=' * 50 + '\n' + os.path.basename(__file__) + '\n' + '=' * 50)
        args.log = my_log
        
        # Initialize storage for results
        seeds = [2, 3, 5, 6, 7, 8, 9, 10, 11, 42]
        total_acc = np.zeros((len(seeds), N))
        pre_acc_all_seeds = np.zeros((len(seeds), N))
        post_acc_all_seeds = np.zeros((len(seeds), N))
        # collect ensemble accuracies per subject
        ensemble_tta_all = []
        ensemble_pre_all = []
        ensemble_post_all = []

        # Iterate through each subject first
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
            args.log.record(info_str)
            
            # Now run all seeds for this subject
            for seed_idx, s in enumerate(seeds):
                args.SEED = s
                print(f"\n--- Running Subject {target_str} with Seed {s} ---")
                args.log.record(f"\n--- Running Subject {target_str} with Seed {s} ---")

                fix_random_seed(args.SEED)
                torch.backends.cudnn.deterministic = True

                args.data = data_name
                print(f"Data: {args.data}, Method: {args.method}, Seed: {args.SEED}")
                args.log.record(f"Data: {args.data}, Method: {args.method}, Seed: {args.SEED}")

                # Run training and evaluation for this subject and seed
                tta_acc, pre_acc, post_acc = train_target(args)
                
                # Store results
                total_acc[seed_idx, idt] = tta_acc
                pre_acc_all_seeds[seed_idx, idt] = pre_acc
                post_acc_all_seeds[seed_idx, idt] = post_acc
                
                # Log results for this subject and seed
                print(f"Subject {target_str} with Seed {s} - TTA Acc: {tta_acc:.3f}, Pre-TTA: {pre_acc:.3f}, Post-TTA: {post_acc:.3f}")
                args.log.record(f"Subject {target_str} with Seed {s} - TTA Acc: {tta_acc:.3f}, Pre-TTA: {pre_acc:.3f}, Post-TTA: {post_acc:.3f}")
                
                # Save per-seed, per-subject results
                np.savetxt(
                    os.path.join(args.result_dir, f"{data_name}_T-TIME_seed_{args.SEED}_subject_{idt}_acc.csv"),
                    np.array([tta_acc, pre_acc, post_acc]), delimiter=","
                )
                
                # Clear GPU memory between seeds
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # After all seeds for this subject, print summary for this subject
            subject_mean_tta = np.mean(total_acc[:, idt])
            subject_std_tta = np.std(total_acc[:, idt])
            subject_mean_pre = np.mean(pre_acc_all_seeds[:, idt])
            subject_std_pre = np.std(pre_acc_all_seeds[:, idt])
            subject_mean_post = np.mean(post_acc_all_seeds[:, idt])
            subject_std_post = np.std(post_acc_all_seeds[:, idt])
            
            # Calculate 95% confidence intervals (for small sample sizes using t-distribution)
            from scipy import stats
            n_seeds = len(seeds)
            confidence = 0.95
            # t-value for 95% confidence with n-1 degrees of freedom
            t_val = stats.t.ppf((1 + confidence) / 2, n_seeds - 1)
            ci_tta = t_val * (subject_std_tta / np.sqrt(n_seeds))
            ci_pre = t_val * (subject_std_pre / np.sqrt(n_seeds))
            ci_post = t_val * (subject_std_post / np.sqrt(n_seeds))
            
            print(f"\n=== Subject {target_str} Summary Across All Seeds ===")
            print(f"TTA Accuracy: {subject_mean_tta:.3f} ± {subject_std_tta:.3f} (95% CI: {subject_mean_tta-ci_tta:.3f} to {subject_mean_tta+ci_tta:.3f})")
            print(f"Pre-TTA Accuracy: {subject_mean_pre:.3f} ± {subject_std_pre:.3f} (95% CI: {subject_mean_pre-ci_pre:.3f} to {subject_mean_pre+ci_pre:.3f})")
            print(f"Post-TTA Accuracy: {subject_mean_post:.3f} ± {subject_std_post:.3f} (95% CI: {subject_mean_post-ci_post:.3f} to {subject_mean_post+ci_post:.3f})")
            args.log.record(f"\n=== Subject {target_str} Summary Across All Seeds ===")
            args.log.record(f"TTA Accuracy: {subject_mean_tta:.3f} ± {subject_std_tta:.3f} (95% CI: {subject_mean_tta-ci_tta:.3f} to {subject_mean_tta+ci_tta:.3f})")
            args.log.record(f"Pre-TTA Accuracy: {subject_mean_pre:.3f} ± {subject_std_pre:.3f} (95% CI: {subject_mean_pre-ci_pre:.3f} to {subject_mean_pre+ci_pre:.3f})")
            args.log.record(f"Post-TTA Accuracy: {subject_mean_post:.3f} ± {subject_std_post:.3f} (95% CI: {subject_mean_post-ci_post:.3f} to {subject_mean_post+ci_post:.3f})")

            # Ensemble evaluation across seeds for this subject
            # reload target labels for this subject
            X_src, y_src, X_tar_sub, y_tar_sub = read_mi_combine_tar(args)

            preds_seeds = []
            for s in seeds:
                pred_file = os.path.join(args.result_dir,
                    f"{args.data_name}_T-TIME_seed_{s}_pred.csv")
                # load ragged CSV: read each line and extend
                flat = []
                with open(pred_file, 'r') as fr:
                    for line in fr:
                        parts = line.strip().split(',')
                        flat.extend([float(x) for x in parts if x])
                preds_seeds.append(np.array(flat))
            preds_arr = np.stack(preds_seeds, axis=0)  # now shape (n_seeds, n_trials)

            # ─── Session‐level TTA ensembles & averaged summary ─────────────────
            session_accs = [[], [], [], []]  # maj, mean, med, sml
            print(f"\nSession-level TTA ensembles for subject {target_str}:")
            args.log.record(f"Session-level TTA ensembles for subject {target_str}:")
            for idx, (start, end) in enumerate(args.tar_bounds):
                true_s = y_tar_sub[start:end]
                sess_mat = preds_arr[:, start:end]
                # 1) Majority on hard labels
                labels_s = (sess_mat > 0.5).astype(int)
                maj_s = (labels_s.sum(axis=0) >= len(seeds)/2).astype(int)
                maj_acc = accuracy_score(true_s, maj_s) * 100
                # 2) MeanProb with session Youden’s J
                mean_probs = sess_mat.mean(axis=0)
                fpr, tpr, ths = roc_curve(true_s, mean_probs)
                best_t = ths[np.argmax(tpr - fpr)]
                mean_s = (mean_probs > best_t).astype(int)
                mean_acc = accuracy_score(true_s, mean_s) * 100
                # 3) MedianProb with session Youden’s J
                med_probs = np.median(sess_mat, axis=0)
                fpr2, tpr2, ths2 = roc_curve(true_s, med_probs)
                best_t2 = ths2[np.argmax(tpr2 - fpr2)]
                med_s = (med_probs > best_t2).astype(int)
                med_acc = accuracy_score(true_s, med_s) * 100
                # 4) SML on hard labels
                sml_s = SML(labels_s)
                sml_acc = accuracy_score(true_s, sml_s) * 100

                # collect
                for lst, v in zip(session_accs, (maj_acc, mean_acc, med_acc, sml_acc)):
                    lst.append(v)

                line = (f" Session {idx+1}: Majority={maj_acc:.2f}%, "
                        f"MeanProb={mean_acc:.2f}%, MedianProb={med_acc:.2f}%, SML={sml_acc:.2f}%")
                print(line); args.log.record(line)

            # averaged across sessions
            avg_maj, avg_mean, avg_med, avg_sml = [np.mean(lst) for lst in session_accs]
            print(f"\nEnsemble TTA (avg of sessions): Majority={avg_maj:.2f}%, "
                  f"MeanProb={avg_mean:.2f}%, MedianProb={avg_med:.2f}%, SML={avg_sml:.2f}%")
            args.log.record(f"Ensemble TTA (avg of sessions): Majority={avg_maj:.2f}%, "
                            f"MeanProb={avg_mean:.2f}%, MedianProb={avg_med:.2f}%, SML={avg_sml:.2f}%")

  
            pre_list = []
            for s in seeds:
                pp = np.loadtxt(os.path.join(args.result_dir,
                    f"{args.data_name}_T-TIME_seed_{s}_pre_probs.csv"), delimiter=',')
                prob1 = pp[:,1] if pp.ndim>1 else pp
                pre_list.append(prob1)
            pre_mat = np.stack(pre_list, axis=0)  # shape (n_seeds, n_trials)
            pre_labels = (pre_mat > 0.5).astype(int)
            # majority
            pre_maj = (pre_labels.sum(axis=0) >= (len(seeds)/2)).astype(int)
            # mean & median decisions
            pre_mean_vote = (pre_mat.mean(axis=0) > 0.5).astype(int)
            pre_med_vote  = (np.median(pre_mat, axis=0) > 0.5).astype(int)
            # SML on hard labels
            pre_sml = SML(pre_labels)
            # accuracies
            acc_pre_maj  = accuracy_score(y_tar_sub, pre_maj) * 100
            acc_pre_mean = accuracy_score(y_tar_sub, pre_mean_vote) * 100
            acc_pre_med  = accuracy_score(y_tar_sub, pre_med_vote) * 100
            acc_pre_sml  = accuracy_score(y_tar_sub, pre_sml) * 100
            # ─── Session-average Pre-TTA ─────────────────────────────────────────
            session_pre = [[], [], [], []]  # maj, mean, med, sml
            for idx, (s, e) in enumerate(args.tar_bounds):
                y_true_s = y_tar_sub[s:e]
                mat = pre_mat[:, s:e]
                # majority
                maj = ((mat > 0.5).sum(axis=0) >= len(seeds)/2).astype(int)
                session_pre[0].append(accuracy_score(y_true_s, maj)*100)
                # mean + Youden
                mp = mat.mean(axis=0)
                fpr, tpr, ths = roc_curve(y_true_s, mp)
                thr = ths[np.argmax(tpr-fpr)]
                session_pre[1].append(accuracy_score(y_true_s,(mp>thr).astype(int))*100)
                # median + Youden
                md = np.median(mat,axis=0)
                fpr2, tpr2, ths2 = roc_curve(y_true_s, md)
                thr2 = ths2[np.argmax(tpr2-fpr2)]
                session_pre[2].append(accuracy_score(y_true_s,(md>thr2).astype(int))*100)
                # SML
                session_pre[3].append(accuracy_score(y_true_s,SML((mat>0.5).astype(int)))*100)
            avg_pre = [np.mean(lst) for lst in session_pre]
            print(f"\nEnsemble Pre-TTA (avg of sessions): Majority={avg_pre[0]:.2f}%, MeanProb={avg_pre[1]:.2f}%, MedianProb={avg_pre[2]:.2f}%, SML={avg_pre[3]:.2f}%")
            args.log.record(f"Ensemble Pre-TTA (avg of sessions): Majority={avg_pre[0]:.2f}%, MeanProb={avg_pre[1]:.2f}%, MedianProb={avg_pre[2]:.2f}%, SML={avg_pre[3]:.2f}%")

            # ─── Session-level Pre-TTA ensembles ─────────────────────────
            print(f"\nSession-level Pre-TTA ensembles for subject {target_str}:")
            args.log.record(f"Session-level Pre-TTA ensembles for subject {target_str}:")
            for idx, (start, end) in enumerate(args.tar_bounds):
                y_true_s = y_tar_sub[start:end]
                sess_mat = pre_mat[:, start:end]
                # per-session majority
                maj_s = (sess_mat > 0.5).sum(axis=0) >= (len(seeds)/2)
                # per-session mean + Youden’s J threshold
                mean_probs = sess_mat.mean(axis=0)
                fpr, tpr, ths = roc_curve(y_true_s, mean_probs)
                best_t = ths[np.argmax(tpr - fpr)]
                mean_s = mean_probs > best_t
                # per-session median + Youden’s J
                med_probs = np.median(sess_mat, axis=0)
                fpr2, tpr2, ths2 = roc_curve(y_true_s, med_probs)
                best_t2 = ths2[np.argmax(tpr2 - fpr2)]
                med_s = med_probs > best_t2
                # per-session SML on hard labels
                sml_s = SML((sess_mat > 0.5).astype(int))
                accs = (
                    accuracy_score(y_true_s, maj_s) * 100,
                    accuracy_score(y_true_s, mean_s) * 100,
                    accuracy_score(y_true_s, med_s) * 100,
                    accuracy_score(y_true_s, sml_s) * 100
                )
                line = (f" Session {idx+1}: Majority={accs[0]:.2f}%, MeanProb={accs[1]:.2f}%, "
                        f"MedianProb={accs[2]:.2f}%, SML={accs[3]:.2f}%")
                print(line); args.log.record(line)

            # Ensemble summary for Post-TTA probabilities across seeds
            # load per-trial post-TTA probs and stack
            post_list = []
            for s in seeds:
                pp = np.loadtxt(os.path.join(args.result_dir,
                    f"{args.data_name}_T-TIME_seed_{s}_post_probs.csv"), delimiter=',')
                prob1 = pp[:,1] if pp.ndim>1 else pp
                post_list.append(prob1)
            post_mat = np.stack(post_list, axis=0)
            post_labels = (post_mat > 0.5).astype(int)
            post_maj = (post_labels.sum(axis=0) >= (len(seeds)/2)).astype(int)
            post_mean_vote = (post_mat.mean(axis=0) > 0.5).astype(int)
            post_med_vote  = (np.median(post_mat, axis=0) > 0.5).astype(int)
            post_sml = SML(post_labels)
            acc_post_maj  = accuracy_score(y_tar_sub, post_maj) * 100
            acc_post_mean = accuracy_score(y_tar_sub, post_mean_vote) * 100
            acc_post_med  = accuracy_score(y_tar_sub, post_med_vote) * 100
            acc_post_sml  = accuracy_score(y_tar_sub, post_sml) * 100
            # ─── Session-average Post-TTA ──────────────────────────────────────
            session_post = [[], [], [], []]
            for idx, (s, e) in enumerate(args.tar_bounds):
                y_true_s = y_tar_sub[s:e]
                mat = post_mat[:, s:e]
                session_post[0].append(accuracy_score(y_true_s, ((mat>0.5).sum(axis=0)>=len(seeds)/2).astype(int))*100)
                mp = mat.mean(axis=0)
                fpr3, tpr3, ths3 = roc_curve(y_true_s, mp)
                thr3 = ths3[np.argmax(tpr3-fpr3)]
                session_post[1].append(accuracy_score(y_true_s,(mp>thr3).astype(int))*100)
                md = np.median(mat,axis=0)
                fpr4, tpr4, ths4 = roc_curve(y_true_s, md)
                thr4 = ths4[np.argmax(tpr4-fpr4)]
                session_post[2].append(accuracy_score(y_true_s,(md>thr4).astype(int))*100)
                session_post[3].append(accuracy_score(y_true_s,SML((mat>0.5).astype(int)))*100)
            avg_post = [np.mean(lst) for lst in session_post]
            print(f"\nEnsemble Post-TTA (avg of sessions): Majority={avg_post[0]:.2f}%, MeanProb={avg_post[1]:.2f}%, MedianProb={avg_post[2]:.2f}%, SML={avg_post[3]:.2f}%")
            args.log.record(f"Ensemble Post-TTA (avg of sessions): Majority={avg_post[0]:.2f}%, MeanProb={avg_post[1]:.2f}%, MedianProb={avg_post[2]:.2f}%, SML={avg_post[3]:.2f}%")

            # ─── Session-level Post-TTA ensembles ────────────────────────────
            print(f"\nSession-level Post-TTA ensembles for subject {target_str}:")
            args.log.record(f"Session-level Post-TTA ensembles for subject {target_str}:")
            for idx, (start, end) in enumerate(args.tar_bounds):
                y_true_s = y_tar_sub[start:end]
                sess_mat = post_mat[:, start:end]
                maj_s   = (sess_mat > 0.5).sum(axis=0) >= (len(seeds)/2)
                mean_probs = sess_mat.mean(axis=0)
                fpr3, tpr3, ths3 = roc_curve(y_true_s, mean_probs)
                best_t3 = ths3[np.argmax(tpr3 - fpr3)]
                mean_s = mean_probs > best_t3
                med_probs = np.median(sess_mat, axis=0)
                fpr4, tpr4, ths4 = roc_curve(y_true_s, med_probs)
                best_t4 = ths4[np.argmax(tpr4 - fpr4)]
                med_s = med_probs > best_t4
                sml_s = SML((sess_mat > 0.5).astype(int))
                accs = (
                    accuracy_score(y_true_s, maj_s) * 100,
                    accuracy_score(y_true_s, mean_s) * 100,
                    accuracy_score(y_true_s, med_s) * 100,
                    accuracy_score(y_true_s, sml_s) * 100
                )
                line = (f" Session {idx+1}: Majority={accs[0]:.2f}%, MeanProb={accs[1]:.2f}%, "
                        f"MedianProb={accs[2]:.2f}%, SML={accs[3]:.2f}%")
                print(line); args.log.record(line)

        # ─── Final summary across all subjects ───────────────────────────────
        print("=== Per-seed × per-subject Accuracies ===")
        header = "Subjects: " + ", ".join(subject_names)
        print(header); args.log.record(header)
        seed_line = "Seeds: " + ", ".join(map(str, seeds))
        print(seed_line); args.log.record(seed_line)

        # TTA matrix
        print("TTA Accs (rows=seeds, cols=subjects):")
        print(np.round(total_acc, 3))
        args.log.record("TTA Accs:\n" + np.array2string(np.round(total_acc, 3)))

        # Pre-TTA matrix
        print("Pre-TTA Accs (rows=seeds, cols=subjects):")
        print(np.round(pre_acc_all_seeds, 3))
        args.log.record("Pre-TTA Accs:\n" + np.array2string(np.round(pre_acc_all_seeds, 3)))

        # Post-TTA matrix
        print("Post-TTA Accs (rows=seeds, cols=subjects):")
        print(np.round(post_acc_all_seeds, 3))
        args.log.record("Post-TTA Accs:\n" + np.array2string(np.round(post_acc_all_seeds, 3)))

        # Per-seed means
        tta_seed_mean  = np.round(np.mean(total_acc, axis=1), 3)
        pre_seed_mean  = np.round(np.mean(pre_acc_all_seeds, axis=1), 3)
        post_seed_mean = np.round(np.mean(post_acc_all_seeds, axis=1), 3)
        print("Overall per-seed means:")
        print("  TTA:    ", tta_seed_mean)
        print("  Pre-TTA:", pre_seed_mean)
        print("  Post-TTA:", post_seed_mean)
        args.log.record(f"Overall per-seed TTA mean:  {tta_seed_mean}")
        args.log.record(f"Overall per-seed Pre-TTA mean:  {pre_seed_mean}")
        args.log.record(f"Overall per-seed Post-TTA mean: {post_seed_mean}")

        # Grand means
        overall_tta  = np.mean(total_acc)
        overall_pre  = np.mean(pre_acc_all_seeds)
        overall_post = np.mean(post_acc_all_seeds)
        summary = (
            f"Grand Means:\n"
            f"  Overall TTA Acc:   {overall_tta:.2f}%\n"
            f"  Overall Pre-TTA:    {overall_pre:.2f}%\n"
            f"  Overall Post-TTA:  {overall_post:.2f}%"
        )
        print(summary)
        args.log.record(summary)

        # ─── Grand ensemble means across subjects ─────────────────────────
        mt = np.array(ensemble_tta_all)
        mp = np.array(ensemble_pre_all)
        ms = np.array(ensemble_post_all)
        gm = (
            f"Grand Ensemble Means:\n"
            f"  TTA    : Majority={mt[:,0].mean():.2f}%, MeanProb={mt[:,1].mean():.2f}%, "
            f"Median={mt[:,2].mean():.2f}%, SML={mt[:,3].mean():.2f}%\n"
            f"  Pre-TTA: Majority={mp[:,0].mean():.2f}%, MeanProb={mp[:,1].mean():.2f}%, "
            f"Median={mp[:,2].mean():.2f}%, SML={mp[:,3].mean():.2f}%\n"
            f"  Post-TTA: Majority={ms[:,0].mean():.2f}%, MeanProb={ms[:,1].mean():.2f}%, "
            f"Median={ms[:,2].mean():.2f}%, SML={ms[:,3].mean():.2f}%"
        )
        print(gm)
        args.log.record(gm)
    # end of for data_name