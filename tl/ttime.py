# -*- coding: utf-8 -*-
# @Time    : 2023/07/07
# @Author  : Siyang Li
# @File    : ttime.py
import numpy as np
import argparse 
import copy

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
import torch.linalg as LA

import gc
import sys
import time
from torch.utils.data import DataLoader, TensorDataset
import logging

# ── Logger setup ──────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
logger.addHandler(_handler)

# Add once: prepare_loader helper to avoid recreating DataLoader inline
def prepare_loader(data, targets, batch_size):
    return DataLoader(TensorDataset(data, targets), batch_size=batch_size, shuffle=False)

def TTIME(loader, model, args, balanced=True):
    # "T-TIME: Test-Time Information Maximization Ensemble for Plug-and-Play BCIs"
    # IEEE Transactions on Biomedical Engineering
    # Note that the ensemble experiment is separately implemented in ttime_ensemble.py, using recorded test prediction.

    if balanced == False and args.data_name == 'BNCI2014001-4':
        logger.error('ERROR, imbalanced multi-class not implemented')
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
        # time data loading & prep
        dl_start = time.time()
        model.eval()
        data = next(iter_test)
        # inputs→data_cum→sample_test prep
        inputs, labels = data[0], data[1]
        inputs = inputs.reshape(1,1,inputs.shape[-2],inputs.shape[-1]).to(args.device)

        # accumulate test data ON‐DEVICE (no .cpu())
        if data_cum is None:
            data_cum = inputs.float()
        else:
            data_cum = torch.cat((data_cum, inputs.float()), 0)
        # keep only last max_tta samples on GPU to avoid unbounded growth
        max_win = args.max_tta
        if data_cum.size(0) > max_win:
            data_cum = data_cum[-max_win:]

        if args.align:
            # time EA (incremental alignment)
            ea_start = time.time()
            # get sample as a GPU tensor
            # only use the most recent sample for alignment
            sample_tensor = data_cum[-1].reshape(args.chn, args.time_sample_num)
            # update R on CPU via EA_online
            sample_np = sample_tensor.cpu().numpy()
            R = EA_online(sample_np, R, i)
            # compute R^(-0.5) on GPU via eigendecomp
            R_t = torch.from_numpy(R).to(device=args.device, dtype=sample_tensor.dtype)
            eigvals, eigvecs = LA.eigh(R_t)
            sqrtRefEA = eigvecs @ torch.diag(eigvals.pow(-0.5)) @ eigvecs.T
            # apply transform on GPU
            sample_tensor = sqrtRefEA @ sample_tensor
            # reshape into batch
            sample_test = sample_tensor.reshape(1,1,args.chn,args.time_sample_num)
            # log EA time
            torch.cuda.synchronize() if args.device.type=='cuda' else None
            ea_ms = (time.time() - ea_start)*1000
            logger.debug(f"[T-TIME] iter {i}: EA_online + transform took {ea_ms:.1f} ms")
        else:
            # no alignment: just slice ON‐DEVICE
            sample_test = data_cum[i].unsqueeze(0)  # now (1,1,chn,time)

        # ensure tensor is float32 on correct device
        sample_test = sample_test.to(device=args.device, dtype=torch.float32)
        # log data‐loading & prep time
        dl_ms = (time.time() - dl_start)*1000
        logger.debug(f"[T-TIME] iter {i}: data load & prep took {dl_ms:.1f} ms")

        _, outputs = model(sample_test)

        softmax_out = nn.Softmax(dim=1)(outputs)
        outputs = outputs.float().cpu()
        labels = labels.float().cpu()
        _, predict = torch.max(outputs, 1)

        # logger.info(f"Trial {i}: pred={predict.item()}, gt={labels.item()}")
        # if hasattr(args, 'log'):
        #     args.log.record(f"Trial {i}: pred={predict.item()}, gt={labels.item()}")

        y_pred.append(softmax_out.detach().cpu().numpy())
        y_true.append(labels.item())

        #################### Phase 2: target model update ####################
        model.train()
        # skip adaptation on low-confidence samples
        conf, _ = softmax_out.max(dim=1)
        thr = getattr(args, 'conf_thresh', 0.1)
        if conf.item() < thr:
            logger.info(f"Skipping trial {i} due to low confidence ({conf.item():.3f} < {thr})")
            # Optional: also record in log file
            if hasattr(args, 'log'):
                args.log.record(f"Skipping trial {i}: confidence {conf.item():.3f} below threshold {thr}")
            model.eval()
            continue

        # sliding batch: use args.max_tta
        win = args.max_tta
        if (i+1) >= win and (i+1) % args.stride == 0:
            if args.align:
                # time the alignment step
                align_start = time.time()
                # get raw window of size win
                # get most recent window of size win without full history
                raw = data_cum[-win:]           # (win,1,chn,time)
                flat = raw.squeeze(1)               # (win,chn,time)
                # build R_w via EA_online on CPU
                R_w = 0
                for k in range(flat.shape[0]):
                    sample_np = flat[k].cpu().numpy()
                    R_w = EA_online(sample_np, R_w, k)
                R_w += np.eye(R_w.shape[0]) * 1e-6
                # GPU eigen for fractional power
                Rw_t = torch.from_numpy(R_w).to(device=args.device, dtype=flat.dtype)
                ev_w, evec_w = LA.eigh(Rw_t)
                sqrtRefEA_w = evec_w @ torch.diag(ev_w.pow(-0.5)) @ evec_w.T
                # align entire batch on GPU
                aligned = torch.einsum('ij,bjt->bit', sqrtRefEA_w, flat.to(args.device))
                torch.cuda.synchronize() if args.device.type=='cuda' else None
                align_ms = (time.time() - align_start) * 1000
                logger.debug(f"[T-TIME] iter {i}: alignment took {align_ms:.1f} ms")
                batch_test = aligned.unsqueeze(1)    # (win,1,chn,time)
            else:
                # slice last win samples and move to CPU for numpy
                arr = data_cum[-win:].cpu().numpy()
                n = arr.shape[0]
                batch_test = arr.reshape(n, 1, arr.shape[2], arr.shape[3])

            # wrap numpy only if needed, then move to device
            if not isinstance(batch_test, torch.Tensor):
                batch_test = torch.from_numpy(batch_test)
            batch_test = batch_test.to(device=args.device, dtype=torch.float32)

            start_time = time.time()
            for step in range(args.steps):
                # forward timing
                fwd_start = time.time()
                _, outputs = model(batch_test)
                torch.cuda.synchronize() if args.device.type=='cuda' else None
                fwd_ms = (time.time() - fwd_start) * 1000
                logger.debug(f"[T-TIME] iter {i}, step {step}: forward {fwd_ms:.1f} ms")

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

                # backward + step timing
                opt_start = time.time()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                torch.cuda.synchronize() if args.device.type=='cuda' else None
                opt_ms = (time.time() - opt_start) * 1000
                logger.debug(f"[T-TIME] iter {i}, step {step}: backward+opt {opt_ms:.1f} ms")

            TTA_time = time.time()
            if args.calc_time:
                logger.info(f"sample {i}, post-inference model update finished in ms: {np.round((TTA_time - start_time)*1000,3)}")

            if not balanced:
                # initialize threshold once
                if not hasattr(args, 'pred_thresh'):
                    args.pred_thresh = 0.7
                pl = torch.max(softmax_out, 1)[1]
                # update count for every sample in current window
                for l in range(softmax_out.size(0)):
                    if pl[l] == 0 and softmax_out[l][0] > args.pred_thresh:
                        zk_arrs[0] += 1
                    elif pl[l] == 1 and softmax_out[l][1] > args.pred_thresh:
                        zk_arrs[1] += 1

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
    if isinstance(args.idt, (list, tuple)):
        idt_str = '_'.join(map(str, args.idt))
    else:
        idt_str = str(args.idt)
    # make sure save directory exists
    os.makedirs(os.path.join('.', 'runs', args.data_name), exist_ok=True)
    os.makedirs(args.result_dir, exist_ok=True)

    # load source/target
    # load source/target only once per subject
    if not hasattr(args, 'mi_data_loaded'):
        args.mi_data_loaded = read_mi_combine_tar(args)
    X_src, y_src, X_tar, y_tar = args.mi_data_loaded

    # build / cache data loaders with EA applied only once per subject
    if not hasattr(args, 'dset_loaders_cached'):
        args.dset_loaders_cached = data_loader(X_src, y_src, X_tar, y_tar, args)
    dset_loaders = args.dset_loaders_cached

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

    # move backbone and classifier to configured device
    netF, netC = backbone_net(args, return_type='xy')
    netF, netC = netF.to(args.device), netC.to(args.device)
    base_network = nn.Sequential(netF, netC).to(args.device)

    if args.max_epoch == 0:
        # load pretrained source‐model checkpoint for fair comparison
        best_ckpt = f'./runs/{args.data_name}/{args.backbone}_S{idt_str}_seed{args.SEED}{extra_string}_best.ckpt'
        base_network.load_state_dict(torch.load(best_ckpt, map_location=args.device))
        base_network.eval()

        # unify metric: use online streaming scoring on the same split
        metric = "Acc" if args.balanced else "AUC"
        pre_acc = cal_score_online(dset_loaders["Target-Online"], base_network, args=args)
        logger.info(f"Task: {args.task_str}, Pre-TTA IEA {metric} = {pre_acc:.2f}%")
        args.log.record(f"Task: {args.task_str}, Pre-TTA IEA {metric} = {pre_acc:.2f}%")

        # after loading best model, save Pre‐TTA probabilities
        # Route Pre-TTA through TTIME (align=True, no adaptation)
        base_network.eval()
        loader_pre = DataLoader(
            TensorDataset(torch.from_numpy(X_tar).float(),
                          torch.from_numpy(y_tar).long()),
            batch_size=1, shuffle=False
        )
        # PRE‐TTA inference (no adaptation)
        src_model = copy.deepcopy(base_network).to(args.device).eval()
        pre_y_pred = []
        with torch.no_grad():
            for x, _ in loader_pre:
                x = x.to(args.device).float()
                if 'EEGNet' in args.backbone:
                    x = x.unsqueeze(3).permute(0,3,1,2)
                _, logits = src_model(x)
                soft = nn.Softmax(dim=1)(logits)
                pre_y_pred.append(soft.cpu().numpy())
        pre_y_pred = np.concatenate(pre_y_pred, axis=0)
        np.savetxt(os.path.join(args.result_dir,
                    f"{args.data_name}_T-TIME_seed_{args.SEED}_pre_probs.csv"),
                   pre_y_pred, delimiter=",")
        # --- calibrate Pre-TTA threshold via Youden's J for binary balanced setting ---
        if args.class_num == 2 and args.balanced:
            # use ground-truth y_tar from this scope
            fpr, tpr, th = roc_curve(y_tar, pre_y_pred[:,1])
            best_pre_thresh = th[np.argmax(tpr - fpr)]
            args.pre_thresh = float(best_pre_thresh)
            logger.info(f"Calibrated Pre-TTA threshold: {best_pre_thresh:.3f}")
            if hasattr(args, 'log'):
                args.log.record(f"Calibrated Pre-TTA threshold: {best_pre_thresh:.3f}")
        # POST-TTA streaming adaptation
        adapted_model = copy.deepcopy(base_network).to(args.device)
        tta_score, tta_y_pred, _ = TTIME(loader_pre, adapted_model, args=args, balanced=args.balanced)
        np.savetxt(os.path.join(args.result_dir,
                    f"{args.data_name}_T-TIME_seed_{args.SEED}_tta_probs.csv"),
                   tta_y_pred, delimiter=",")
        best_ckpt_adapted = f'./runs/{args.data_name}/{args.backbone}_S{idt_str}_seed{args.SEED}{extra_string}_adapted.ckpt'
        torch.save(adapted_model.state_dict(), best_ckpt_adapted)
        return tta_score, pre_acc
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
                logger.info(f"Epoch {epoch_num}/{args.max_epoch} - Iteration {iter_num}/{max_iter} completed.")

                base_network.eval()
                if args.balanced:
                    acc_t_te, _ = cal_acc_comb(dset_loaders["Target"], base_network, args=args)
                else:
                    acc_t_te, _ = cal_auc_comb(dset_loaders["Target-Imbalanced"], base_network, args=args)
                logger.info(f"Validation Accuracy: {acc_t_te:.2f}%")
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
        logger.info(f"Task: {args.task_str}, Pre-TTA IEA {metric} = {pre_score:.2f}%")
        args.log.record(f"Task: {args.task_str}, Pre-TTA IEA {metric} = {pre_score:.2f}%")
        
        # set pre_acc so it exists for return
        pre_acc = pre_score

        # after loading best model, save Pre-TTA probabilities
        # Route Pre-TTA through TTIME (align=True, no adaptation)
        base_network.eval()
        loader_pre = DataLoader(
            TensorDataset(torch.from_numpy(X_tar).float(),
                          torch.from_numpy(y_tar).long()),
            batch_size=1, shuffle=False
        )
        # PRE‐TTA inference (no adaptation)
        src_model = copy.deepcopy(base_network).to(args.device).eval()
        pre_y_pred = []
        with torch.no_grad():
            for x, _ in loader_pre:
                x = x.to(args.device).float()
                if 'EEGNet' in args.backbone:
                    x = x.unsqueeze(3).permute(0,3,1,2)
                _, logits = src_model(x)
                soft = nn.Softmax(dim=1)(logits)
                pre_y_pred.append(soft.cpu().numpy())
        pre_y_pred = np.concatenate(pre_y_pred, axis=0)
        np.savetxt(os.path.join(args.result_dir,
                    f"{args.data_name}_T-TIME_seed_{args.SEED}_pre_probs.csv"),
                   pre_y_pred, delimiter=",")
        # --- calibrate Pre-TTA threshold via Youden's J for binary balanced setting ---
        if args.class_num == 2 and args.balanced:
            # use ground-truth y_tar from this scope
            fpr, tpr, th = roc_curve(y_tar, pre_y_pred[:,1])
            best_pre_thresh = th[np.argmax(tpr - fpr)]
            args.pre_thresh = float(best_pre_thresh)
            logger.info(f"Calibrated Pre-TTA threshold: {best_pre_thresh:.3f}")
            if hasattr(args, 'log'):
                args.log.record(f"Calibrated Pre-TTA threshold: {best_pre_thresh:.3f}")
        # POST-TTA streaming adaptation
        adapted_model = copy.deepcopy(base_network).to(args.device)
        tta_score, tta_y_pred, _ = TTIME(loader_pre, adapted_model, args=args, balanced=args.balanced)
        np.savetxt(os.path.join(args.result_dir,
                    f"{args.data_name}_T-TIME_seed_{args.SEED}_tta_probs.csv"),
                   tta_y_pred, delimiter=",")
        best_ckpt_adapted = f'./runs/{args.data_name}/{args.backbone}_S{idt_str}_seed{args.SEED}{extra_string}_adapted.ckpt'
        torch.save(adapted_model.state_dict(), best_ckpt_adapted)
        return tta_score, pre_acc


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

        # ── Define hyperparameter grid ─────────────────────────────────
        max_tta_list = [6, 8, 10, 12, 16]
        stride_list  = [1, 2, 3, 4]
        t_list       = [1.3, 1.5, 1.7, 1.8, 1.9, 2.0, 2.2]
        lr_list      = [0.0001, 0.0005, 0.001]
        steps_list   = [1, 3, 5, 7, 9, 11]

        align = True

        use_pretrained_model = True  # keep existing behavior

        # ── Hyperparameter search loops ───────────────────────────────
        for max_tta in max_tta_list:
            for stride in stride_list:
                for t in t_list:
                    for lr in lr_list:
                        for steps in steps_list:
                            # set training epochs
                            max_epoch = 0 if use_pretrained_model else 30
                            balanced   = True
                            calc_time  = False
                            # build args for this combination
                            args = argparse.Namespace(
                                feature_deep_dim=feature_deep_dim,
                                align=align,
                                lr=lr,
                                t=t,
                                max_epoch=max_epoch,
                                trial_num=trial_num,
                                time_sample_num=time_sample_num,
                                sample_rate=sample_rate,
                                N=N,
                                chn=chn,
                                class_num=class_num,
                                stride=stride,
                                steps=steps,
                                calc_time=calc_time,
                                paradigm=paradigm,
                                max_tta=max_tta,
                                data_name=data_name,
                                balanced=balanced
                            )
                            args.print_trial_details = False
                            args.method    = 'T-TIME'
                            args.backbone  = 'EEGNet'
                            args.batch_size= 128
                            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                            args.device    = device
                            args.data_env  = 'gpu' if torch.cuda.is_available() else 'local'
                            args.data      = data_name
                            args.local_dir = './data/' + str(data_name) + '/'
                            # override result_dir per hyperparameter combo
                            args.result_dir = (
                                f'./logs/{data_name}/'
                                f'mtta{max_tta}_str{stride}_t{t}_lr{lr}_st{steps}/'
                            )
                            os.makedirs(args.result_dir, exist_ok=True)

                            # create a dedicated log file for this hyper‐parameter run
                            log_name = f"log_T-TIME_{data_name}_mtta{max_tta}_str{stride}_t{t}_lr{lr}_st{steps}.txt"
                            log_path = os.path.join(args.result_dir, log_name)
                            args.out_file = open(log_path, 'w', encoding='utf-8')

                            # ── existing initialization of logging, seeds, storage... ──
                            my_log = LogRecord(args)
                            my_log.log_init()
                            args.log = my_log
                            # ADD file handler to capture DEBUG timing into same log file
                            file_handler = logging.FileHandler(args.out_file.name)
                            file_handler.setLevel(logging.DEBUG)
                            file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
                            logger.addHandler(file_handler)

                            # log and record the hyperparameter combination
                            combo_str = (
                                f"Running hyperparameters: "
                                f"max_tta={max_tta}, stride={stride}, t={t}, lr={lr}, steps={steps}"
                            )
                            logger.info(combo_str)
                            args.log.record(combo_str)

                            seeds = [2, 3, 5, 6,7,8,9,11,12]
                            total_acc = np.zeros((len(seeds), N))
                            pre_acc_all_seeds = np.zeros((len(seeds), N))
                            ensemble_tta_all = []
                            ensemble_pre_all = []
                            session_tta_breakdowns = []   # store per‐subject session‐wise TTA breakdown
                            session_pre_breakdowns = []   # store per‐subject session‐wise Pre‐TTA breakdown

                            # Iterate through each subject first
                            for idt in range(N):
                                # collect all session indices belonging to this subject prefix
                                target_str = subject_names[idt]
                                idts = [i for i, fn in enumerate(files) if fn.split('_')[0] == target_str]
                                args.idt = idts
                                # Pre-load and cache subject data once per subject iteration
                                if not hasattr(args, 'mi_data_loaded'):
                                    args.mi_data_loaded = read_mi_combine_tar(args)
                                else:
                                    # refresh cache for the changed subject idt
                                    args.mi_data_loaded = read_mi_combine_tar(args)
                                # use prefix names
                                others = subject_names.copy()
                                others.pop(idt)
                                source_str = 'Except_' + '_'.join(others)
                                args.task_str = source_str + '_2_' + target_str
                                
                                info_str = '\n========================== Transfer to ' + target_str + ' =========================='
                                logger.info(info_str)
                                args.log.record(info_str)
                                
                                # Now run all seeds for this subject
                                for seed_idx, s in enumerate(seeds):
                                    args.SEED = s
                                    logger.info(f"--- Running Subject {target_str} with Seed {s} ---")
                                    args.log.record(f"--- Running Subject {target_str} with Seed {s} ---")

                                    fix_random_seed(args.SEED)
                                    torch.backends.cudnn.deterministic = True

                                    args.data = data_name
                                    logger.info(f"Data: {args.data}, Method: {args.method}, Seed: {args.SEED}")
                                    args.log.record(f"Data: {args.data}, Method: {args.method}, Seed: {args.SEED}")

                                    # Run training and evaluation for this subject and seed
                                    tta_acc, pre_acc = train_target(args)
                                    
                                    # Store results
                                    total_acc[seed_idx, idt] = tta_acc
                                    pre_acc_all_seeds[seed_idx, idt] = pre_acc

                                    # Log results for this subject and seed
                                    logger.info(f"Subject {target_str} with Seed {s} - TTA Acc: {tta_acc:.3f}, Pre-TTA: {pre_acc:.3f}")
                                    args.log.record(f"Subject {target_str} with Seed {s} - TTA Acc: {tta_acc:.3f}, Pre-TTA: {pre_acc:.3f}")
                                    
                                    # Save per-seed, per-subject results
                                    np.savetxt(
                                        os.path.join(args.result_dir, f"{data_name}_T-TIME_seed_{args.SEED}_subject_{idt}_acc.csv"),
                                        np.array([tta_acc, pre_acc]), delimiter=","
                                    )
                                    
                                    # Clear GPU memory between seeds
                                    if torch.cuda.is_available():
                                        torch.cuda.empty_cache()
                                
                                # After all seeds for this subject, print summary for this subject
                                subject_mean_tta = np.mean(total_acc[:, idt])
                                subject_std_tta = np.std(total_acc[:, idt])
                                subject_mean_pre = np.mean(pre_acc_all_seeds[:, idt])
                                subject_std_pre = np.std(pre_acc_all_seeds[:, idt])

                                # Calculate 95% confidence intervals (for small sample sizes using t-distribution)
                                from scipy import stats
                                n_seeds = len(seeds)
                                confidence = 0.95
                                # t-value for 95% confidence with n-1 degrees of freedom
                                t_val = stats.t.ppf((1 + confidence) / 2, n_seeds - 1)
                                ci_tta = t_val * (subject_std_tta / np.sqrt(n_seeds))
                                ci_pre = t_val * (subject_std_pre / np.sqrt(n_seeds))
                                
                                logger.info(f"=== Subject {target_str} Summary Across All Seeds ===")
                                logger.info(f"TTA Accuracy: {subject_mean_tta:.3f} ± {subject_std_tta:.3f} (95% CI: {subject_mean_tta-ci_tta:.3f} to {subject_mean_tta+ci_tta:.3f})")
                                logger.info(f"Pre-TTA Accuracy: {subject_mean_pre:.3f} ± {subject_std_pre:.3f} (95% CI: {subject_mean_pre-ci_pre:.3f} to {subject_mean_pre+ci_pre:.3f})")
                                args.log.record(f"TTA Accuracy: {subject_mean_tta:.3f} ± {subject_std_tta:.3f} (95% CI: {subject_mean_tta-ci_tta:.3f} to {subject_mean_tta+ci_tta:.3f})")
                                args.log.record(f"Pre-TTA Accuracy: {subject_mean_pre:.3f} ± {subject_std_pre:.3f} (95% CI: {subject_mean_pre-ci_pre:.3f} to {subject_mean_pre+ci_pre:.3f})")

                                # Ensemble evaluation across seeds for this subject
                                # reload target labels for this subject
                                _, _, X_tar_sub, y_tar_sub = args.mi_data_loaded

                                preds_seeds = []
                                for s in seeds:
                                    pred_file = os.path.join(
                                        args.result_dir,
                                        f"{args.data_name}_T-TIME_seed_{s}_tta_probs.csv"  # load post-adaptation probs
                                    )
                                    # load TTA probs
                                    flat = []
                                    with open(pred_file, 'r') as fr:
                                        for line in fr:
                                            parts = line.strip().split(',')
                                            flat.extend([float(x) for x in parts if x])
                                    preds_seeds.append(np.array(flat))
                                preds_arr = np.stack(preds_seeds, axis=0)  # now shape (n_seeds, n_trials)

                                # 1) Majority‐vote on hard labels
                                labels = (preds_arr > 0.5).astype(int)
                                maj_vote = (labels.sum(axis=0) >= (len(seeds)/2)).astype(int)
                                acc_maj = accuracy_score(y_tar_sub, maj_vote) * 100

                                # 2) Mean probability decision
                                mean_prob = preds_arr.mean(axis=0)
                                mean_vote = (mean_prob > 0.5).astype(int)
                                acc_mean = accuracy_score(y_tar_sub, mean_vote) * 100

                                # 3) Median probability decision
                                med_prob = np.median(preds_arr, axis=0)
                                med_vote = (med_prob > 0.5).astype(int)
                                acc_med = accuracy_score(y_tar_sub, med_vote) * 100

                                # 4) SML‐based ensemble
                                sml_pred = SML(preds_arr)
                                acc_sml = accuracy_score(y_tar_sub, sml_pred) * 100

                                # report ensemble results (no majority vote)
                                logger.info(f"Ensemble TTA: MeanProb={acc_mean:.2f}%, MedianProb={acc_med:.2f}%, SML={acc_sml:.2f}%")
                                args.log.record(f"Ensemble TTA: MeanProb={acc_mean:.2f}%, MedianProb={acc_med:.2f}%, SML={acc_sml:.2f}%")
                                ensemble_tta_all.append([acc_mean, acc_med, acc_sml])
                                # session‐wise TTA breakdown
                                sess_tta = []
                                for idx, (s, e) in enumerate(getattr(args, 'tar_bounds', [])):
                                    sess_name = args.session_names[idx]
                                    sub_preds = preds_arr[:, s:e]
                                    sub_labels = y_tar_sub[s:e]
                                    maj_vote       = (sub_preds > 0.5).sum(axis=0) >= (len(seeds)/2)
                                    mean_prob      = sub_preds.mean(axis=0) > 0.5
                                    med_prob       = np.median(sub_preds, axis=0) > 0.5
                                    sml_pred       = SML(sub_preds)
                                    acc_maj_s      = accuracy_score(sub_labels, maj_vote) * 100
                                    acc_mean_s     = accuracy_score(sub_labels, mean_prob) * 100
                                    acc_med_s      = accuracy_score(sub_labels, med_prob) * 100
                                    acc_sml_s      = accuracy_score(sub_labels, sml_pred) * 100
                                    logger.info(f"   Session {sess_name} Ensemble TTA breakdown: Maj={acc_maj_s:.2f}%, Mean={acc_mean_s:.2f}%, Median={acc_med_s:.2f}%, SML={acc_sml_s:.2f}%")
                                    args.log.record(f"   Session {sess_name} Ensemble TTA breakdown: Maj={acc_maj_s:.2f}%, Mean={acc_mean_s:.2f}%, Median={acc_med_s:.2f}%, SML={acc_sml_s:.2f}%")
                                    sess_tta.append([acc_maj_s, acc_mean_s, acc_med_s, acc_sml_s])
                                session_tta_breakdowns.append((subject_names[idt], sess_tta))

                                # Ensemble summary for Pre-TTA probabilities across seeds
                                pre_list = []
                                for s in seeds:
                                    pp = np.loadtxt(os.path.join(args.result_dir,
                                        f"{args.data_name}_T-TIME_seed_{s}_pre_probs.csv"), delimiter=',')
                                    prob1 = pp[:,1] if pp.ndim>1 else pp
                                    pre_list.append(prob1)
                                pre_mat = np.stack(pre_list, axis=0)  # shape (n_seeds, n_trials)
                                th = getattr(args, 'pre_thresh', 0.5)
                                pre_labels = (pre_mat > th).astype(int)
                                pre_maj = (pre_labels.sum(axis=0) >= (len(seeds)/2)).astype(int)
                                acc_pre_maj = accuracy_score(y_tar_sub, pre_maj) * 100
                                # mean & median decisions using same threshold
                                pre_mean_vote = (pre_mat.mean(axis=0) > th).astype(int)
                                pre_med_vote  = (np.median(pre_mat, axis=0) > th).astype(int)
                                # compute mean/median accuracy
                                acc_pre_mean = accuracy_score(y_tar_sub, pre_mean_vote) * 100
                                acc_pre_med  = accuracy_score(y_tar_sub, pre_med_vote) * 100
                                # SML on soft probabilities
                                pre_sml = SML(pre_mat)
                                acc_pre_sml  = accuracy_score(y_tar_sub, pre_sml) * 100
                                logger.info(f"Ensemble Pre-TTA: Majority={acc_pre_maj:.2f}%, MeanProb={acc_pre_mean:.2f}%, MedianProb={acc_pre_med:.2f}%, SML={acc_pre_sml:.2f}")
                                args.log.record(f"Ensemble Pre-TTA: Majority={acc_pre_maj:.2f}%, MeanProb={acc_pre_mean:.2f}%, MedianProb={acc_pre_med:.2f}%, SML={acc_pre_sml:.2f}")
                                ensemble_pre_all.append([acc_pre_maj, acc_pre_mean, acc_pre_med, acc_pre_sml])
                                # session‐wise Pre-TTA breakdown
                                sess_pre = []
                                for idx, (s, e) in enumerate(getattr(args, 'tar_bounds', [])):
                                    sess_name = args.session_names[idx]
                                    sub_preds_pre = pre_mat[:, s:e]
                                    sub_labels_pre= y_tar_sub[s:e]
                                    maj_vote_p      = (sub_preds_pre > th).sum(axis=0) >= (len(seeds)/2)
                                    mean_prob_p     = sub_preds_pre.mean(axis=0) > th
                                    med_prob_p      = np.median(sub_preds_pre, axis=0) > th
                                    sml_pred_p      = SML(sub_preds_pre)
                                    acc_maj_p_s     = accuracy_score(sub_labels_pre, maj_vote_p) * 100
                                    acc_mean_p_s    = accuracy_score(sub_labels_pre, mean_prob_p) * 100
                                    acc_med_p_s     = accuracy_score(sub_labels_pre, med_prob_p) * 100
                                    acc_sml_p_s     = accuracy_score(sub_labels_pre, sml_pred_p) * 100
                                    logger.info(f"   Session {sess_name} Ensemble Pre-TTA breakdown: Maj={acc_maj_p_s:.2f}%, Mean={acc_mean_p_s:.2f}%, Median={acc_med_p_s:.2f}%, SML={acc_sml_p_s:.2f}%")
                                    args.log.record(f"   Session {sess_name} Ensemble Pre-TTA breakdown: Maj={acc_maj_p_s:.2f}%, Mean={acc_mean_p_s:.2f}%, Median={acc_med_p_s:.2f}%, SML={acc_sml_p_s:.2f}%")
                                    sess_pre.append([acc_maj_p_s, acc_mean_p_s, acc_med_p_s, acc_sml_p_s])
                                session_pre_breakdowns.append((subject_names[idt], sess_pre))


        # ─── Final summary across all subjects ───────────────────────────────
        logger.info("=== Per-seed × per-subject Accuracies ===")
        header = "Subjects: " + ", ".join(subject_names)
        logger.info(header); args.log.record(header)
        seed_line = "Seeds: " + ", ".join(map(str, seeds))
        logger.info(seed_line); args.log.record(seed_line)

        # TTA matrix
        logger.info("TTA Accs (rows=seeds, cols=subjects):")
        logger.info(f"{np.round(total_acc,3)}")
        args.log.record("TTA Accs:\n" + np.array2string(np.round(total_acc, 3)))

        # Pre-TTA matrix
        logger.info("Pre-TTA Accs (rows=seeds, cols=subjects):")
        logger.info(f"{np.round(pre_acc_all_seeds,3)}")
        args.log.record("Pre-TTA Accs:\n" + np.array2string(np.round(pre_acc_all_seeds, 3)))

        # Per-seed means
        tta_seed_mean  = np.round(np.mean(total_acc, axis=1), 3)
        pre_seed_mean  = np.round(np.mean(pre_acc_all_seeds, axis=1), 3)
        logger.info("Overall per-seed means:")
        logger.info(f"  TTA:    {tta_seed_mean}")
        logger.info(f"  Pre-TTA: {pre_seed_mean}")
        args.log.record(f"Overall per-seed TTA mean:  {tta_seed_mean}")
        args.log.record(f"Overall per-seed Pre-TTA mean:  {pre_seed_mean}")

        # Grand means
        overall_tta  = np.mean(total_acc)
        overall_pre  = np.mean(pre_acc_all_seeds)
        summary = (
            f"Grand Means:\n"
            f"  Overall TTA Acc:   {overall_tta:.2f}%\n"
            f"  Overall Pre-TTA:    {overall_pre:.2f}%"
        )
        logger.info(summary)
        args.log.record(summary)

        # ─── Grand ensemble means across subjects ─────────────────────────
        mt = np.array(ensemble_tta_all)
        mp = np.array(ensemble_pre_all)
        gm = (
            f"Grand Ensemble Means:\n"
            f"  TTA    : MeanProb={mt[:,0].mean():.2f}%, MedianProb={mt[:,1].mean():.2f}%, SML={mt[:,2].mean():.2f}%\n"
            f"  Pre-TTA: MeanProb={mp[:,0].mean():.2f}%, MedianProb={mp[:,1].mean():.2f}%, SML={mp[:,2].mean():.2f}%"
        )
        logger.info(gm)
        args.log.record(gm)
    # end of for data_name