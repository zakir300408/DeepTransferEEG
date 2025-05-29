# -*- coding: utf-8 -*-
# @Time    : 2023/07/07
# @Author  : Siyang Li
# @File    : ttime.py
import numpy as np
import argparse 
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from utils.network import backbone_net
from utils.LogRecord import LogRecord
from utils.dataloader import read_mi_combine_tar
from utils.utils import fix_random_seed, cal_acc_comb, data_loader, cal_auc_comb, cal_score_online
from utils.alg_utils import EA_online, EA
from utils.loss import Entropy
from sklearn.metrics import roc_auc_score, accuracy_score
from ttime_ensemble import SML                # << add this import
import torch.linalg as LA
import json
from itertools import product
import sys
import time
from torch.utils.data import DataLoader, TensorDataset
import logging
import os                                 # <— ensure os is imported for fsync
import scipy.stats as stats
from multiprocessing import Pool, cpu_count
import copy

# ── Logger setup ──────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
logger.addHandler(_handler)

# ── module‐level worker ────────────────────────────────────────────────────────
def _run_one_seed_global(params):
    """
    Worker for one seed. Logs clean ASCII and uses UTF-8 file encoding.
    """
    args, idt, files, subject_names, s = params

    # 1) Copy & seed
    args_local = copy.deepcopy(args)
    args_local.SEED = s
    fix_random_seed(s)
    torch.backends.cudnn.deterministic = True

    # 2) Seed‐specific log file (UTF-8 encoded)
    seed_log = f"log_T-TIME_{args_local.data_name}_{args_local.task_str}_seed{s}.txt"
    seed_path = os.path.join(args_local.result_dir, seed_log)
    try:
        args_local.out_file.close()
    except:
        pass
    args_local.out_file = open(seed_path, 'w', encoding='utf-8')

    # Reconfigure the **same** module logger for this file
    seed_logger = logging.getLogger(__name__)
    seed_logger.setLevel(logging.INFO)
    # remove any old FileHandlers
    for h in list(seed_logger.handlers):
        if isinstance(h, logging.FileHandler):
            seed_logger.removeHandler(h)
    # add new FileHandler with utf-8
    fh = logging.FileHandler(seed_path, encoding='utf-8')
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
    seed_logger.addHandler(fh)

    # Re-init your LogRecord wrapper
    args_local.log = LogRecord(args_local)
    args_local.log.log_init()

    # 3) Log start
    seed_logger.info(f"[Seed {s}] START")

    # 4) Run training
    tta_acc, pre_acc = train_target(args_local)

    # 5) Log results (ASCII arrow -> )
    seed_logger.info(f"[Seed {s}] Results -> TTA={tta_acc:.2f}%, Pre={pre_acc:.2f}%")
    args_local.log.record(f"Seed {s}: TTA {tta_acc:.2f}%, Pre {pre_acc:.2f}%")

    # 6) Save per‐seed CSV
    np.savetxt(
        os.path.join(args_local.result_dir,
                     f"{args_local.data_name}_T-TIME_seed_{s}_subject_{idt}_acc.csv"),
        np.array([tta_acc, pre_acc]),
        delimiter=","
    )

    # 7) Free GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return tta_acc, pre_acc


# helper: reset running stats of all BatchNorm layers
def _reset_batchnorm(m):
    if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        m.reset_running_stats()

# Add once: prepare_loader helper to avoid recreating DataLoader inline
def prepare_loader(data, targets, batch_size, num_workers=8, pin_memory=True):
    return DataLoader(
        TensorDataset(data, targets),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

def TTIME(loader, model, args, balanced=True):
    # "T-TIME: Test-Time Information Maximization Ensemble for Plug-and-Play BCIs"
    # IEEE Transactions on Biomedical Engineering
    # Note that the ensemble experiment is separately implemented in ttime_ensemble.py, using recorded test prediction.

    if balanced == False and args.data_name == 'BNCI2014001-4':
        logger.error('ERROR, imbalanced multi-class not implemented')
        sys.exit(0)

    y_true = []
    y_pred = []
    # Remove y_pred_post - no longer needed

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    

    # initialize test reference matrix for Incremental EA
    if args.align:
        R = 0

    if not balanced:
        zk_arrs = np.zeros(2)
        c = 4

    # Initialize data_cum before the loop
    data_cum = None
    # loop through test data stream one by one using native DataLoader iteration
    for i, (inputs, labels) in enumerate(loader):
        #################### Phase 1: target label prediction ####################
        # time data loading & prep
        dl_start = time.time()
        model.eval()
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
            if args.calc_time:
                torch.cuda.synchronize() if args.device.type=='cuda' else None
                ea_ms = (time.time() - ea_start)*1000
                logger.debug(f"[T-TIME] iter {i}: EA_online + transform took {ea_ms:.1f} ms")
        else:
            # no alignment: just slice ON‐DEVICE
            sample_test = data_cum[i].unsqueeze(0)  # now (1,1,chn,time)

        # ensure tensor is float32 on correct device
        sample_test = sample_test.to(device=args.device, dtype=torch.float32)

        # --- Phase 1 inference (no grad) ---------------------------
        with torch.no_grad():
            _, outputs = model(sample_test)

        if args.calc_time:
            dl_ms = (time.time() - dl_start)*1000
            logger.debug(f"[T-TIME] iter {i}: data load & prep took {dl_ms:.1f} ms")

        softmax_out = nn.Softmax(dim=1)(outputs)

        outputs = outputs.float().cpu()
        labels = labels.float().cpu()
        _, predict = torch.max(outputs, 1)

        # Record Phase 1 predictions only
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
                if args.calc_time:
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

            for step in range(args.steps):
                if args.calc_time:
                    opt_start = time.time()
                optimizer.zero_grad()
                if args.calc_time:
                    fwd_start = time.time()
                _, outputs = model(batch_test)
                if args.calc_time:
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

                loss.backward()
                optimizer.step()
                if args.calc_time:
                    torch.cuda.synchronize() if args.device.type=='cuda' else None
                    opt_ms = (time.time() - opt_start) * 1000
                    logger.debug(f"[T-TIME] iter {i}, step {step}: backward+opt {opt_ms:.1f} ms")

            if args.calc_time:
                tta_time = time.time()
                logger.info(f"sample {i}, post-inference model update finished in ms: {np.round((tta_time - dl_start)*1000,3)}")

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

        model.eval()  # restore eval mode for next iteration
        # Remove Phase 3 - no post-adaptation inference needed

    # --- after loop: compute metrics on Phase 1 preds ---
    if balanced:
        if args.class_num == 2:
            y_scores = np.array(y_pred).reshape(-1, args.class_num)[:, 1]
            thresh = getattr(args, 'pre_thresh', 0.5)
            preds = (y_scores > thresh).astype(int)
            score = accuracy_score(y_true, preds)
            y_pred_final = y_scores
        else:
            arr = torch.from_numpy(np.array(y_pred)).to(torch.float32)
            _, predict = torch.max(arr.reshape(-1, args.class_num), 1)
            preds = predict.cpu().numpy().astype(int)
            score = accuracy_score(y_true, preds)
            y_pred_final = preds
    else:
        arr = torch.from_numpy(np.array(y_pred)).to(torch.float32)
        y_scores = arr.reshape(-1, args.class_num)[:, 1].cpu().numpy()
        score = roc_auc_score(y_true, y_scores)
        y_pred_final = y_scores

    if args.align:
        return score * 100, y_pred_final, sqrtRefEA
    else:
        return score * 100, y_pred_final, None


def train_target(args):
    if not args.align:
        extra_string = '_noEA'
    else:
        extra_string = ''
    # at top of function they already compute idt_str for ckpt names
    if isinstance(args.idt, (list, tuple)):
        idt_str = '_'.join(map(str, args.idt))
    else:
        idt_str = str(args.idt)
    # make sure save directory exists
    os.makedirs(os.path.join('.', 'runs', args.data_name), exist_ok=True)
    os.makedirs(args.result_dir, exist_ok=True)

    # always load fresh source/target data for current subject
    X_src, y_src, X_tar, y_tar = read_mi_combine_tar(args)

    # build data loaders for current subject without caching
    dset_loaders = data_loader(X_src, y_src, X_tar, y_tar, args)

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
        # verify filename encodes current target sessions
        expected = f"_S{idt_str}_"
        if expected not in best_ckpt:
            logger.error(f"Checkpoint filename '{best_ckpt}' does not match target sessions {args.idt}")
            sys.exit(1)
        logger.debug(f"[DEBUG] Loading pretrained source-model checkpoint: {best_ckpt}")
        if not os.path.isfile(best_ckpt):
            logger.warning(f"[WARNING] Pretrained checkpoint not found: {best_ckpt}")
        base_network.load_state_dict(torch.load(best_ckpt, map_location=args.device))
        # WARNING: confirm loaded model excludes any target-session data
        logger.info   (f"[CONFIRM] Loaded source model '{best_ckpt}' excludes target sessions {args.idt}")
        base_network.eval()

        # unify metric: use online streaming scoring on the same split
        metric = "Acc" if args.balanced else "AUC"
        pre_acc = cal_score_online(dset_loaders["Target-Online"], base_network, args=args)
        logger.info(f"Task: {args.task_str}, Pre-TTA IEA {metric} = {pre_acc:.2f}%")
        args.log.record(f"Task: {args.task_str}, Pre-TTA IEA {metric} = {pre_acc:.2f}%")

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
        # initialize EA reference for pre-TTA
        if args.align:
            # init R to mean source-domain covariance (with small ridge)
            covs = np.array([np.cov(x) for x in X_src])
            R = covs.mean(axis=0) + np.eye(covs.shape[1]) * 1e-6
        else:
            R = None

        pre_y_pred = []
        with torch.no_grad():
            for i, (x, _) in enumerate(loader_pre):
                x = x.to(args.device).float()
                if 'EEGNet' in args.backbone:
                    x = x.unsqueeze(3).permute(0,3,1,2)
                if args.align:
                    # update & apply EA whitening per sample
                    sample = x.squeeze(0).squeeze(0)               # (chn, time)
                    sample_np = sample.cpu().numpy()
                    # -- whiten with old R first (no snooping) --
                    R_reg = R + np.eye(R.shape[0]) * 1e-6
                    R_t   = torch.from_numpy(R_reg).to(device=args.device, dtype=sample.dtype)
                    ev, evec     = LA.eigh(R_t)
                    sqrtRefEA    = evec @ torch.diag(ev.pow(-0.5)) @ evec.T
                    sample       = sqrtRefEA @ sample
                    # -- now update R to include this new sample --
                    R = EA_online(sample_np, R, i)
                    R += np.eye(R.shape[0]) * 1e-6
                    x = sample.reshape(1,1,args.chn,args.time_sample_num)
                _, logits = src_model(x)
                soft = nn.Softmax(dim=1)(logits)
                pre_y_pred.append(soft.cpu().numpy())
        pre_y_pred = np.concatenate(pre_y_pred, axis=0)
        pos_probs = pre_y_pred[:, 1]
        np.savetxt(
            os.path.join(args.result_dir,
                         f"{args.data_name}_T-TIME_seed_{args.SEED}_subj{idt_str}_pre_probs.csv"),
            pos_probs, delimiter=",")
        # use fixed threshold for Pre-TTA
        args.pre_thresh = 0.5
        # POST-TTA streaming adaptation
        adapted_model = copy.deepcopy(base_network).to(args.device)
        adapted_model.apply(_reset_batchnorm)           # <<< reset BN stats
        tta_score, tta_y_pred, _ = TTIME(loader_pre, adapted_model, args=args, balanced=args.balanced)
        np.savetxt(
            os.path.join(args.result_dir,
                         f"{args.data_name}_T-TIME_seed_{args.SEED}_subj{idt_str}_tta_probs.csv"),
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
        # WARNING: confirm loaded model excludes any target-session data
        logger.info   (f"[CONFIRM] Loaded best model '{best_ckpt}' excludes target sessions {args.idt}")

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
        # initialize EA reference for pre-TTA
        if args.align:
            covs = np.array([np.cov(x) for x in X_src])
            R = covs.mean(axis=0) + np.eye(covs.shape[1]) * 1e-6
        else:
            R = None

        pre_y_pred = []
        with torch.no_grad():
            for i, (x, _) in enumerate(loader_pre):
                x = x.to(args.device).float()
                if 'EEGNet' in args.backbone:
                    x = x.unsqueeze(3).permute(0,3,1,2)
                if args.align:
                    # update & apply EA whitening per sample
                    sample = x.squeeze(0).squeeze(0)               # (chn, time)
                    sample_np = sample.cpu().numpy()
                    # -- whiten with old R first (no snooping) --
                    R_reg = R + np.eye(R.shape[0]) * 1e-6
                    R_t   = torch.from_numpy(R_reg).to(device=args.device, dtype=sample.dtype)
                    ev, evec     = LA.eigh(R_t)
                    sqrtRefEA    = evec @ torch.diag(ev.pow(-0.5)) @ evec.T
                    sample       = sqrtRefEA @ sample
                    # -- now update R to include this new sample --
                    R = EA_online(sample_np, R, i)
                    R += np.eye(R.shape[0]) * 1e-6
                    x = sample.reshape(1,1,args.chn,args.time_sample_num)
                _, logits = src_model(x)
                soft = nn.Softmax(dim=1)(logits)
                pre_y_pred.append(soft.cpu().numpy())
        pre_y_pred = np.concatenate(pre_y_pred, axis=0)
        # current: saves both class‐0 and class‐1 probs shape (n_samples,2)
        # np.savetxt(os.path.join(args.result_dir,
        #            f"{args.data_name}_T-TIME_seed_{args.SEED}_pre_probs.csv"),
        #           pre_y_pred, delimiter=",")
        # replace with only the positive‐class probs:
        pos_probs = pre_y_pred[:, 1]
        np.savetxt(
            os.path.join(args.result_dir,
                         f"{args.data_name}_T-TIME_seed_{args.SEED}_subj{idt_str}_pre_probs.csv"),
            pos_probs, delimiter=",")
        # use fixed threshold for Pre-TTA
        args.pre_thresh = 0.5
        # POST-TTA streaming adaptation
        adapted_model = copy.deepcopy(base_network).to(args.device)
        adapted_model.apply(_reset_batchnorm)           # <<< reset BN stats
        tta_score, tta_y_pred, _ = TTIME(loader_pre, adapted_model, args=args, balanced=args.balanced)
        np.savetxt(
            os.path.join(args.result_dir,
                         f"{args.data_name}_T-TIME_seed_{args.SEED}_subj{idt_str}_tta_probs.csv"),
            tta_y_pred, delimiter=",")
        best_ckpt_adapted = f'./runs/{args.data_name}/{args.backbone}_S{idt_str}_seed{args.SEED}{extra_string}_adapted.ckpt'
        torch.save(adapted_model.state_dict(), best_ckpt_adapted)
        return tta_score, pre_acc



def load_metadata(data_name):
    meta_path = f'./data/{data_name}/meta.csv'
    df_meta = pd.read_csv(meta_path)
    files = df_meta['file'].tolist()
    subject_names = sorted({f.split('_')[0] for f in files})
    return df_meta, files, subject_names


def get_dataset_params(data_name, subject_names, df_meta):
    if data_name == 'BNCI2014001':
        return 'MI', 9, 22, 2, 1001, 250, 144, 248
    elif data_name == 'BNCI2014002':
        return 'MI', 14, 15, 2, 2561, 512, 100, 640
    elif data_name == 'BNCI2015001':
        return 'MI', 12, 13, 2, 2561, 512, 200, 640
    elif data_name == 'CustomEpoch':
        paradigm = 'MI'
        N = len(subject_names)
        chn, class_num, time_sample_num, sample_rate = 31, 2, 1515, 200
        feature_deep_dim = 1504
        trial_num = int(df_meta['n_trials'].sum())
        return paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim
    else:
        raise ValueError(f"Unknown data_name {data_name}")


def build_hyperparam_grid():
    return {
        't':       [1.5, 1.7, 1.9, 2.0],
        'lr':      [0.0001, 0.0005],
        'steps':   [1, 3],
    }


def build_base_args(data_name, paradigm, N, chn, class_num,
                    time_sample_num, sample_rate, trial_num, feature_deep_dim):
    args = argparse.Namespace()
    args.data_name        = data_name
    args.feature_deep_dim = feature_deep_dim
    args.paradigm         = paradigm
    args.N                = N
    args.chn              = chn
    args.class_num        = class_num
    args.time_sample_num  = time_sample_num
    args.sample_rate      = sample_rate
    args.trial_num        = trial_num
    args.print_trial_details = False
    args.method           = 'T-TIME'
    args.backbone         = 'EEGNet'
    args.batch_size       = 128
    args.align            = True
    args.use_pretrained_model = True
    args.balanced         = True
    args.calc_time        = False
    args.max_parallel_seeds = 4
    # Fixed hyperparameters
    args.max_tta          = 8
    args.stride           = 1
    args.device           = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.data_env         = 'gpu' if torch.cuda.is_available() else 'local'
    return args



def setup_run(args, data_name, hp):
    # hyperparams from grid
    args.t       = hp['t']
    args.lr      = hp['lr']
    args.steps   = hp['steps']
    # prepare a concise task identifier used for log filenames
    args.task_str = f"mtta{args.max_tta}_str{args.stride}_t{args.t}_lr{args.lr}_st{args.steps}"
    # epochs
    args.max_epoch = 0 if args.use_pretrained_model else 30

    # paths
    args.data      = data_name
    args.local_dir = f'./data/{data_name}/'
    args.result_dir = (
        f'./logs/{data_name}/'
        f'mtta{args.max_tta}_str{args.stride}_t{args.t}_lr{args.lr}_st{args.steps}/'
    )
    os.makedirs(args.result_dir, exist_ok=True)

    # open a UTF-8–encoded run-level log
    log_name = (
        f"log_T-TIME_{data_name}_"
        f"mtta{args.max_tta}_str{args.stride}_t{args.t}_"
        f"lr{args.lr}_st{args.steps}.txt"
    )
    log_path = os.path.join(args.result_dir, log_name)
    args.out_file = open(log_path, 'w', encoding='utf-8')

    # hook up LogRecord (if you’re using it)
    args.log = LogRecord(args)
    args.log.log_init()

    # reconfigure the module‐level logger to write into that file (UTF-8)
    for h in list(logger.handlers):
        if isinstance(h, logging.FileHandler):
            logger.removeHandler(h)
    fh = logging.FileHandler(log_path, encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
    logger.addHandler(fh)

    combo_str = (
        f"Running hyperparameters: "
        f"max_tta={args.max_tta}, stride={args.stride}, t={args.t}, "
        f"lr={args.lr}, steps={args.steps}"
    )
    logger.info(combo_str)
    args.log.record(combo_str)



def run_subject(args, idt, files, subject_names, seeds):
    """
    1) Runs all seeds (in batches of args.max_parallel_seeds)
    2) Logs per-seed and overall summaries
    3) Logs *per-session* ensemble breakdown
    4) Returns exactly what build_and_save_results needs
    """
    # — identify target & name task —
    target = subject_names[idt]
    args.idt = [i for i,f in enumerate(files) if f.split('_')[0] == target]

    # build per-session bounds & names (so your breakdown loop will run)
    df_meta = pd.read_csv(os.path.join(args.local_dir, 'meta.csv'))
    counts  = df_meta['n_trials'].values
    sel = counts[args.idt]
    starts = np.concatenate(([0], np.cumsum(sel)[:-1]))
    ends   = np.cumsum(sel)
    args.tar_bounds    = [(s,e) for s,e in zip(starts, ends)]
    args.session_names = [df_meta['file'].iloc[i] for i in args.idt]
    
    # derive the same subject‐ID string
    idt_str = '_'.join(map(str, args.idt))
    
    # — header —
    logger.info(f"\n=== Transfer to {target} ===")
    args.log.record(f"Transfer to {target}")

    # — prepare parallel seed params —
    sane = {k: v for k, v in vars(args).items() if k not in ('log','out_file','mi_data_loaded')}
    sanitized = argparse.Namespace(**sane)
    params = [(sanitized, idt, files, subject_names, s) for s in seeds]

    # — dispatch in batches of K seeds —
    K = getattr(args, 'max_parallel_seeds', 4)
    results = []
    
    # Process all batches sequentially
    for i in range(0, len(seeds), K):
        batch_seeds = seeds[i:i+K]
        batch_params = [(sanitized, idt, files, subject_names, s) for s in batch_seeds]
        
        logger.info(f"Processing seeds batch {i//K + 1}/{(len(seeds)-1)//K + 1}: {batch_seeds}")
        
        with Pool(processes=len(batch_params)) as pool:
            batch_results = pool.map(_run_one_seed_global, batch_params)
            results.extend(batch_results)
        
        # Force cleanup between batches
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info(f"Completed batch {i//K + 1}, processed {len(results)}/{len(seeds)} seeds so far")

    logger.info(f"Finished processing all {len(results)} seeds for subject {target}")
    
    # — per-seed summary —
    total_tta = np.array([r[0] for r in results])
    total_pre = np.array([r[1] for r in results])
    for s, tta, pre in zip(seeds, total_tta, total_pre):
        logger.info(f"[Summary] Seed {s} -> TTA={tta:.2f}%, Pre={pre:.2f}%")
        args.log.record(f"Seed {s} summary: TTA {tta:.2f}%, Pre {pre:.2f}%")

    # — subject-level 95% CI —
    n     = len(seeds)
    t_val = stats.t.ppf(0.975, n-1)
    mu_t, sd_t = total_tta.mean(), total_tta.std()
    ci_t = t_val * sd_t / np.sqrt(n)
    mu_p, sd_p = total_pre.mean(), total_pre.std()
    ci_p = t_val * sd_p / np.sqrt(n)
    summary = (
        f"{target}: TTA {mu_t:.3f}±{sd_t:.3f} "
        f"(95% CI [{mu_t-ci_t:.3f}, {mu_t+ci_t:.3f}]); "
        f"Pre {mu_p:.3f}±{sd_p:.3f} "
        f"(95% CI [{mu_p-ci_p:.3f}, {mu_p+ci_p:.3f}])"
    )
    logger.info(summary)
    args.log.record(summary)

    # — reload data & stack per-seed prob files —
    _, _, X_tar, y_tar = read_mi_combine_tar(args)
    preds_pre = np.vstack([
        np.loadtxt(os.path.join(
            args.result_dir,
            f"{args.data_name}_T-TIME_seed_{s}_subj{idt_str}_pre_probs.csv"
        ), delimiter=",")
        for s in seeds
    ])
    preds_tta = np.vstack([
        np.loadtxt(os.path.join(
            args.result_dir,
            f"{args.data_name}_T-TIME_seed_{s}_subj{idt_str}_tta_probs.csv"
        ), delimiter=",")
        for s in seeds
    ])

    # — overall ensemble accuracies —
    mean_vote_tta = (preds_tta.mean(0) > 0.5).astype(int)
    med_vote_tta  = (np.median(preds_tta, 0) > 0.5).astype(int)
    sml_vote_tta  = SML(preds_tta)
    acc_tta = [
        accuracy_score(y_tar, mean_vote_tta) * 100,
        accuracy_score(y_tar, med_vote_tta)  * 100,
        accuracy_score(y_tar, sml_vote_tta)  * 100
    ]

    mean_vote_pre = (preds_pre.mean(0) > 0.5).astype(int)
    med_vote_pre  = (np.median(preds_pre, 0) > 0.5).astype(int)
    sml_vote_pre  = SML(preds_pre)
    acc_pre_ens = [
        accuracy_score(y_tar, mean_vote_pre) * 100,
        accuracy_score(y_tar, med_vote_pre)  * 100,
        accuracy_score(y_tar, sml_vote_pre)  * 100
    ]

    # — log overall ensembles —
    logger.info(
        f"Ensemble TTA:    MeanProb={acc_tta[0]:.2f}%, "
        f"MedianProb={acc_tta[1]:.2f}%, SML={acc_tta[2]:.2f}%"
    )
    args.log.record(
        f"Ensemble TTA:    MeanProb={acc_tta[0]:.2f}%, "
        f"MedianProb={acc_tta[1]:.2f}%, SML={acc_tta[2]:.2f}%"
    )
    logger.info(
        f"Ensemble Pre-TTA: MeanProb={acc_pre_ens[0]:.2f}%, "
        f"MedianProb={acc_pre_ens[1]:.2f}%, SML={acc_pre_ens[2]:.2f}%"
    )
    args.log.record(
        f"Ensemble Pre-TTA: MeanProb={acc_pre_ens[0]:.2f}%, "
        f"MedianProb={acc_pre_ens[1]:.2f}%, SML={acc_pre_ens[2]:.2f}%"
    )

    # — **now** log per-session breakdowns —
    sess_tta_break = []
    sess_pre_break = []
    for idx, (s_idx, e_idx) in enumerate(getattr(args, 'tar_bounds', [])):
        sess = args.session_names[idx]
        subp_t = preds_tta[:, s_idx:e_idx]; suby = y_tar[s_idx:e_idx]
        votes_t = [
            accuracy_score(suby, (subp_t.mean(0)>0.5).astype(int))*100,
            accuracy_score(suby, (np.median(subp_t,0)>0.5).astype(int))*100,
            accuracy_score(suby, SML(subp_t))*100
        ]
        sess_tta_break.append([sess] + votes_t)
        logger.info(f"TTA session {sess}: Mean={votes_t[0]:.2f}%, "
                    f"Median={votes_t[1]:.2f}%, SML={votes_t[2]:.2f}%")
        args.log.record(f"TTA session {sess}: Mean={votes_t[0]:.2f}%, "
                        f"Median={votes_t[1]:.2f}%, SML={votes_t[2]:.2f}%")

        subp_p = preds_pre[:, s_idx:e_idx]
        votes_p = [
            accuracy_score(suby, (subp_p.mean(0)>0.5).astype(int))*100,
            accuracy_score(suby, (np.median(subp_p,0)>0.5).astype(int))*100,
            accuracy_score(suby, SML(subp_p))*100
        ]
        sess_pre_break.append([sess] + votes_p)
        logger.info(f"Pre-TTA session {sess}: Mean={votes_p[0]:.2f}%, "
                    f"Median={votes_p[1]:.2f}%, SML={votes_p[2]:.2f}%")
        args.log.record(f"Pre-TTA session {sess}: Mean={votes_p[0]:.2f}%, "
                        f"Median={votes_p[1]:.2f}%, SML={votes_p[2]:.2f}%")

    return total_tta, total_pre, acc_tta, acc_pre_ens, sess_tta_break, sess_pre_break



def build_and_save_results(args,
                           total_tta, total_pre,
                           ens_tta_list, ens_pre_list,
                           session_tta_breaks, session_pre_breaks):
    overall = {
        'tta': {
            'mean':   float(total_tta.mean()),
            'median': float(np.median(total_tta)),
            'std':    float(total_tta.std())
        },
        'pre_tta': {
            'mean':   float(total_pre.mean()),
            'median': float(np.median(total_pre)),
            'std':    float(total_pre.std())
        }
    }
    ens_tta_arr = np.array(ens_tta_list)
    ens_pre_arr = np.array(ens_pre_list)
    ensemble = {
        'tta': {
            'mean_prob':   float(ens_tta_arr[:, 0].mean()),
            'median_prob': float(ens_tta_arr[:, 1].mean()),
            'sml':         float(ens_tta_arr[:, 2].mean())
        },
        'pre_tta': {
            'mean_prob':   float(ens_pre_arr[:, 0].mean()),
            'median_prob': float(ens_pre_arr[:, 1].mean()),
            'sml':         float(ens_pre_arr[:, 2].mean())
        }
    }

    result = {
        'hyperparams': vars(args).copy(),
        'per_seed_per_subject': {
            'tta':     total_tta.tolist(),
            'pre_tta': total_pre.tolist()
        },
        'overall': overall,
        'ensemble': ensemble,
        'session_tta_breakdowns':     session_tta_breaks,
        'session_pre_tta_breakdowns': session_pre_breaks
    }

    # drop un-serializable items
    for k in ['mi_data_loaded', 'log', 'out_file']:
        result['hyperparams'].pop(k, None)

    path = os.path.join(
        args.result_dir,
        f"results_mtta{args.max_tta}_str{args.stride}_t{args.t}"
        f"_lr{args.lr}_st{args.steps}.json"
    )
    with open(path, 'w', encoding='utf-8') as jf:
        json.dump(result, jf, default=lambda o: str(o), indent=2)
        jf.flush(); os.fsync(jf.fileno())

    logger.info(f"Saved structured results to {path}")
    args.log.record(f"Saved structured results to {path}")




def main():
    data_names = ['CustomEpoch']
    for data_name in data_names:
        df_meta, files, subject_names = load_metadata(data_name)
        paradigm, N, chn, class_num, tsn, sr, tn, fdd = get_dataset_params(
            data_name, subject_names, df_meta
        )
        grid = build_hyperparam_grid()
        base_args = build_base_args(
            data_name, paradigm, N, chn, class_num, tsn, sr, tn, fdd
        )
        seeds = [2,3,5,6,7,8,9,12]

        for hp_values in product(*grid.values()):
            hp = dict(zip(grid.keys(), hp_values))
            args = argparse.Namespace(**vars(base_args))
            setup_run(args, data_name, hp)

            all_tta_acc    = []
            all_pre_acc    = []
            all_ens_tta    = []
            all_ens_pre    = []
            all_sess_tta   = []
            all_sess_pre   = []

            for idt in range(N):
                tta, pre, etta, epre, sb_tta, sb_pre = run_subject(
                    args, idt, files, subject_names, seeds
                )
                all_tta_acc.append(tta)
                all_pre_acc.append(pre)
                all_ens_tta.append(etta)
                all_ens_pre.append(epre)

                # prepend subject name to each session row
                all_sess_tta += [[subject_names[idt]] + row for row in sb_tta]
                all_sess_pre += [[subject_names[idt]] + row for row in sb_pre]

            # flatten arrays as before …
            tta_arr      = np.hstack(all_tta_acc)
            pre_arr      = np.hstack(all_pre_acc)
            ens_tta_flat = [x for trio in all_ens_tta for x in [trio]]
            ens_pre_flat = [x for trio in all_ens_pre for x in [trio]]

            build_and_save_results(
                args,
                tta_arr,
                pre_arr,
                ens_tta_flat,
                ens_pre_flat,
                all_sess_tta,
                all_sess_pre    # ← pass new
            )


if __name__ == '__main__':
    main()