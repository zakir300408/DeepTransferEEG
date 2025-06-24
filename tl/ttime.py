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
    y_pred_post = []                # add container for post-adaptation preds

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

        # --- Phase 3 (post-adaptation inference) ---
        with torch.no_grad():
            _, outputs_post = model(sample_test)
        soft_post = nn.Softmax(dim=1)(outputs_post)
        y_pred_post.append(soft_post.cpu().numpy())

    # --- after loop: compute metrics on post-adaptation preds ---
    if balanced:
        if args.class_num == 2:
            y_scores = np.array(y_pred_post).reshape(-1, args.class_num)[:, 1]
            thresh = getattr(args, 'pre_thresh', 0.5)
            preds = (y_scores > thresh).astype(int)
            score = accuracy_score(y_true, preds)
            y_pred_final = y_scores
        else:
            arr = torch.from_numpy(np.array(y_pred_post)).to(torch.float32)
            _, predict = torch.max(arr.reshape(-1, args.class_num), 1)
            preds = predict.cpu().numpy().astype(int)
            score = accuracy_score(y_true, preds)
            y_pred_final = preds
    else:
        arr = torch.from_numpy(np.array(y_pred_post)).to(torch.float32)
        y_scores = arr.reshape(-1, args.class_num)[:, 1].cpu().numpy()
        score = roc_auc_score(y_true, y_scores)
        y_pred_final = y_scores

    if args.align:
        return score * 100, y_pred_final, sqrtRefEA
    else:
        return score * 100, y_pred_final, None


def setup_experiment_dirs(args):
    """Prepare run directories, compute checkpoint suffix and idt string."""
    extra = '_noEA' if not args.align else ''
    if isinstance(args.idt, (list, tuple)):
        idt_str = '_'.join(map(str, args.idt))
    else:
        idt_str = str(args.idt)

    base = os.path.join('.', 'runs', args.data_name)
    os.makedirs(base, exist_ok=True)
    os.makedirs(args.result_dir, exist_ok=True)

    return extra, idt_str

def load_data_and_build_loaders(args):
    """Load source/target data and build PyTorch DataLoaders."""
    X_src, y_src, X_tar, y_tar = read_mi_combine_tar(args)
    loaders = data_loader(X_src, y_src, X_tar, y_tar, args)
    return X_src, y_src, X_tar, y_tar, loaders

def configure_custom_epoch(args):
    """If using CustomEpoch, set up per-session bounds & names on args."""
    if args.data != 'CustomEpoch':
        return
    df = pd.read_csv('./data/CustomEpoch/meta.csv')
    counts = df['n_trials'].values
    idts = args.idt if isinstance(args.idt, (list, tuple)) else [args.idt]
    sel = counts[idts]
    starts = np.concatenate(([0], np.cumsum(sel)[:-1]))
    ends   = np.cumsum(sel)
    args.tar_bounds     = list(zip(starts, ends))
    args.session_names  = [df['file'].iloc[i] for i in idts]

def initialize_network(args):
    """Build and move backbone+classifier to the target device."""
    netF, netC = backbone_net(args, return_type='xy')
    netF, netC = netF.to(args.device), netC.to(args.device)
    return nn.Sequential(netF, netC).to(args.device)

def zero_epoch_flow(args, base_net, X_src, y_src, X_tar, y_tar, idt_str, extra):
    """Handle the args.max_epoch == 0 case: load checkpoint, do Pre-TTA (EA) & TTA."""
    # Load source checkpoint
    ckpt = f'./runs/{args.data_name}/{args.backbone}_S{idt_str}_seed{args.SEED}{extra}_best.ckpt'
    if not os.path.isfile(ckpt):
        logger.warning(f"Pretrained checkpoint not found: {ckpt}")
    base_net.load_state_dict(torch.load(ckpt, map_location=args.device))
    base_net.eval()
    logger.info(f"[CONFIRM] Loaded source model excludes targets {args.idt}")
    
    # helper to run inference & collect softmax probs, with optional EA alignment
    def infer_probs(model, loader, align):
        R = None
        if align:
            # init R from source covariance
            covs = np.array([np.cov(x) for x in X_src])
            R = covs.mean(axis=0) + np.eye(covs.shape[1]) * 1e-6
        probs = []
        with torch.no_grad():
            for i, (x, _) in enumerate(loader):
                x = x.to(args.device).float()
                if 'EEGNet' in args.backbone:
                    x = x.unsqueeze(3).permute(0,3,1,2)
                if align:
                    sample_np = x.squeeze().cpu().numpy()
                    R_reg     = R + np.eye(R.shape[0]) * 1e-6
                    ev, evec  = LA.eigh(torch.from_numpy(R_reg).to(device=args.device, dtype=x.dtype))
                    sqrtR     = evec @ torch.diag(ev.pow(-0.5)) @ evec.T
                    sample    = sqrtR @ x.squeeze(0).squeeze(0)
                    # update R incrementally
                    R         = EA_online(sample_np, R, i) + np.eye(R.shape[0]) * 1e-6
                    x         = sample.reshape(1,1,args.chn,args.time_sample_num)
                _, logits = model(x)
                probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        return np.concatenate(probs, axis=0)

    # prepare a single-trial loader for target data
    loader_pre = DataLoader(
        TensorDataset(torch.from_numpy(X_tar).float(),
                      torch.from_numpy(y_tar).long()),
        batch_size=1, shuffle=False
    )

    # --- EA Pre-TTA ---
    adapt = copy.deepcopy(base_net).to(args.device)
    # adapt.apply(_reset_batchnorm)    # reset all BN running stats
    adapt.eval()

    # compute and log Pre-TTA accuracy (with EA)
    pre_acc = cal_score_online(loader_pre, adapt, args=args)
    metric = "Acc" if args.balanced else "AUC"
    logger.info(f"Pre-TTA EA {metric} = {pre_acc:.2f}%")
    args.log.record(f"Pre-TTA EA {metric} = {pre_acc:.2f}%")

    # now **with** EA alignment
    bn_reset_probs = infer_probs(adapt, loader_pre, align=True)[:, 1]
    np.savetxt(
        os.path.join(args.result_dir,
                     f"{args.data_name}_T-TIME_seed_{args.SEED}_subj{idt_str}_pre_probs.csv"),
        bn_reset_probs,
        delimiter=","
    )

    # --- Full T-TTA on the BN-reset+EA model ---
    tta_score, tta_probs, _ = TTIME(loader_pre, adapt, args=args, balanced=args.balanced)
    np.savetxt(
        os.path.join(args.result_dir,
                     f"{args.data_name}_T-TIME_seed_{args.SEED}_subj{idt_str}_tta_probs.csv"),
        tta_probs,
        delimiter=","
    )

    # save the adapted model
    torch.save(
        adapt.state_dict(),
        f'./runs/{args.data_name}/{args.backbone}_S{idt_str}_seed{args.SEED}{extra}_adapted.ckpt'
    )

    return tta_score, pre_acc



def train_and_evaluate_flow(args, base_net, X_src, y_src, X_tar, y_tar, dset_loaders, idt_str, extra):
    """Handle the args.max_epoch > 0 case: train, save best, then Pre-TTA & TTA."""
    criterion = nn.CrossEntropyLoss()
    opt_f = optim.Adam(base_net[0].parameters(), lr=args.lr)
    opt_c = optim.Adam(base_net[1].parameters(), lr=args.lr)

    max_iter = args.max_epoch * len(dset_loaders["source"])
    interval  = max_iter // args.max_epoch
    args.max_iter = max_iter

    best_acc = 0.0
    ckpt_best = f'./runs/{args.data_name}/{args.backbone}_S{idt_str}_seed{args.SEED}{extra}_best.ckpt'

    base_net.train()
    it_src = iter(dset_loaders["source"])
    for it in range(1, max_iter+1):
        try:
            xs, ys = next(it_src)
        except StopIteration:
            it_src = iter(dset_loaders["source"])
            xs, ys = next(it_src)
        if xs.size(0) == 1:
            continue

        xs, ys = xs.to(args.device), ys.to(args.device)
        feats, outs = base_net(xs)
        loss = criterion(outs, ys)

        opt_f.zero_grad(); opt_c.zero_grad()
        loss.backward()
        opt_f.step(); opt_c.step()

        if it % interval == 0 or it == max_iter:
            epoch = min(it//interval, args.max_epoch)
            logger.info(f"Epoch {epoch}/{args.max_epoch} done")
            base_net.eval()
            if args.balanced:
                acc, _ = cal_acc_comb(dset_loaders["Target"], base_net, args=args)
            else:
                acc, _ = cal_auc_comb(dset_loaders["Target-Imbalanced"], base_net, args=args)
            logger.info(f"Val Acc: {acc:.2f}%")
            if acc > best_acc:
                best_acc = acc
                torch.save(base_net.state_dict(), ckpt_best)
            base_net.train()

    # load best, then create fresh copy
    base_net.load_state_dict(torch.load(ckpt_best, map_location=args.device))
    logger.info(f"[CONFIRM] Loaded best model excludes targets {args.idt}")
    
    fresh_net = copy.deepcopy(base_net).to(args.device)
    # fresh_net.apply(_reset_batchnorm)  # BN reset skipped
    
    return zero_epoch_flow(args, fresh_net, X_src, y_src, X_tar, y_tar, idt_str, extra)

def train_target(args):
    """Top‐level entry for each seed: either zero-epoch or full train then zero_epoch_flow."""
    extra, idt_str = setup_experiment_dirs(args)
    X_src, y_src, X_tar, y_tar, loaders = load_data_and_build_loaders(args)
    configure_custom_epoch(args)
    base_net = initialize_network(args)

    if args.max_epoch == 0:
        # only Pre-TTA & T-TTA on the source model
        return zero_epoch_flow(args, base_net, X_src, y_src, X_tar, y_tar, idt_str, extra)
    else:
        # train for args.max_epoch then do zero_epoch_flow
        return train_and_evaluate_flow(
            args, base_net, X_src, y_src, X_tar, y_tar,
            loaders, idt_str, extra
        )


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
        chn, class_num, time_sample_num, sample_rate = 27, 2, 725, 100
        feature_deep_dim = 704
        trial_num = int(df_meta['n_trials'].sum())
        return paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim
    else:
        raise ValueError(f"Unknown data_name {data_name}")


def build_hyperparam_grid():
    return {
        't':       [1.5],
        'lr':      [0.0005],
        'steps':   [1],
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
    args.use_pretrained_model = False
    args.balanced         = True
    args.calc_time        = False
    args.max_parallel_seeds = 1
    # Fixed hyperparameters
    args.max_tta          = 15
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
    args.max_epoch = 0 if args.use_pretrained_model else 50

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
    # build underscored idt_str
    idt_str = '_'.join(map(str, args.idt))

    # build per-session bounds & names
    df_meta = pd.read_csv(os.path.join(args.local_dir, 'meta.csv'))
    counts  = df_meta['n_trials'].values
    sel     = counts[args.idt]
    starts  = np.concatenate(([0], np.cumsum(sel)[:-1]))
    ends    = np.cumsum(sel)
    args.tar_bounds    = [(s,e) for s,e in zip(starts, ends)]
    args.session_names = [df_meta['file'].iloc[i] for i in args.idt]
    
    logger.info(f"\n=== Transfer to {target} ===")
    args.log.record(f"Transfer to {target}")

    # dispatch seeds in parallel batches
    sane = {k:v for k,v in vars(args).items() if k not in ('log','out_file','mi_data_loaded')}
    sanitized = argparse.Namespace(**sane)
    results = []
    K = getattr(args, 'max_parallel_seeds', 4)
    for i in range(0, len(seeds), K):
        batch = seeds[i:i+K]
        params = [(sanitized, idt, files, subject_names, s) for s in batch]
        logger.info(f"Processing seeds batch {batch}")
        with Pool(processes=len(batch)) as pool:
            results.extend(pool.map(_run_one_seed_global, params))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info(f"Completed {len(results)}/{len(seeds)} seeds")

    # per-seed summary
    total_tta = np.array([r[0] for r in results])
    total_pre = np.array([r[1] for r in results])
    for s, tta, pre in zip(seeds, total_tta, total_pre):
        logger.info(f"[Summary] Seed {s} -> TTA={tta:.2f}%, Pre={pre:.2f}%")
        args.log.record(f"Seed {s} summary: TTA {tta:.2f}%, Pre {pre:.2f}%")

    # subject-level CI
    n, t_val = len(seeds), stats.t.ppf(0.975, len(seeds)-1)
    ci_t = t_val * total_tta.std() / np.sqrt(n)
    ci_p = t_val * total_pre.std() / np.sqrt(n)
    summary = (
      f"{target}: TTA {total_tta.mean():.3f}±{total_tta.std():.3f} "
      f"(95% CI [{total_tta.mean()-ci_t:.3f}, {total_tta.mean()+ci_t:.3f}]); "
      f"Pre {total_pre.mean():.3f}±{total_pre.std():.3f} "
      f"(95% CI [{total_pre.mean()-ci_p:.3f}, {total_pre.mean()+ci_p:.3f}])"
    )
    logger.info(summary)
    args.log.record(summary)

    # reload target labels & stack per-seed CSVs
    _, _, X_tar, y_tar = read_mi_combine_tar(args)
    preds_pre = np.vstack([
        np.loadtxt(
            os.path.join(
              args.result_dir,
              f"{args.data_name}_T-TIME_seed_{s}_subj{idt_str}_pre_probs.csv"
            ),
            delimiter=","
        )
        for s in seeds
    ])
    preds_tta = np.vstack([
        np.loadtxt(
            os.path.join(
              args.result_dir,
              f"{args.data_name}_T-TIME_seed_{s}_subj{idt_str}_tta_probs.csv"
            ),
            delimiter=","
        )
        for s in seeds
    ])

    # overall ensemble metrics
    # TTA as before
    mean_vote_tta = (preds_tta.mean(0)>0.5).astype(int)
    med_vote_tta  = (np.median(preds_tta,0)>0.5).astype(int)
    sml_vote_tta  = SML(preds_tta)
    acc_tta = [
      accuracy_score(y_tar, mean_vote_tta)*100,
      accuracy_score(y_tar, med_vote_tta)*100,
      accuracy_score(y_tar, sml_vote_tta)*100
    ]

    # Pre-TTA: **majority vote** on hard labels
    hard_pre     = (preds_pre>0.5).astype(int)
    maj_vote_pre = (hard_pre.sum(axis=0) > (hard_pre.shape[0]/2)).astype(int)
    med_vote_pre = (np.median(preds_pre,0)>0.5).astype(int)
    sml_vote_pre = SML(preds_pre)
    acc_pre_ens = [
      accuracy_score(y_tar, maj_vote_pre)*100,
      accuracy_score(y_tar, med_vote_pre)*100,
      accuracy_score(y_tar, sml_vote_pre)*100
    ]

    logger.info(
      f"Ensemble TTA:    MeanProb={acc_tta[0]:.2f}%, "
      f"MedianProb={acc_tta[1]:.2f}%, SML={acc_tta[2]:.2f}%"
    )
    args.log.record(
      f"Ensemble TTA:    MeanProb={acc_tta[0]:.2f}%, "
      f"MedianProb={acc_tta[1]:.2f}%, SML={acc_tta[2]:.2f}%"
    )
    logger.info(
      f"Ensemble Pre-TTA: MajorityVote={acc_pre_ens[0]:.2f}%, "
      f"MedianProb={acc_pre_ens[1]:.2f}%, SML={acc_pre_ens[2]:.2f}%"
    )
    args.log.record(
      f"Ensemble Pre-TTA: MajorityVote={acc_pre_ens[0]:.2f}%, "
      f"MedianProb={acc_pre_ens[1]:.2f}%, SML={acc_pre_ens[2]:.2f}%"
    )

    # per-session breakdown
    sess_tta_break = []
    sess_pre_break = []
    for idx, (s_idx,e_idx) in enumerate(args.tar_bounds):
        sess = args.session_names[idx]
        suby = y_tar[s_idx:e_idx]

        # TTA session
        subp_t = preds_tta[:, s_idx:e_idx]
        votes_t = [
          accuracy_score(suby, (subp_t.mean(0)>0.5).astype(int))*100,
          accuracy_score(suby, (np.median(subp_t,0)>0.5).astype(int))*100,
          accuracy_score(suby, SML(subp_t))*100
        ]
        sess_tta_break.append([sess]+votes_t)
        logger.info(f"TTA session {sess}: Mean={votes_t[0]:.2f}%, Median={votes_t[1]:.2f}%, SML={votes_t[2]:.2f}%")
        args.log.record(f"TTA session {sess}: Mean={votes_t[0]:.2f}%, Median={votes_t[1]:.2f}%, SML={votes_t[2]:.2f}%")

        # Pre-TTA session: majority vote
        subp_p   = preds_pre[:, s_idx:e_idx]
        hard_sp  = (subp_p>0.5).astype(int)
        maj_sp   = (hard_sp.sum(axis=0) > (hard_sp.shape[0]/2)).astype(int)
        med_sp   = (np.median(subp_p,0)>0.5).astype(int)
        sml_sp   = SML(subp_p)
        votes_p  = [
          accuracy_score(suby, maj_sp)*100,
          accuracy_score(suby, med_sp)*100,
          accuracy_score(suby, sml_sp)*100
        ]
        sess_pre_break.append([sess]+votes_p)
        logger.info(f"Pre-TTA session {sess}: Maj={votes_p[0]:.2f}%, Median={votes_p[1]:.2f}%, SML={votes_p[2]:.2f}%")
        args.log.record(f"Pre-TTA session {sess}: Maj={votes_p[0]:.2f}%, Median={votes_p[1]:.2f}%, SML={votes_p[2]:.2f}%")

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
        seeds = [2,3,4,5]

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