# Monkey-patch typing.Self for older Pythons so torch._dynamo can import it
import typing
try:
    from typing import Self
except ImportError:
    from typing_extensions import Self
typing.Self = Self

import os
import sys
import argparse
import typing
import torch
import torch.nn as nn
import torch.optim as optim
import torch.linalg as LA
import numpy as np

# add project root, tl, and runs directories to path
def add_project_paths():
    project_root = os.path.dirname(os.path.dirname(__file__))
    for sub in ("", "tl", "runs"):
        p = os.path.join(project_root, sub) if sub else project_root
        if p not in sys.path:
            sys.path.insert(0, p)

add_project_paths()

from utils.alg_utils import EA_online
from utils.network import backbone_net
from utils.loss import Entropy


def load_pretrained_model(args):
    """
    Instantiate backbone plus classifier, load pretrained weights, return eval model
    """
    netF, netC = backbone_net(args, return_type="xy")
    model = nn.Sequential(netF, netC).to(args.device)

    all_subs = "_".join(map(str, range(12)))
    ckpt_path = (
        f"./runs/{args.data_name}/"
        f"{args.backbone}_S{all_subs}_seed{args.SEED}_best.ckpt"
    )
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    state = torch.load(ckpt_path, map_location=args.device)
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded pretrained model from {ckpt_path}")
    return model


def setup_pre_tta_model(args):
    """
    Prepare model and initial EA state for pre-TTA inference
    """
    model = load_pretrained_model(args)
    R = 0 if args.align else None
    print("Pre-TTA model setup complete")
    return model, R


def setup_tta_model(args):
    """
    Prepare model, optimizer, EA state, and data buffer for full TTA
    """
    model, R = setup_pre_tta_model(args)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    data_cum = None
    print("TTA model setup complete (with optimizer and buffer)")
    return model, optimizer, R, data_cum


def apply_ea_alignment(sample, R, trial_idx, device):
    """
    Update R with the new sample and return aligned tensor plus new R
    """
    xn = sample.cpu().numpy()
    R = EA_online(xn, R, trial_idx)
    Rr = R + np.eye(R.shape[0]) * 1e-6
    Rt = torch.from_numpy(Rr).to(device=device, dtype=sample.dtype)
    vals, vecs = LA.eigh(Rt)
    T = vecs @ torch.diag(vals.pow(-0.5)) @ vecs.T
    return T @ sample, R


def infer_pre_tta(model, trial, args, R=None, trial_idx=0):
    """
    Single pre-TTA inference step with optional EA update.
    Returns (probs, R)
    """
    if not isinstance(trial, torch.Tensor):
        trial = torch.from_numpy(trial)
    trial = trial.to(args.device, dtype=torch.float32)

    if args.align and R is not None:
        trial, R = apply_ea_alignment(trial, R, trial_idx, args.device)

    inp = trial.view(1, 1, args.chn, args.time_sample_num)
    model.eval()
    with torch.no_grad():
        _, out = model(inp)
        probs = torch.softmax(out, dim=1)

    return probs.cpu().numpy(), R


def should_adapt(softmax_out, trial_idx, args):
    """
    Decide whether to perform test-time adaptation on this trial
    """
    conf, _ = softmax_out.max(dim=1)
    return (
        conf.item() >= args.conf_thresh
        and (trial_idx + 1) >= args.max_tta
        and (trial_idx + 1) % args.stride == 0
    )


def prepare_batch(data_cum, args, R=None):
    """
    Slice the last window from data_cum, optionally align it, return (batch_test, R)
    """
    win = args.max_tta
    raw = data_cum[-win:].squeeze(1)

    if args.align and R is not None:
        Rr = R + np.eye(R.shape[0]) * 1e-6
        Rt = torch.from_numpy(Rr).to(device=raw.device, dtype=raw.dtype)
        vals, vecs = LA.eigh(Rt)
        T = vecs @ torch.diag(vals.pow(-0.5)) @ vecs.T
        aligned = torch.einsum("ij,bjt->bit", T, raw)
        batch_test = aligned.unsqueeze(1)
    else:
        batch_test = data_cum[-win:]

    return batch_test, R


def perform_adaptation_steps(model, optimizer, batch_test, args):
    """
    Run adaptation gradient steps on batch_test
    """
    for _ in range(args.steps):
        optimizer.zero_grad()
        _, out = model(batch_test)
        p = torch.softmax(out / args.t, dim=1)
        cem = torch.mean(Entropy(p))
        m = p.mean(dim=0)
        mdr = torch.sum(m * torch.log(m + args.epsilon))
        (cem + mdr).backward()
        optimizer.step()


def infer_tta(model, optimizer, trial, args, R=None, data_cum=None, trial_idx=0):
    """
    Full TTA pipeline: initial inference, optional adaptation, final inference.
    Returns (probs, R, data_cum)
    """
    if not isinstance(trial, torch.Tensor):
        trial = torch.from_numpy(trial)
    trial = trial.to(args.device, dtype=torch.float32)
    sample = trial.view(1, 1, args.chn, args.time_sample_num)

    data_cum = sample if data_cum is None else torch.cat((data_cum, sample), 0)
    if data_cum.size(0) > args.max_tta:
        data_cum = data_cum[-args.max_tta :]

    if args.align and R is not None:
        aligned, R = apply_ea_alignment(trial, R, trial_idx, args.device)
        sample_test = aligned.view(1, 1, args.chn, args.time_sample_num)
    else:
        sample_test = sample

    model.eval()
    with torch.no_grad():
        _, out1 = model(sample_test)
        softmax_out = torch.softmax(out1, dim=1)

    if should_adapt(softmax_out, trial_idx, args):
        batch_test, R = prepare_batch(data_cum, args, R)
        model.train()
        perform_adaptation_steps(model, optimizer, batch_test, args)

        model.eval()
        with torch.no_grad():
            _, out2 = model(sample_test)
            softmax_out = torch.softmax(out2, dim=1)

    return softmax_out.cpu().numpy(), R, data_cum


def setup_inference_pipeline(seed=2, mode="tta"):
    """
    Build args and call either pre-TTA or TTA setup.
    Returns pipeline components plus args and mode
    """
    args = argparse.Namespace(
        data_name="CustomEpoch",
        SEED=seed,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        backbone="EEGNet",
        chn=27,
        time_sample_num=725,
        class_num=2,
        feature_deep_dim=704,
        align=True,
        lr=1e-4,
        max_tta=8,
        stride=1,
        steps=1,
        t=1.7,
        conf_thresh=0.1,
        epsilon=1e-5,
        sample_rate=100,
    )

    if mode == "pre_tta":
        model, R = setup_pre_tta_model(args)
        return model, R, args, mode
    elif mode == "tta":
        model, optimizer, R, data_cum = setup_tta_model(args)
        return (model, optimizer, R, data_cum), args, mode
    else:
        raise ValueError("mode must be 'pre_tta' or 'tta'")


if __name__ == "__main__":
    print("Model loading and setup functions ready.")
    try:
        model, R, args, mode = setup_inference_pipeline(
            mode="pre_tta"
        )
        print(f"Successfully setup {mode} pipeline")
    except Exception as e:
        print(f"Setup failed: {e}")
