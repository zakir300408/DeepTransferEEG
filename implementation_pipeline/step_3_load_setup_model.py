# step_3_load_setup_model.py
# -----------------------------------------------------------
# 2025-07-07 – patched version
#   • Patch 1 : BN-only adaptation (TENT-style)
#   • Patch 2 : rolling deque buffer
#   • Patch 2b: use torch.cat (not stack) → correct tensor rank
# -----------------------------------------------------------

# ---- compat shim for typing.Self (torch.compile / dynamo) ----
import typing
try:
    from typing import Self
except ImportError:
    from typing_extensions import Self
typing.Self = Self

import os
import sys
import argparse
import glob
import copy
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.linalg as LA
import numpy as np

# ------------------------------------------------------------------
# Helper: add project root + sub-dirs to PYTHONPATH
# ------------------------------------------------------------------
def add_project_paths(root=None, subs=("", "tl", "runs")):
    root = root or os.path.dirname(os.path.dirname(__file__))
    for sub in subs:
        p = os.path.join(root, sub) if sub else root
        if p not in sys.path:
            sys.path.insert(0, p)

add_project_paths()

# ------------------------------------------------------------------
# Project-specific utilities (unchanged)
# ------------------------------------------------------------------
from utils.alg_utils import EA_online
from utils.network import backbone_net
from utils.loss import Entropy
from implementation_pipeline.utils_gui.constants import (
    FEATURE_DEEP_DIM, SAMPLE_RATE, CHN, TIME_SAMPLE_NUM,
    LR, MAX_TTA, STRIDE, STEPS, T, CONF_THRESH
)

# ------------------------------------------------------------------
# Patch 1: only BN affine parameters are trainable
# ------------------------------------------------------------------
def trainable_bn_params(model: nn.Module):
    """Yield BatchNorm γ,β parameters and freeze all others."""
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.weight.requires_grad = True
            m.bias.requires_grad = True
            m.track_running_stats = False  # use current batch stats (TENT)
            yield m.weight
            yield m.bias
        else:
            for p in m.parameters(recurse=False):
                p.requires_grad = False

# ------------------------------------------------------------------
# Base inference engine
# ------------------------------------------------------------------
class InferenceEngine:
    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.models: list[nn.Sequential] = []
        self._build_model()
        self._load_weights()

    # ----- model build / load ------------------------------------------------
    def _build_model(self):
        netF, netC = backbone_net(self.args, return_type="xy")
        self._netF = netF
        self._netC = netC

    def _load_weights(self):
        ckpt_dir = f"./runs/{self.args.data_name}"
        pattern = f"{self.args.backbone}_S*_seed{self.args.SEED}_best.ckpt"
        paths = sorted(glob.glob(os.path.join(ckpt_dir, pattern)))
        if not paths:
            raise FileNotFoundError(
                f"No checkpoints found matching {pattern} in {ckpt_dir}"
            )

        protoF, protoC = self._netF, self._netC
        for ckpt in paths:
            netF = copy.deepcopy(protoF)
            netC = copy.deepcopy(protoC)
            model = nn.Sequential(netF, netC).to(self.device)

            state = torch.load(ckpt, map_location=self.device)
            model.load_state_dict(state)
            model.eval()
            self.models.append(model)
            print(f"✔ Loaded model from {ckpt}")

        self.model = self.models[0]  # primary for adaptation

    # ----- helpers -----------------------------------------------------------
    def _to_tensor(self, data):
        self._diagnose_input(data, context="to_tensor")
        t = torch.from_numpy(data) if not isinstance(data, torch.Tensor) else data
        return t.to(self.device, dtype=torch.float32)

    def _get_transform(self, R):
        if np.isnan(R).any() or np.isinf(R).any():
            print("DIAG: NaN/Inf in covariance R")
        Rt = torch.from_numpy(R + np.eye(R.shape[0]) * 1e-6).to(
            self.device, dtype=torch.float32
        )
        vals, vecs = LA.eigh(Rt)
        if torch.isnan(vals).any() or torch.isinf(vals).any() or (vals <= 0).any():
            print("DIAG: issues in eigenvalues")
        return vecs @ torch.diag(vals.pow(-0.5)) @ vecs.T

    def _align_sample(self, sample, R, trial_idx):
        arr = sample.cpu().numpy() if hasattr(sample, "cpu") else sample
        if np.isnan(arr).any() or np.isinf(arr).any() or np.all(arr == 0):
            print("DIAG: invalid sample before alignment")
        R_new = EA_online(arr, R, trial_idx)
        if np.isnan(R_new).any() or np.isinf(R_new).any():
            print("DIAG: invalid R_new")
        T = self._get_transform(R_new)
        aligned = T @ sample
        return aligned, R_new

    def _predict(self, inp):
        if torch.isnan(inp).any() or torch.isinf(inp).any():
            print("WARN: NaN/Inf in model input")
        probs = []
        for m in self.models:
            with torch.no_grad():
                _, out = m(inp)
                probs.append(torch.softmax(out, 1))
        return torch.stack(probs, 0).mean(0)

    def _diagnose_input(self, data, context=""):
        arr = data if isinstance(data, np.ndarray) else data.cpu().numpy()
        if np.isnan(arr).any() or np.isinf(arr).any():
            print(f"DIAG: NaN/Inf detected in {context}")

# ------------------------------------------------------------------
# Pre-TTA engine (no adaptation)
# ------------------------------------------------------------------
class PreTTAEngine(InferenceEngine):
    def __init__(self, args):
        super().__init__(args)
        self.R = np.zeros((args.chn, args.chn)) if args.align else None

    def infer(self, trial, trial_idx=0):
        x = self._to_tensor(trial).view(1, 1, self.args.chn, self.args.time_sample_num)
        if self.args.align and self.R is not None:
            sample = x.squeeze(0).squeeze(0)
            aligned, self.R = self._align_sample(sample, self.R, trial_idx)
            x = aligned.view(1, 1, self.args.chn, self.args.time_sample_num)
        probs = self._predict(x)
        return probs.cpu().numpy(), self.R

# ------------------------------------------------------------------
# TTA engine with Patch 1 + Patch 2 (+ 2b)
# ------------------------------------------------------------------
class TTAEngine(InferenceEngine):
    def __init__(self, args):
        super().__init__(args)

        # Patch 1: BN-only parameters
        self.optimizers = [
            optim.Adam(list(trainable_bn_params(m)), lr=args.lr)
            for m in self.models
        ]

        self.R = np.zeros((args.chn, args.chn)) if args.align else None
        self.buffer: deque[torch.Tensor] | None = None  # Patch 2

    def infer(self, trial, trial_idx=0):
        x = self._to_tensor(trial).view(1, 1, self.args.chn, self.args.time_sample_num)

        # Patch 2: rolling window via deque
        if self.buffer is None:
            self.buffer = deque(maxlen=self.args.max_tta)
        self.buffer.append(x.detach())  # keep clone

        # optional per-sample alignment
        if self.args.align and self.R is not None:
            sample = x.squeeze(0).squeeze(0)
            aligned, self.R = self._align_sample(sample, self.R, trial_idx)
            x_test = aligned.view(1, 1, self.args.chn, self.args.time_sample_num)
        else:
            x_test = x

        # forward pass
        softmax_out = self._predict(x_test)
        conf = softmax_out.max(1).values

        # trigger adaptation
        if (
            len(self.buffer) == self.args.max_tta
            and conf.item() >= self.args.conf_thresh
            and (trial_idx + 1) % self.args.stride == 0
        ):
            print(f">>> Adapting at trial {trial_idx+1} (conf={conf.item():.3f})")
            batch = torch.cat(list(self.buffer), 0)  # Patch 2b → (W,1,C,T)

            if self.args.align and self.R is not None:
                Tmat = self._get_transform(self.R)
                raw = batch.squeeze(1)               # (W,C,T)
                aligned = torch.einsum('ij,bjt->bit', Tmat, raw)
                batch = aligned.unsqueeze(1)         # back to (W,1,C,T)

            for m in self.models: m.train()
            for step in range(self.args.steps):
                for opt in self.optimizers: opt.zero_grad()
                total_loss = 0.0
                for m in self.models:
                    _, out = m(batch)
                    p = torch.softmax(out / self.args.t, 1)
                    loss = (
                        torch.mean(Entropy(p)) +
                        torch.sum(p.mean(0) * torch.log(p.mean(0) + self.args.epsilon))
                    )
                    total_loss += loss
                total_loss.backward()
                for opt in self.optimizers: opt.step()
                print(f"    Step {step+1}/{self.args.steps}, loss={total_loss.item():.4f}")
            for m in self.models: m.eval()
            softmax_out = self._predict(x_test)

        return softmax_out.cpu().numpy(), self.R, self.buffer

# ------------------------------------------------------------------
# helper to assemble an engine quickly
# ------------------------------------------------------------------
def setup_inference_pipeline(seed=2, mode="tta"):
    args = argparse.Namespace(
        data_name="CustomEpoch",
        SEED=seed,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        backbone="EEGNet",
        chn=CHN,
        time_sample_num=TIME_SAMPLE_NUM,
        class_num=2,
        feature_deep_dim=FEATURE_DEEP_DIM,
        align=True,
        lr=LR,
        max_tta=MAX_TTA,
        stride=STRIDE,
        steps=STEPS,
        t=T,
        conf_thresh=CONF_THRESH,
        epsilon=1e-5,
        sample_rate=SAMPLE_RATE,
    )

    if mode == "pre_tta":
        engine = PreTTAEngine(args)
    elif mode == "tta":
        engine = TTAEngine(args)
    else:
        raise ValueError("mode must be 'pre_tta' or 'tta'")

    print(f"✔ {mode.upper()} engine setup complete (seed={seed})")
    return engine

# ------------------------------------------------------------------
# CLI sanity check
# ------------------------------------------------------------------
if __name__ == "__main__":
    engine = setup_inference_pipeline(mode="pre_tta")
    print("Inference engine ready.")
