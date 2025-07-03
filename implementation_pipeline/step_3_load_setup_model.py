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
import torch
import torch.nn as nn
import torch.optim as optim
import torch.linalg as LA
import numpy as np
import glob
import copy  # new import

def add_project_paths(root=None, subs=("", "tl", "runs")):
    root = root or os.path.dirname(os.path.dirname(__file__))
    for sub in subs:
        p = os.path.join(root, sub) if sub else root
        if p not in sys.path:
            sys.path.insert(0, p)

add_project_paths()

from utils.alg_utils import EA_online
from utils.network import backbone_net
from utils.loss import Entropy
from implementation_pipeline.utils_gui.constants import (
    FEATURE_DEEP_DIM, SAMPLE_RATE, CHN, TIME_SAMPLE_NUM,
    LR, MAX_TTA, STRIDE, STEPS, T, CONF_THRESH
)




class InferenceEngine:
    """Base class: handles model loading, tensor conversion, prediction & alignment utils."""
    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.models: list[nn.Sequential] = []
        self._build_model()
        self._load_weights()

    def _build_model(self):
        # build prototype only (no weights yet)
        netF, netC = backbone_net(self.args, return_type="xy")
        self._netF = netF
        self._netC = netC

    def _load_weights(self):
        ckpt_dir = f"./runs/{self.args.data_name}"
        pattern = f"{self.args.backbone}_S*_seed{self.args.SEED}_best.ckpt"
        paths = sorted(glob.glob(os.path.join(ckpt_dir, pattern)))
        if not paths:
            raise FileNotFoundError(f"No checkpoints found matching {pattern} in {ckpt_dir}")

        # use the one built prototype to instantiate all ensemble members
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

        # pick primary model for adaptation/optimizer
        self.model = self.models[0]

    def _to_tensor(self, data):
        self._diagnose_input(data, context="to_tensor")
        t = torch.from_numpy(data) if not isinstance(data, torch.Tensor) else data
        return t.to(self.device, dtype=torch.float32)

    def _get_transform(self, R):
        # Diagnostic: check R before eig
        if np.isnan(R).any() or np.isinf(R).any():
            print("DIAGNOSTIC: NaN/Inf in covariance matrix R before eig")
        Rr = R + np.eye(R.shape[0]) * 1e-6
        Rt = torch.from_numpy(Rr).to(self.device, dtype=torch.float32)
        vals, vecs = LA.eigh(Rt)
        # Diagnostic: check eigenvalues
        if torch.isnan(vals).any() or torch.isinf(vals).any():
            print("DIAGNOSTIC: NaN/Inf in eigenvalues of R")
        if (vals <= 0).any():
            print("DIAGNOSTIC: Non-positive eigenvalues in R, min eigenvalue:", vals.min().item())
        return vecs @ torch.diag(vals.pow(-0.5)) @ vecs.T

    def _align_sample(self, sample, R, trial_idx):
        # Diagnostic: check sample before alignment
        arr = sample.cpu().numpy() if hasattr(sample, "cpu") else sample
        if np.isnan(arr).any() or np.isinf(arr).any():
            print("DIAGNOSTIC: NaN/Inf in sample before alignment")
        if np.all(arr == 0):
            print("DIAGNOSTIC: Sample before alignment is all zeros")
        R_new = EA_online(arr, R, trial_idx)
        # Diagnostic: check R_new after EA_online
        if np.isnan(R_new).any() or np.isinf(R_new).any():
            print("DIAGNOSTIC: NaN/Inf in R_new after EA_online")
        T = self._get_transform(R_new)
        aligned = T @ sample
        # Diagnostic: check aligned output
        if torch.isnan(aligned).any() or torch.isinf(aligned).any():
            print("DIAGNOSTIC: NaN/Inf in aligned sample output")
        return aligned, R_new

    def _predict(self, inp):
        # Check for NaN/Inf in input
        if torch.isnan(inp).any() or torch.isinf(inp).any():
            print("WARNING: Input to model contains NaN or Inf")
        # ensemble inference
        probs_list = []
        for m in self.models:
            with torch.no_grad():
                _, out = m(inp)
                probs_list.append(torch.softmax(out, dim=1))
        stacked = torch.stack(probs_list, dim=0)  # (n_models, batch, classes)
        return stacked.mean(dim=0)

    def _diagnose_input(self, data, context=""):
        """Prints stats and locations of NaN/Inf in input data for debugging."""
        arr = data if isinstance(data, np.ndarray) else data.cpu().numpy()
        nan_mask = np.isnan(arr)
        inf_mask = np.isinf(arr)
        if nan_mask.any() or inf_mask.any():
            print(f"DIAGNOSTIC: Detected NaN/Inf in input {context}")
            print(f"  Shape: {arr.shape}")
            print(f"  NaN count: {np.sum(nan_mask)}")
            print(f"  Inf count: {np.sum(inf_mask)}")
            # Optionally print indices (for small arrays)
            if arr.size < 1000:
                print(f"  NaN indices: {np.argwhere(nan_mask)}")
                print(f"  Inf indices: {np.argwhere(inf_mask)}")
            print(f"  Min: {np.nanmin(arr)}, Max: {np.nanmax(arr)}")


class PreTTAEngine(InferenceEngine):
    """Pre-TTA inference: just optional alignment + one softmax."""
    def __init__(self, args):
        super().__init__(args)
        self.R = np.zeros((args.chn, args.chn)) if args.align else None

    def infer(self, trial, trial_idx=0):
        x = self._to_tensor(trial).view(1, 1, self.args.chn, self.args.time_sample_num)
        # Diagnostic: check after tensor conversion
        self._diagnose_input(x.cpu().numpy(), context="after _to_tensor in PreTTAEngine")
        if self.args.align and self.R is not None:
            sample = x.squeeze(0).squeeze(0)
            aligned, self.R = self._align_sample(sample, self.R, trial_idx)
            x = aligned.view(1, 1, self.args.chn, self.args.time_sample_num)
            # Diagnostic: check after alignment
            self._diagnose_input(x.cpu().numpy(), context="after alignment in PreTTAEngine")
        probs = self._predict(x)
        return probs.cpu().numpy(), self.R


class TTAEngine(InferenceEngine):
    """Full TTA pipeline: rolling buffer, optional alignment, adaptation, re-infer."""
    def __init__(self, args):
        super().__init__(args)
        # one optimizer for each ensemble member
        self.optimizers = [optim.Adam(m.parameters(), lr=args.lr) for m in self.models]
        self.R = np.zeros((args.chn, args.chn)) if args.align else None
        self.buffer = None

    def infer(self, trial, trial_idx=0):
        x = self._to_tensor(trial).view(1, 1, self.args.chn, self.args.time_sample_num)
        # Diagnostic: check after tensor conversion
        self._diagnose_input(x.cpu().numpy(), context="after _to_tensor in TTAEngine")
        self.buffer = x if self.buffer is None else torch.cat((self.buffer, x), dim=0)
        if self.buffer.size(0) > self.args.max_tta:
            self.buffer = self.buffer[-self.args.max_tta:]

        if self.args.align and self.R is not None:
            sample = x.squeeze(0).squeeze(0)
            aligned, self.R = self._align_sample(sample, self.R, trial_idx)
            x_test = aligned.view(1, 1, self.args.chn, self.args.time_sample_num)
            # Diagnostic: check after alignment
            self._diagnose_input(x_test.cpu().numpy(), context="after alignment in TTAEngine")
        else:
            x_test = x

        # first-pass inference
        softmax_out = self._predict(x_test)
        conf, _ = softmax_out.max(dim=1)

        # decide if we should adapt
        if (conf.item() >= self.args.conf_thresh
            and (trial_idx + 1) >= self.args.max_tta
            and (trial_idx + 1) % self.args.stride == 0):
            print(f">>> Adapting at trial {trial_idx+1} (conf={conf.item():.3f})")

            batch = self.buffer[-self.args.max_tta:]
            if self.args.align and self.R is not None:
                T = self._get_transform(self.R)
                raw = batch.squeeze(1)  # (win, chn, time)
                aligned = torch.einsum('ij,bjt->bit', T, raw)
                batch = aligned.unsqueeze(1)

            # adaptation steps for every model in the ensemble
            for m in self.models:
                m.train()
            for step in range(self.args.steps):
                # zero all optimizers
                for opt in self.optimizers:
                    opt.zero_grad()
                # accumulate loss across ensemble
                total_loss = 0.0
                for m in self.models:
                    _, out = m(batch)
                    p = torch.softmax(out / self.args.t, dim=1)
                    loss = (
                        torch.mean(Entropy(p)) +
                        torch.sum(p.mean(dim=0) * torch.log(p.mean(dim=0) + self.args.epsilon))
                    )
                    total_loss = total_loss + loss
                total_loss.backward()
                # step all optimizers
                for opt in self.optimizers:
                    opt.step()
                print(f"    Step {step+1}/{self.args.steps}, total_loss={total_loss.item():.4f}")
            for m in self.models:
                m.eval()

            # re-infer after adaptation
            softmax_out = self._predict(x_test)

        return softmax_out.cpu().numpy(), self.R, self.buffer


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

    print(f"✔ {mode.upper()} engine setup complete")
    return engine


if __name__ == "__main__":
    engine = setup_inference_pipeline(mode="pre_tta")
    print("Inference engine ready.")
