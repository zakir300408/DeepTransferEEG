# Monkey-patch typing.Self for older Pythons so torch._dynamo can import it
import typing
try:
    from typing import Self as _Self
except ImportError:
    from typing_extensions import _Self
typing.Self = _Self

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.linalg as LA
import numpy as np

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
        self._build_model()

    def _build_model(self):
        netF, netC = backbone_net(self.args, return_type="xy")
        self.model = nn.Sequential(netF, netC).to(self.device)
        self._load_weights()

    def _load_weights(self):
        all_subs = "_".join(map(str, range(12)))
        ckpt_dir = f"./runs/{self.args.data_name}"
        ckpt_name = f"{self.args.backbone}_S{all_subs}_seed{self.args.SEED}_best.ckpt"
        ckpt_path = os.path.join(ckpt_dir, ckpt_name)
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        state = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()
        print(f"✔ Loaded pretrained model from {ckpt_path}")

    def _to_tensor(self, data):
        t = torch.from_numpy(data) if not isinstance(data, torch.Tensor) else data
        return t.to(self.device, dtype=torch.float32)

    def _get_transform(self, R):
        Rr = R + np.eye(R.shape[0]) * 1e-6
        Rt = torch.from_numpy(Rr).to(self.device, dtype=torch.float32)
        vals, vecs = LA.eigh(Rt)
        return vecs @ torch.diag(vals.pow(-0.5)) @ vecs.T

    def _align_sample(self, sample, R, trial_idx):
        # sample: Tensor of shape (chn, time)
        R_new = EA_online(sample.cpu().numpy(), R, trial_idx)
        T = self._get_transform(R_new)
        return T @ sample, R_new

    def _predict(self, inp):
        self.model.eval()
        with torch.no_grad():
            _, out = self.model(inp)
            return torch.softmax(out, dim=1)


class PreTTAEngine(InferenceEngine):
    """Pre-TTA inference: just optional alignment + one softmax."""
    def __init__(self, args):
        super().__init__(args)
        self.R = 0 if args.align else None

    def infer(self, trial, trial_idx=0):
        x = self._to_tensor(trial).view(1, 1, self.args.chn, self.args.time_sample_num)
        if self.args.align and self.R is not None:
            # strip batch/channel dims for alignment
            sample = x.squeeze(0).squeeze(0)
            aligned, self.R = self._align_sample(sample, self.R, trial_idx)
            x = aligned.view(1, 1, self.args.chn, self.args.time_sample_num)
        probs = self._predict(x)
        return probs.cpu().numpy(), self.R


class TTAEngine(InferenceEngine):
    """Full TTA pipeline: rolling buffer, optional alignment, adaptation, re-infer."""
    def __init__(self, args):
        super().__init__(args)
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)
        self.R = 0 if args.align else None
        self.buffer = None

    def infer(self, trial, trial_idx=0):
        # convert + shape
        x = self._to_tensor(trial).view(1, 1, self.args.chn, self.args.time_sample_num)
        # rolling buffer
        self.buffer = x if self.buffer is None else torch.cat((self.buffer, x), dim=0)
        if self.buffer.size(0) > self.args.max_tta:
            self.buffer = self.buffer[-self.args.max_tta:]

        # alignment for this sample
        if self.args.align and self.R is not None:
            sample = x.squeeze(0).squeeze(0)
            aligned, self.R = self._align_sample(sample, self.R, trial_idx)
            x_test = aligned.view(1, 1, self.args.chn, self.args.time_sample_num)
        else:
            x_test = x

        # first-pass inference
        softmax_out = self._predict(x_test)
        conf, _ = softmax_out.max(dim=1)

        # decide if we should adapt
        if (conf.item() >= self.args.conf_thresh
            and (trial_idx + 1) >= self.args.max_tta
            and (trial_idx + 1) % self.args.stride == 0):

            batch = self.buffer[-self.args.max_tta:]
            # optional alignment of entire batch
            if self.args.align and self.R is not None:
                T = self._get_transform(self.R)
                raw = batch.squeeze(1)  # (win, chn, time)
                aligned = torch.einsum('ij,bjt->bit', T, raw)
                batch = aligned.unsqueeze(1)

            # adaptation steps
            self.model.train()
            for _ in range(self.args.steps):
                self.optimizer.zero_grad()
                _, out = self.model(batch)
                p = torch.softmax(out / self.args.t, dim=1)
                loss = (
                    torch.mean(Entropy(p)) +
                    torch.sum(p.mean(dim=0) * torch.log(p.mean(dim=0) + self.args.epsilon))
                )
                loss.backward()
                self.optimizer.step()
            self.model.eval()

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
