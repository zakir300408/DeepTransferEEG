"""
step_3_load_setup_model.py
────────────────────────────────────────────────────────────
2025-07-07  • EEGNet TTA engine with incremental robustness patches

Patches already applied
-----------------------
 0.  BN-only adaptation (TENT-style)
 2a. Rolling window via collections.deque
 2b. Use torch.cat (not stack) so tensor ranks stay correct
 3.  Percentile gating: adapt using top-p % most-confident samples
 4.  Entropy-plateau early-stop with patience
 5.  Dynamic confidence gating: adapt top_p using entropy history

To keep the diff history clean, *no* further changes are mixed in.
"""

# ----------------------------------------------------------------------
#  Std-lib & compatibility shims
# ----------------------------------------------------------------------
import typing
try:
    from typing import Self
except ImportError:  # Python < 3.11
    from typing_extensions import Self
typing.Self = Self

import os
import sys
import glob
import copy
import argparse
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.linalg as LA

# ----------------------------------------------------------------------
#  Project-specific helpers (imported from your repo)
# ----------------------------------------------------------------------
def add_project_paths(root: str | None = None, subs=("", "tl", "runs")) -> None:
    """Prepend repo sub-folders to PYTHONPATH so relative imports work."""
    root = root or Path(__file__).resolve().parents[1]
    for sub in subs:
        p = str(Path(root, sub)) if sub else str(root)
        if p not in sys.path:
            sys.path.insert(0, p)

add_project_paths()

# project modules -----------------------------------------------------
from utils.alg_utils import EA_online
from utils.network import backbone_net
from utils.loss import Entropy
from implementation_pipeline.utils_gui.constants import (  # noqa: E501
    FEATURE_DEEP_DIM, SAMPLE_RATE, CHN, TIME_SAMPLE_NUM,
    LR, MAX_TTA, STRIDE, STEPS, T, CONF_THRESH,
)

# ----------------------------------------------------------------------
#  Patch-0 helper: expose only BN γ/β parameters
# ----------------------------------------------------------------------
def trainable_bn_params(model: nn.Module):
    """
    Yield BatchNorm affine parameters and freeze all others.

    Following TENT, we also switch off running-stat tracking so each
    BN uses only the current batch statistics.
    """
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.weight.requires_grad = m.bias.requires_grad = True
            m.track_running_stats = False
            yield m.weight
            yield m.bias
        else:
            # disable gradients for everything else
            for p in m.parameters(recurse=False):
                p.requires_grad = False

# ----------------------------------------------------------------------
#  Base inference engine
# ----------------------------------------------------------------------
class InferenceEngine:
    """Handles model construction, weight loading and shared utilities."""

    def __init__(self, args: argparse.Namespace):
        self.args   = args
        self.device = args.device
        self.models: list[nn.Sequential] = []   # ensemble
        self._build_model_prototype()
        self._load_weights()

    # -- model build / load -------------------------------------------------
    def _build_model_prototype(self) -> None:
        netF, netC = backbone_net(self.args, return_type="xy")
        self._netF = netF
        self._netC = netC

    def _load_weights(self) -> None:
        ckpt_dir = Path("runs") / self.args.data_name
        pattern  = f"{self.args.backbone}_S*_seed{self.args.SEED}_best.ckpt"
        paths    = sorted(ckpt_dir.glob(pattern))
        if not paths:
            raise FileNotFoundError(
                f"No checkpoints match {pattern!r} in {ckpt_dir}"
            )

        protoF, protoC = self._netF, self._netC
        for ckpt in paths:
            netF = copy.deepcopy(protoF)
            netC = copy.deepcopy(protoC)
            model = nn.Sequential(netF, netC).to(self.device)

            state = torch.load(ckpt, map_location=self.device)
            model.load_state_dict(state, strict=True)
            model.eval()
            self.models.append(model)
            print(f"✔ Loaded ensemble member from {ckpt}")

        self.model = self.models[0]  # first one is “primary”

    # -- tensor / alignment helpers ----------------------------------------
    def _to_tensor(self, arr: np.ndarray | torch.Tensor) -> torch.Tensor:
        if not isinstance(arr, torch.Tensor):
            arr = torch.from_numpy(arr)
        return arr.to(self.device, dtype=torch.float32)

    def _diagnose_array(self, arr: np.ndarray, msg: str = "") -> None:
        if np.isnan(arr).any() or np.isinf(arr).any():
            print(f"[DIAG] NaN/Inf detected {msg} "
                  f"(shape={arr.shape}, NaN={np.isnan(arr).sum()}, "
                  f"Inf={np.isinf(arr).sum()})")

    def _get_transform(self, R: np.ndarray) -> torch.Tensor:
        """Whitener via eigen-decomp (adds εI for stability)."""
        self._diagnose_array(R, "in covariance R")
        Rt = torch.from_numpy(R + np.eye(R.shape[0]) * 1e-6).to(
            self.device, dtype=torch.float32
        )
        vals, vecs = LA.eigh(Rt)
        if (vals <= 0).any():
            print("[DIAG] non-positive eigenvalues; min =", vals.min().item())
        return vecs @ torch.diag(vals.pow(-0.5)) @ vecs.T

    def _align_sample(
        self, x: torch.Tensor, R: np.ndarray, idx: int
    ) -> tuple[torch.Tensor, np.ndarray]:
        """Eye-artifacts alignment on-the-fly (EA_online)."""
        arr = x.cpu().numpy() if isinstance(x, torch.Tensor) else x
        self._diagnose_array(arr, "before EA_online")
        R_new = EA_online(arr, R, idx)
        Tmat  = self._get_transform(R_new)
        aligned = Tmat @ x
        return aligned, R_new

    def _predict(self, inp: torch.Tensor, temp: float = 1.0) -> torch.Tensor:
        """Ensemble average of softmax probabilities with temperature."""
        probs = []
        for m in self.models:
            with torch.no_grad():
                _, out = m(inp)
                probs.append(torch.softmax(out / temp, 1))
        return torch.stack(probs, 0).mean(0)

# ----------------------------------------------------------------------
#  Pre-TTA engine (no adaptation, optional per-trial alignment)
# ----------------------------------------------------------------------
class PreTTAEngine(InferenceEngine):
    def __init__(self, args):
        super().__init__(args)
        self.R = np.zeros((args.chn, args.chn)) if args.align else None

    @torch.no_grad()
    def infer(self, trial: np.ndarray, idx: int = 0):
        x = self._to_tensor(trial).view(1, 1, self.args.chn, self.args.time_sample_num)

        if self.args.align and self.R is not None:
            sample  = x.squeeze(0).squeeze(0)
            aligned, self.R = self._align_sample(sample, self.R, idx)
            x = aligned.view(1, 1, self.args.chn, self.args.time_sample_num)

        probs = self._predict(x)
        return probs.cpu().numpy(), self.R, None

# ----------------------------------------------------------------------
#  TTA engine with robustness patches
# ----------------------------------------------------------------------
class TTAEngine(InferenceEngine):
    """Stream-time adaptation with BN-only updates and robust triggers."""

    def __init__(self, args):
        super().__init__(args)

        # BN-only optimisers (Patch-0)
        self.optimizers = [
            optim.Adam(list(trainable_bn_params(m)), lr=args.lr)
            for m in self.models
        ]

        # State buffers
        self.R           = np.zeros((args.chn, args.chn)) if args.align else None
        self.buffer      : deque[torch.Tensor] = deque(maxlen=args.max_tta)
        self.buffer_conf : deque[float]        = deque(maxlen=args.max_tta)
        
        # New: Entropy history buffer for dynamic confidence gating
        self.entropy_history : deque[float] = deque(maxlen=args.max_tta)

        # Early-stop state
        self.patience = args.patience

    def _calculate_dynamic_top_p(self):
        """Calculate dynamic top_p based on entropy history quantile."""
        if len(self.entropy_history) < 3:  # Need minimum history
            return self.args.top_p  # Fall back to default

        entropy_arr = np.array(self.entropy_history)
        q75 = np.quantile(entropy_arr, 0.75)
        dynamic_p = 1.0 - q75                      # invert entropy → confidence
        dynamic_p = min(0.95, max(0.1, dynamic_p)) # clip between 0.1 and 0.9
        return dynamic_p

    # ------------------------------------------------------------------
    def infer(self, trial: np.ndarray, idx: int = 0):
        x = self._to_tensor(trial).view(1, 1, self.args.chn, self.args.time_sample_num)

        # store raw sample & provisional confidence
        self.buffer.append(x.detach())

        # alignment for on-line prediction input
        if self.args.align and self.R is not None:
            sample   = x.squeeze(0).squeeze(0)
            aligned, self.R = self._align_sample(sample, self.R, idx)
            x_test = aligned.view(1, 1, self.args.chn, self.args.time_sample_num)
        else:
            x_test = x

        # first pass (apply temperature)
        soft_out = self._predict(x_test, self.args.t)
        conf_val = soft_out.max(1).values.item()
        self.buffer_conf.append(conf_val)
        
        # Calculate and store entropy for dynamic gating
        with torch.no_grad():
            entropy_val = Entropy(soft_out).item()
            self.entropy_history.append(entropy_val)

        # ---------- adaptation trigger --------------------------------
        if (
            len(self.buffer) == self.args.max_tta
            and (idx + 1) % self.args.stride == 0
        ):
            print(f">>> Adapting at trial {idx+1}")

            # Dynamic confidence gating: use entropy history to determine top_p
            dynamic_top_p = self._calculate_dynamic_top_p()
            print(f"    · Using dynamic top_p: {dynamic_top_p:.3f}")
            
            # Select top samples based on dynamic threshold
            k = max(1, int(np.ceil(dynamic_top_p * len(self.buffer))))
            conf_arr = np.array(self.buffer_conf)
            top_idx  = conf_arr.argsort()[::-1][:k]        # k highest confidences
            batch    = torch.cat([self.buffer[i] for i in top_idx], 0)  # (k,1,C,T)

            # window-level alignment (optional)
            if self.args.align and self.R is not None:
                Tmat = self._get_transform(self.R)
                raw  = batch.squeeze(1)                    # (k,C,T)
                batch = torch.einsum('ij,bjt->bit', Tmat, raw).unsqueeze(1)

            # ---------- inner optimisation (entropy plateau) ----------
            best_e     = float("inf")
            wait       = 0
            best_state = [copy.deepcopy(m.state_dict()) for m in self.models]

            for step in range(self.args.steps):
                for opt in self.optimizers:
                    opt.zero_grad()

                tot_loss, ent_acc = 0.0, 0.0
                for m in self.models:
                    _, out = m(batch)
                    p   = torch.softmax(out / self.args.t, 1)
                    ent = torch.mean(Entropy(p))
                    loss = ent + torch.sum(p.mean(0) * torch.log(p.mean(0) + self.args.epsilon))
                    tot_loss += loss
                    ent_acc  += ent.item()

                tot_loss.backward()
                for opt in self.optimizers:
                    opt.step()

                ent_avg = ent_acc / len(self.models)
                if ent_avg < best_e - 1e-4:          # improvement
                    best_state = [copy.deepcopy(m.state_dict()) for m in self.models]
                    best_e = ent_avg
                    wait   = 0
                else:
                    wait += 1
                if wait == self.patience:
                    print(f"    · early-stop at step {step+1} (entropy plateau)")
                    break

            # load best snapshot & switch back to eval
            for m, s in zip(self.models, best_state):
                m.load_state_dict(s)
                m.eval()

            # re-infer after adaptation (consistent temperature)
            soft_out = self._predict(x_test, self.args.t)

        return soft_out.cpu().numpy(), self.R, None

# ----------------------------------------------------------------------
#  Helper: assemble an inference engine
# ----------------------------------------------------------------------
def setup_inference_pipeline(seed: int = 2, mode: str = "tta"):
    args = argparse.Namespace(
        # dataset / model specs -----------------------
        data_name="CustomEpoch",
        SEED=seed,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        backbone="EEGNet",
        chn=CHN,
        time_sample_num=TIME_SAMPLE_NUM,
        class_num=2,
        feature_deep_dim=FEATURE_DEEP_DIM,
        # TTA knobs -----------------------------------
        align=True,
        lr=LR,
        max_tta=MAX_TTA,
        stride=STRIDE,
        steps=STEPS,
        t=T,
        epsilon=1e-5,
        # new robustness knobs ------------------------
        top_p=0.6,         # initial/fallback value - now dynamically adjusted
        patience=6,         # early-stop patience
        # legacy (still used by PreTTA) ---------------
        conf_thresh=CONF_THRESH,
        sample_rate=SAMPLE_RATE,
    )

    engine_cls = {"pre_tta": PreTTAEngine, "tta": TTAEngine}.get(mode)
    if engine_cls is None:
        raise ValueError("mode must be 'pre_tta' or 'tta'")
    engine = engine_cls(args)
    print(f"✔ {mode.upper()} engine ready (seed={seed})")
    return engine

# ----------------------------------------------------------------------
#  Quick CLI sanity check
# ----------------------------------------------------------------------
if __name__ == "__main__":
    eng = setup_inference_pipeline(mode="pre_tta")
    print("✓ self-test finished – engine instantiated.")
    eng = setup_inference_pipeline(mode="pre_tta")
    print("✓ self-test finished – engine instantiated.")
