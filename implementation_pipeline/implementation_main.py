import numpy as np
import os
import glob
import json

from step_3_load_setup_model import setup_inference_pipeline
from utils_gui.constants import SEEDS, CONF_THRESH

class EnsembleRunner:
    def __init__(self, seeds, mode="both", threshold: float = 0.5):
        """
        seeds: list of random seeds
        mode: one of "pre_tta", "tta", or "both"
        threshold: probability cutoff for positive class
        """
        self.mode = mode
        self._trial_idx = 0
        self.threshold = threshold

        # build per‐seed engines
        self.pre_engines = {}
        self.tta_engines = {}
        for seed in seeds:
            if mode in ("pre_tta", "both"):
                self.pre_engines[seed] = setup_inference_pipeline(seed, mode="pre_tta")
            if mode in ("tta", "both"):
                self.tta_engines[seed] = setup_inference_pipeline(seed, mode="tta")

    def predict(self, trial: np.ndarray):
        """
        Run one trial through each seed’s engine(s), update their internal state,
        and return all the ensemble‐aggregated labels and probabilities.
        """
        # Pre-TTA ensemble
        avg_pre = label_pre = probs_pre = None
        if self.mode in ("pre_tta", "both"):
            probs_pre = []
            shp = None
            for eng in self.pre_engines.values():
                p, _, _ = eng.infer(trial, self._trial_idx)   # returns (probs, R)
                p0 = np.take(p, 0, axis=0)  # always take first sample along axis0
                if shp is None:
                    shp = p0.shape
                elif p0.shape != shp:
                    raise ValueError(f"Per-seed prob shape mismatch: {p0.shape} vs {shp}")
                probs_pre.append(p0.astype(np.float32))
            probs_pre = np.stack(probs_pre, axis=0)
            avg_pre = probs_pre.mean(axis=0)
            label_pre = int(avg_pre[1] > self.threshold)

        # TTA ensemble
        avg_tta = label_tta = probs_tta = None
        if self.mode in ("tta", "both"):
            probs_tta = []
            shp = None
            for eng in self.tta_engines.values():
                p, _, _ = eng.infer(trial, self._trial_idx)  # returns (probs, R, buffer)
                p0 = np.take(p, 0, axis=0)
                if shp is None:
                    shp = p0.shape
                elif p0.shape != shp:
                    raise ValueError(f"Per-seed prob shape mismatch: {p0.shape} vs {shp}")
                probs_tta.append(p0.astype(np.float32))
            probs_tta = np.stack(probs_tta, axis=0)
            avg_tta = probs_tta.mean(axis=0)
            label_tta = int(avg_tta[1] > self.threshold)

        self._trial_idx += 1
        if self.mode == "pre_tta":
            return label_pre, avg_pre, probs_pre
        elif self.mode == "tta":
            return label_tta, avg_tta, probs_tta
        else:  # both
            return label_pre, label_tta, avg_pre, avg_tta, probs_pre, probs_tta


if __name__ == "__main__":
    runner = EnsembleRunner(seeds=SEEDS, mode="both", threshold=CONF_THRESH)
    data_dir = r"E:\Exoskeleton_DL\DeepTransferEEG\iplementaion_runn\shen_2_20250708_161905"
    # sort by numeric trial index to ensure correct adaptation order
    files = sorted(
        glob.glob(os.path.join(data_dir, "trial_*_fixation.npy")),
        key=lambda f: int(os.path.basename(f).split("_")[1]),
    )

    # load and validate ground truth + predicted labels
    with open(os.path.join(data_dir, "trial_results.json"), "r") as jf:
        results = json.load(jf)
    # determine trials_list: support dict-with-"trials" or plain list
    if isinstance(results, dict) and "trials" in results and isinstance(results["trials"], list):
        trials_list = results["trials"]
    elif isinstance(results, list):
        trials_list = results
    else:
        raise ValueError("trial_results.json must be a dict with a 'trials' list or a list of trials")
    # validate each trial entry
    for i, r in enumerate(trials_list, start=1):
        for key in ("trial_index", "ground_truth", "predicted_label"):
            if key not in r:
                raise KeyError(f"Missing '{key}' in trial entry {i}: {r}")
    # build strict maps
    gt_map   = {r["trial_index"]: r["ground_truth"]    for r in trials_list}
    pred_map = {r["trial_index"]: r["predicted_label"] for r in trials_list}

    pre_correct = tta_correct = 0
    n_pre = n_tta = 0

    for f in files:
        trial = np.load(f)
        tidx = int(os.path.basename(f).split("_")[1])
        gt = gt_map[tidx]   # strict lookup

        # unpack based on mode
        if runner.mode == "both":
            pre_lbl, tta_lbl, avg_pre, avg_tta, p_pre, p_tta = runner.predict(trial)
        elif runner.mode == "pre_tta":
            pre_lbl, avg_pre, p_pre = runner.predict(trial)
            tta_lbl, avg_tta, p_tta = None, None, None
        else:  # "tta"
            tta_lbl, avg_tta, p_tta = runner.predict(trial)
            pre_lbl, avg_pre, p_pre = None, None, None

        print(f"\nFile: {os.path.basename(f)}")
        if runner.mode in ("pre_tta", "both"):
            print(f"  Pre-TTA → label={pre_lbl}, avg_probs={avg_pre}")
            print(f"    per-seed: {p_pre}")
            # update pre-TTA accuracy
            if pre_lbl == gt:
                pre_correct += 1
            n_pre += 1
        if runner.mode in ("tta", "both"):
            print(f"  TTA     → label={tta_lbl}, avg_probs={avg_tta}")
            print(f"    per-seed: {p_tta}")
            # update TTA accuracy
            if tta_lbl == gt:
                tta_correct += 1
            n_tta += 1

    # final accuracy
    if runner.mode in ("pre_tta", "both"):
        print(f"\nPre-TTA accuracy: {pre_correct}/{n_pre} = {pre_correct/n_pre:.2%}")
    if runner.mode in ("tta", "both"):
        print(f"TTA accuracy: {tta_correct}/{n_tta} = {tta_correct/n_tta:.2%}")
