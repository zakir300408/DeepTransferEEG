import numpy as np
import os
import glob

from step_3_load_setup_model import setup_inference_pipeline

class EnsembleRunner:
    def __init__(self, seeds, mode="both"):
        """
        seeds: list of random seeds
        mode: one of "pre_tta", "tta", or "both"
        """
        self.mode = mode
        self._trial_idx = 0

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
            for eng in self.pre_engines.values():
                p, _ = eng.infer(trial, self._trial_idx)   # returns (probs, R)
                probs_pre.append(p.squeeze(0))
            probs_pre = np.stack(probs_pre, axis=0)
            avg_pre = probs_pre.mean(axis=0)
            label_pre = int(avg_pre[1] > 0.5)

        # TTA ensemble
        avg_tta = label_tta = probs_tta = None
        if self.mode in ("tta", "both"):
            probs_tta = []
            for eng in self.tta_engines.values():
                p, _, _ = eng.infer(trial, self._trial_idx)  # returns (probs, R, buffer)
                probs_tta.append(p.squeeze(0))
            probs_tta = np.stack(probs_tta, axis=0)
            avg_tta = probs_tta.mean(axis=0)
            label_tta = int(avg_tta[1] > 0.5)

        self._trial_idx += 1
        return label_pre, label_tta, avg_pre, avg_tta, probs_pre, probs_tta


if __name__ == "__main__":
    runner = EnsembleRunner(seeds=[2, 3], mode="both")
    data_dir = r"E:\Exoskeleton_DL\DeepTransferEEG\testt\rer_1_20250619_113035"
    files = sorted(glob.glob(os.path.join(data_dir, "trial_*_fixation.npy")))

    for f in files:
        trial = np.load(f)
        pre_lbl, tta_lbl, avg_pre, avg_tta, p_pre, p_tta = runner.predict(trial)
        print(f"\nFile: {os.path.basename(f)}")
        if runner.mode in ("pre_tta", "both"):
            print(f"  Pre-TTA → label={pre_lbl}, avg_probs={avg_pre}")
            print(f"    per-seed: {p_pre}")
        if runner.mode in ("tta", "both"):
            print(f"  TTA     → label={tta_lbl}, avg_probs={avg_tta}")
            print(f"    per-seed: {p_tta}")
