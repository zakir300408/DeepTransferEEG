import numpy as np
import os
import glob

from step_3_load_setup_model import setup_inference_pipeline, infer_pre_tta, infer_tta

class EnsembleRunner:
    def __init__(self, seeds, sample_rate):
        # Initialize all your seeds’ models, R’s, buffers, etc. once
        self.pre_state = {}
        self.tta_state = {}
        for seed in seeds:
            m_pre, R_pre, args_pre, _ = setup_inference_pipeline(
                seed, mode="pre_tta"
            )
            self.pre_state[seed] = {"model": m_pre, "R": R_pre, "args": args_pre}

            (m_tta, opt, R_tta, buf), args_tta, _ = setup_inference_pipeline(
                seed, mode="tta"
            )
            self.tta_state[seed] = {
                "model": m_tta,
                "opt": opt,
                "R": R_tta,
                "buffer": buf,
                "args": args_tta,
            }

        self.sample_rate = sample_rate
        self._trial_idx = 0

    def predict(self, trial):
        """
        Call this on each *single* trial in sequence.
        It updates each seed’s R (and TTA buffer) in-place and returns
        (ensemble_label_pre_tta, ensemble_label_tta).
        """
        # input trial is already preprocessed; skip preprocess_trial

        # Pre-TTA: collect full class probabilities
        probs_pre = []
        for seed, st in self.pre_state.items():
            a = st["args"]
            p, new_R = infer_pre_tta(
                model=st["model"],
                trial=trial[:, :a.time_sample_num],
                args=a,
                R=st["R"],
                trial_idx=self._trial_idx
            )
            st["R"] = new_R
            probs_pre.append(p[0])  # [p_class0, p_class1]

        # TTA: collect full class probabilities
        probs_tta = []
        for seed, st in self.tta_state.items():
            a = st["args"]
            p, new_R, new_buf = infer_tta(
                model=st["model"],
                optimizer=st["opt"],
                trial=trial[:, :a.time_sample_num],
                args=a,
                R=st["R"],
                data_cum=st["buffer"],
                trial_idx=self._trial_idx
            )
            st["R"] = new_R
            st["buffer"] = new_buf
            probs_tta.append(p[0])  # [p_class0, p_class1]

        # Ensemble: average probabilities and threshold at 0.5 for the positive class
        avg_pre = np.mean(np.stack(probs_pre, axis=0), axis=0)
        avg_tta = np.mean(np.stack(probs_tta, axis=0), axis=0)
        label_pre = int(avg_pre[1] > 0.5)
        label_tta = int(avg_tta[1] > 0.5)

        self._trial_idx += 1
        return label_pre, label_tta, avg_pre, avg_tta, probs_pre, probs_tta

# ----------------------
# Example live usage:
# ----------------------
if __name__ == "__main__":
    runner = EnsembleRunner(
        seeds=[2,3],
        sample_rate=100
    )
    # iterate over all fixation trials without resetting models/covariances
    data_dir = r"E:\Exoskeleton_DL\DeepTransferEEG\testt\rer_1_20250619_113035"
    pattern = os.path.join(data_dir, "trial_*_fixation.npy")
    trial_files = sorted(glob.glob(pattern))
    for trial_file in trial_files:
        trial = np.load(trial_file)
        pre_lbl, tta_lbl, avg_pre, avg_tta, p_pre, p_tta = runner.predict(trial)
        print(f"\nFile: {os.path.basename(trial_file)}")
        print(f"  Pre-TTA → label={pre_lbl}, avg_probs={avg_pre}")
        print(f"  TTA    → label={tta_lbl}, avg_probs={avg_tta}")
        print(f"  Per-seed Pre-TTA: {p_pre}")
        print(f"  Per-seed TTA:     {p_tta}")