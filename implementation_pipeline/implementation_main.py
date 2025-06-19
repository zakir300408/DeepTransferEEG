import numpy as np

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


#let's rather load a single trial from a path instead of using the dataset loader
    # Example: load a single trial from a file
    trial = np.load(r"E:\Exoskeleton_DL\DeepTransferEEG\testt\rer_1_20250619_113035\trial_1_fixation.npy")  # shape (n_ch, n_time)
    pre_lbl, tta_lbl, avg_pre, avg_tta, p_pre, p_tta = runner.predict(trial)
    print(f"Pred→ pre-TTA={pre_lbl}, tta-ensemble={tta_lbl}, avg_pre={avg_pre}, avg_tta={avg_tta}")
    print(f"Individual Probabilities pre-TTA: {p_pre}, TTA: {p_tta}")
    print("=== Prediction Results ===")
    print(f"Pre-TTA Label: {pre_lbl}, TTA Label: {tta_lbl}")
    print(f"Average Pre-TTA probabilities [class0, class1]: {avg_pre}")
    print(f"Average TTA probabilities [class0, class1]: {avg_tta}")
    print("--- Per-seed Pre-TTA probabilities [class0, class1]:")
    for seed, prob in zip(runner.pre_state.keys(), p_pre):
        print(f"  Seed {seed}: {prob}")
    print("--- Per-seed TTA probabilities [class0, class1]:")
    for seed, prob in zip(runner.tta_state.keys(), p_tta):
        print(f"  Seed {seed}: {prob}")