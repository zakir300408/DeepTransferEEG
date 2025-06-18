import numpy as np

from step_2_preprocess import preprocess_trial
from step_3_load_setup_model import setup_inference_pipeline, infer_pre_tta, infer_tta
from ttime_ensemble import SML

class EnsembleRunner:
    def __init__(self, dataset_name, subject_id, seeds, sample_rate):
        # Initialize all your seeds’ models, R’s, buffers, etc. once
        self.pre_state = {}
        self.tta_state = {}
        for seed in seeds:
            m_pre, R_pre, args_pre, _ = setup_inference_pipeline(
                dataset_name, subject_id, seed, mode="pre_tta"
            )
            self.pre_state[seed] = {"model": m_pre, "R": R_pre, "args": args_pre}

            (m_tta, opt, R_tta, buf), args_tta, _ = setup_inference_pipeline(
                dataset_name, subject_id, seed, mode="tta"
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
        proc = preprocess_trial(trial, self.sample_rate)

        # Pre-TTA
        probs_pre = []
        for seed, st in self.pre_state.items():
            a = st["args"]
            p, new_R = infer_pre_tta(
                model=st["model"],
                trial=proc[:, :a.time_sample_num],
                args=a,
                R=st["R"],
                trial_idx=self._trial_idx
            )
            st["R"] = new_R
            probs_pre.append(p[0, 1])

        # TTA
        probs_tta = []
        for seed, st in self.tta_state.items():
            a = st["args"]
            p, new_R, new_buf = infer_tta(
                model=st["model"],
                optimizer=st["opt"],
                trial=proc[:, :a.time_sample_num],
                args=a,
                R=st["R"],
                data_cum=st["buffer"],
                trial_idx=self._trial_idx
            )
            st["R"] = new_R
            st["buffer"] = new_buf
            probs_tta.append(p[0, 1])

        # Ensemble helper
        def _vote(arr):
            try:
                return int(SML(arr.reshape(-1,1))[0])
            except Exception:
                return int(arr.mean() > 0.5)

        label_pre = _vote(np.array(probs_pre))
        label_tta = _vote(np.array(probs_tta))

        self._trial_idx += 1
        return label_pre, label_tta

# ----------------------
# Example live usage:
# ----------------------
if __name__ == "__main__":
    from step_1_load_data import load_custom_epoch_data

    # you only use load_custom_epoch_data here to simulate getting trials
    X, y, _ = load_custom_epoch_data(dataset_name="CustomEpoch")

    runner = EnsembleRunner(
        dataset_name="CustomEpoch",
        subject_id=0,
        seeds=[2,3],
        sample_rate=100
    )

    # # now “live” process 50 trials, one at a time
    # for trial in X[:50]:
    #     pre_lbl, tta_lbl = runner.predict(trial)
    #     print(f"Pred→ pre-TTA={pre_lbl}, tta-ensemble={tta_lbl}")

#let's rather load a single trial from a path instead of using the dataset loader
    # Example: load a single trial from a file
    trial = np.load(r"E:\Exoskeleton_DL\DeepTransferEEG\testt\uu_1_20250618_161656\trial_3_fixation.npy")  # shape (n_ch, n_time)
    pre_lbl, tta_lbl = runner.predict(trial)
    print(f"Pred→ pre-TTA={pre_lbl}, tta-ensemble={tta_lbl}")