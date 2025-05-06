"""

Online EA-CCSP-LDA vs. CSP-LDA baseline vs. EA-LOSO baseline, with full verbosity
and with LOSO models trained on EA-aligned source sessions:

  • Build two source domains per subject:
      – RAW source  (unused by LOSO now)
      – EA-aligned source (each other-subject session aligned individually)
  • For each target subject:
      – Train & save an **EA-LOSO** model (CSP-LDA) once on all EA-aligned source trials.
  • For each session of that subject:
      – Repeat N_REPEAT times with a fixed RNG seed:
          1) Sample MAX_N target indices (wraparound, ensure both classes)
          2) leftover = all other indices
          3) For each n in [MIN_N, …, MAX_N]:
             a) train_idx = pool[:n], test_idx = pool[n:]+leftover
             b) **Baseline CSP-LDA**: train on raw target[train_idx], test on raw target[test_idx]
             c) **EA-LOSO**: apply pretrained LOSO model (trained on EA-aligned source)
                to EA-aligned target[test_idx]
             d) **EA-CCSP-LDA**: compute EA on raw target[train_idx], align all target,
                train on [aligned target[train_idx] + EA-aligned source], test on aligned remainder
      – After repeats: compute mean±std for each n and method.
  • Verbose logs: every pool sampling, EA transforms, training sizes, test sizes, accuracies.
"""
import os, glob, json, pickle
import numpy as np
from utils.Euclidean_alignment     import EuclideanAlignment
from utils.CSPClassifier                 import CSPClassifier
from scipy.io import loadmat
from scipy.signal import butter, filtfilt

# ─── Configuration ────────────────────────────────────────────────────────────
ROOT_FOLDER     = r"E:\Exoskeleton_DL\XK_work\Data_Epoch"
MIN_N           = 4
MAX_N           = 20
N_STEP          = 4
N_REPEAT        = 3
RNG_SEED        = 42
CSP_PAIRS       = 6
CLASSIFIER_TYPE = "LDA"
OUTPUT_FOLDER   = "ea_ccsp_lda_with_loso_ea_verbose"
MODEL_FOLDER    = os.path.join(OUTPUT_FOLDER, "models")
# ─────────────────────────────────────────────────────────────────────────────

###############################################################################
# Data Loading Functions
###############################################################################
def preprocess_trial(trial, Fs=200, filter_band=(8, 30), time_window=(1, 4)):
    """
    Preprocess a single EEG trial: filter and extract time window.
    
    Parameters:
    - trial: Raw EEG trial data
    - Fs: Sampling frequency in Hz
    - filter_band: Tuple of (low_freq, high_freq) for bandpass filtering
    - time_window: Tuple of (start_time, end_time) in seconds
    
    Returns:
    - Preprocessed trial with shape (channels, samples)
    """
    filt_n = 4
    Wn = [filter_band[0] / (Fs/2), filter_band[1] / (Fs/2)]
    filter_b, filter_a = butter(filt_n, Wn, btype='band')
    
    # Filter the trial data
    data_filter = filtfilt(filter_b, filter_a, trial, axis=0)
    
    # Extract the specified time window
    StartTimePoint = int(time_window[0] * Fs)
    EndTimePoint = int(time_window[1] * Fs)
    segment = data_filter[StartTimePoint:EndTimePoint, :]
    
    # Transpose to get (channels, samples) format
    return segment.T

def load_segmented_data(mat_file, Fs=200):
    """
    Load a MAT file containing pre-segmented data with keys:
      - 'MyEpoch': shape (n_trials, 1600, 31)
      - 'MyLabel': shape (1, n_trials)
    
    Each trial is preprocessed using the preprocess_trial function.
    
    Returns raw EEG trials and their labels.
    """
    mat = loadmat(mat_file)
    MyEpoch = mat['MyEpoch']  
    MyLabel = mat['MyLabel']  
    n_trials = MyEpoch.shape[0]
    labels = MyLabel.flatten()  
    
    print(f"Processing file: {mat_file} ({n_trials} trials)")
    
    raw_trials = []
    for i in range(n_trials):
        trial = MyEpoch[i, :, :]
        X = preprocess_trial(trial, Fs=Fs, filter_band=(8, 40), time_window=(1, 4))
        raw_trials.append(X)
    
    trials_Type = labels.tolist()
    return raw_trials, trials_Type

def compute_curves_verbose(
    target_trials, target_labels,
    source_ea, source_ea_labels,
    clf_loso,
    n_repeat=N_REPEAT, seed=RNG_SEED,
    apply_ea_loso=False, apply_ea_ccsp=False,
    loso_name="EA-LOSO", ccsp_name="EA-CCSP"
):
    steps     = list(range(MIN_N, MAX_N+1, N_STEP))
    n_steps   = len(steps)
    n_trials  = len(target_trials)

    base_accs = np.zeros((n_repeat, n_steps))
    loso_accs = np.zeros((n_repeat, n_steps))
    ccsp_accs = np.zeros((n_repeat, n_steps))

    rng = np.random.default_rng(seed)
    for r in range(n_repeat):
        print(f"\n  >>> Repeat {r+1}/{n_repeat} <<<")
        pool = list(rng.integers(0, n_trials, size=MAX_N))
        while len({target_labels[i] for i in pool[:MIN_N]}) < 2:
            pool = list(rng.integers(0, n_trials, size=MAX_N))
        leftover = [i for i in range(n_trials) if i not in pool]

        print(f"    Pool indices: {pool}")
        print(f"    Leftover test indices: {leftover}")

        for idx, n in enumerate(steps):
            train_idx = pool[:n]
            test_idx  = pool[n:] + leftover

            print(f"\n    --- n = {n} calibration trials ---")
            print(f"    train_idx: {train_idx}")
            print(f"    test_idx : {test_idx}")

            # Baseline CSP-LDA (target-only)
            X_cal   = [target_trials[i] for i in train_idx]
            y_cal   = [target_labels[i] for i in train_idx]
            X_test  = [target_trials[i] for i in test_idx]
            y_test  = [target_labels[i] for i in test_idx]
            print(f"    [Baseline] Training on {len(X_cal)} raw target trials")
            clf_base = CSPClassifier(
                X_cal[0].shape[0], CSP_PAIRS,
                classifier_type=CLASSIFIER_TYPE
            )
            clf_base.fit(X_cal, y_cal)
            acc_base, _ = clf_base.score(X_test, y_test)
            print(f"    [Baseline] Accuracy = {acc_base*100:.2f}%")
            base_accs[r, idx] = acc_base

            # EA-LOSO block
            if apply_ea_loso:
                print(f"    [{loso_name}] Testing pretrained {loso_name} on {len(test_idx)} aligned target trials")
                # We need to align test trials with target-level EA
                T_tgt = EuclideanAlignment.compute_transformation(
                    [target_trials[i] for i in train_idx]
                )
                tgt_aligned = [T_tgt @ X for X in target_trials]
                X_test_ea = [tgt_aligned[i] for i in test_idx]
                acc_loso, _ = clf_loso.score(X_test_ea, y_test)
                print(f"    [{loso_name}] Accuracy = {acc_loso*100:.2f}%")
                loso_accs[r, idx] = acc_loso
            else:
                print("    [LOSO] Skipping EA alignment")
                acc_loso = np.nan

            # EA-CCSP block
            if apply_ea_ccsp:
                print(f"    [{ccsp_name}] Computing EA on raw target trials {train_idx}")
                # T_tgt already computed
                print(f"      -> EA transform shape: {T_tgt.shape}")
                print(f"    [{ccsp_name}] Applying EA to all {n_trials} target trials")
                X_cal_ea = [tgt_aligned[i] for i in train_idx]
                X_test_ccsp = X_test_ea  # same aligned test
                print(f"    [{ccsp_name}] Training on {len(X_cal_ea)} aligned target + "
                      f"{len(source_ea)} aligned source trials")
                X_train = X_cal_ea + source_ea
                y_train = y_cal     + source_ea_labels
                clf_ccsp = CSPClassifier(
                    X_train[0].shape[0], CSP_PAIRS,
                    classifier_type=CLASSIFIER_TYPE
                )
                clf_ccsp.fit(X_train, y_train)
                acc_ccsp, _ = clf_ccsp.score(X_test_ccsp, y_test)
                print(f"    [{ccsp_name}] Accuracy = {acc_ccsp*100:.2f}%")
                ccsp_accs[r, idx] = acc_ccsp
            else:
                print("    [CCSP] Skipping EA alignment")
                acc_ccsp = np.nan

    # mean ± std
    baseline_stats = {}
    loso_stats     = {}
    ea_ccsp_stats  = {}
    for i, n in enumerate(steps):
        mb, sb = base_accs[:,i].mean(), base_accs[:,i].std(ddof=1)
        ml, sl = loso_accs[:,i].mean(), loso_accs[:,i].std(ddof=1)
        mc, sc = ccsp_accs[:,i].mean(), ccsp_accs[:,i].std(ddof=1)
        baseline_stats[n] = (float(mb), float(sb))
        loso_stats[n]     = (float(ml), float(sl))
        ea_ccsp_stats[n]  = (float(mc), float(sc))

    return baseline_stats, loso_stats, ea_ccsp_stats

def main(apply_ea_loso=True, apply_ea_ccsp=True, loso_name="EA-LOSO", ccsp_name="EA-CCSP"):
    os.makedirs(MODEL_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    subjects = sorted(
        d for d in os.listdir(ROOT_FOLDER)
        if os.path.isdir(os.path.join(ROOT_FOLDER, d))
    )

    data = {}
    for subj in subjects:
        sessions = []
        for m in sorted(glob.glob(os.path.join(ROOT_FOLDER, subj, "*.mat"))):
            trials, labels = load_segmented_data(m)
            name = os.path.splitext(os.path.basename(m))[0]
            sessions.append((name, trials, labels))
        data[subj] = sessions

    all_results = {}

    for subj in subjects:
        print(f"\n=== Subject: {subj} ===")

        # Build EA-aligned source domain
        source_ea, source_ea_labels = [], []
        for other in subjects:
            if other == subj: continue
            for sess, s_tr, s_lb in data[other]:
                print(f"  [Source EA] subj={other}, sess={sess}, trials={len(s_tr)}")
                T_src = EuclideanAlignment.compute_transformation(s_tr)
                aligned = [T_src @ X for X in s_tr]
                source_ea.extend(aligned)
                source_ea_labels.extend(s_lb)
        print(f"  Source EA total trials: {len(source_ea)}")

        # Train EA-LOSO once per subject
        print("  [EA-LOSO] Training LOSO model on EA-aligned source")
        clf_loso = CSPClassifier(
            source_ea[0].shape[0], CSP_PAIRS,
            classifier_type=CLASSIFIER_TYPE
        )
        clf_loso.fit(source_ea, source_ea_labels)
        model_path = os.path.join(MODEL_FOLDER, f"ea_loso_{subj}.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(clf_loso, f)
        print(f"  [EA-LOSO] Saved model to {model_path}")

        subj_results = {}
        for sess, tgt_tr, tgt_lb in data[subj]:
            print(f"\n Processing session '{sess}' ({len(tgt_tr)} trials)")
            if len(tgt_tr) < MIN_N:
                print("  SKIP: not enough trials")
                continue

            b_stats, l_stats, c_stats = compute_curves_verbose(
                tgt_tr, tgt_lb,
                source_ea, source_ea_labels,
                clf_loso=clf_loso,
                apply_ea_loso=apply_ea_loso,
                apply_ea_ccsp=apply_ea_ccsp,
                loso_name=loso_name,
                ccsp_name=ccsp_name
            )

            print("\n  n  | Baseline    | EA-LOSO    | EA-CCSP")
            for n in sorted(b_stats):
                mb, sb = b_stats[n]
                ml, sl = l_stats[n]
                mc, sc = c_stats[n]
                print(f" {n:2d}  | {mb*100:5.2f}±{sb*100:5.2f}% | "
                      f"{ml*100:5.2f}±{sl*100:5.2f}% | "
                      f"{mc*100:5.2f}±{sc*100:5.2f}%")

            subj_results[sess] = {
                "baseline": b_stats,
                "ea_loso" : l_stats,
                "ea_ccsp" : c_stats
            }

        all_results[subj] = subj_results

    # Save
    out_file = os.path.join(OUTPUT_FOLDER, "results_with_ea_loso_verbose.json")
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_file}")

if __name__ == "__main__":
    main()
