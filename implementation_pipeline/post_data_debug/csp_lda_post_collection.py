#!/usr/bin/env python3
"""
EEG Trial Processing Script
Loads EDF trials and labels, preprocesses the data, trains CSP+LDA, and evaluates performance.
"""

# ----------------------------
# Configuration & Constants
# ----------------------------
DATA_FOLDER           = r"E:\Exoskeleton_DL\DeepTransferEEG\iplementaion_runn\Zhao Xu_1_20250701_171837"
LABEL_FILE            = "trial_results.json"
TRIAL_PATTERN         = "trial_{idx}_raw.edf"
TRUNCATE_START_SEC    = 5.5
TRUNCATE_END_SEC      = 2
BANDPASS_LOW_HZ       = 8.0
BANDPASS_HIGH_HZ      = 32.0
NOTCH_FREQ_HZ         = 50.0
ORIGINAL_SFREQ_HZ     = 500
TARGET_SFREQ_HZ       = 100
NUM_TRAIN_TRIALS      = 20
CSP_NUM_COMPONENTS    = 4
LOG_FILE              = "processing.log"

# ----------------------------
# Imports & Logging Setup
# ----------------------------
import os
import glob
import json
import logging

import numpy as np
import mne
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import accuracy_score

# Configure logger (console + file)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
fmt = "%(asctime)s %(levelname)s: %(message)s"
handlers = [
    logging.StreamHandler(),
    logging.FileHandler(LOG_FILE, mode="w")
]
for h in handlers:
    h.setFormatter(logging.Formatter(fmt))
    logger.addHandler(h)


# ----------------------------
# Data Loading Functions
# ----------------------------
def load_labels(label_path: str) -> dict[int, int]:
    with open(label_path, "r") as f:
        trials = json.load(f).get("trials", [])
    return {t["trial_index"]: t["ground_truth"] for t in trials}


def load_trials(data_folder: str) -> list[tuple[int, mne.io.Raw]]:
    raws: list[tuple[int, mne.io.Raw]] = []
    pattern = os.path.join(data_folder, "trial_*_raw.edf")
    for path in glob.glob(pattern):
        idx = int(os.path.basename(path).split("_")[1])
        raw = mne.io.read_raw_edf(path, preload=True)
        raws.append((idx, raw))
    return sorted(raws, key=lambda x: x[0])


def load_data(data_folder: str) -> tuple[list[mne.io.Raw], dict[int,int]]:
    label_path = os.path.join(data_folder, LABEL_FILE)
    labels = load_labels(label_path)
    trials = load_trials(data_folder)
    trial_indices = [idx for idx, _ in trials]

    missing_in_json = set(trial_indices) - set(labels)
    missing_edf     = set(labels) - set(trial_indices)
    if missing_in_json or missing_edf:
        raise ValueError(
            f"Mismatch between EDF and JSON entries: "
            f"EDF only: {missing_in_json}, JSON only: {missing_edf}"
        )

    raws = [raw for _, raw in trials]
    logger.info(f"Loaded {len(raws)} trials and {len(labels)} labels")
    sample = raws[0]
    logger.info(
        f"Sample shape: {sample.n_times}×{sample.info['nchan']}, "
        f"sfreq={sample.info['sfreq']}Hz, "
        f"channels={sample.ch_names}"
    )
    return raws, labels


# ----------------------------
# Preprocessing Function
# ----------------------------
def preprocess_trial(raw: mne.io.Raw) -> np.ndarray:
    raw_copy = raw.copy()
    orig_shape = raw_copy.get_data().shape
    logger.info(f"Preprocessing trial: original shape {orig_shape}")

    # Print total duration of the trial
    total_duration = raw_copy.times[-1]
    logger.info(f"Total duration of trial: {total_duration:.2f} seconds")

    # Adaptive cropping
    duration = raw_copy.times[-1]
    if TRUNCATE_START_SEC + TRUNCATE_END_SEC >= duration:
        logger.warning(
            f"Truncate window ({TRUNCATE_START_SEC}s + {TRUNCATE_END_SEC}s) "
            f"exceeds duration ({duration:.2f}s). Skipping crop."
        )
        tmin, tmax = 0.0, duration
    else:
        tmin = TRUNCATE_START_SEC
        tmax = duration - TRUNCATE_END_SEC
    raw_copy.crop(tmin=tmin, tmax=tmax)
    post_crop_shape = raw_copy.get_data().shape
    logger.info(f"After truncation: shape {post_crop_shape}")

    # Bandpass
    raw_copy.filter(
        l_freq=BANDPASS_LOW_HZ,
        h_freq=BANDPASS_HIGH_HZ,
        fir_design='firwin'
    )

    # FFT‐based notch (no long FIR filter)
    raw_copy.notch_filter(
        freqs=NOTCH_FREQ_HZ,
        method='spectrum_fit'
    )

    # Resample
    raw_copy.resample(sfreq=TARGET_SFREQ_HZ)
    processed = raw_copy.get_data()
    logger.info(
        f"After resample: shape {processed.shape}, "
        f"sfreq={raw_copy.info['sfreq']}Hz"
    )
    return processed


# ----------------------------
# CSP + LDA Functions
# ----------------------------
def fit_csp_and_lda(
    X_train: np.ndarray, y_train: np.ndarray
) -> tuple[CSP, LDA]:
    csp = CSP(
        n_components=CSP_NUM_COMPONENTS,
        reg=None, log=False, norm_trace=False
    )
    csp.fit(X_train, y_train)
    X_feat = csp.transform(X_train)

    lda = LDA()
    lda.fit(X_feat, y_train)
    return csp, lda


def evaluate(
    csp: CSP, lda: LDA, X_test: np.ndarray, y_test: np.ndarray
) -> float:
    X_feat = csp.transform(X_test)
    y_pred = lda.predict(X_feat)
    acc = accuracy_score(y_test, y_pred)
    logger.info(f"Test accuracy: {acc:.3%}")
    return acc


# ----------------------------
# Main Orchestration
# ----------------------------
def main():
    raws, labels = load_data(DATA_FOLDER)
    trial_idxs = sorted(labels.keys())

    processed = [preprocess_trial(r) for r in raws]
    X = np.array(processed)  # (n_trials, n_chans, n_times)
    y = np.array([labels[i] for i in trial_idxs])

    # Split
    X_train, X_test = X[:NUM_TRAIN_TRIALS], X[NUM_TRAIN_TRIALS:]
    y_train, y_test = y[:NUM_TRAIN_TRIALS], y[NUM_TRAIN_TRIALS:]

    csp, lda = fit_csp_and_lda(X_train, y_train)
    _ = evaluate(csp, lda, X_test, y_test)


if __name__ == "__main__":
    main()
