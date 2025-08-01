# check_segments_corr.py
# Find each trial_*_fixation_raw.npy inside trial_*_raw.npy
# using cross‑channel Pearson correlation instead of strict equality.
#
# Usage:   python check_segments_corr.py
# (no arguments – ROOT_DIR is hard‑coded)

import json, numpy as np, pathlib, sys

# ──────────────────── EDIT THIS IF YOU MOVE THE DATA ────────────────────
ROOT_DIR = pathlib.Path(
    r"E:\Exoskeleton_DL\DeepTransferEEG\iplementaion_runn\ttttttttt_11_20250722_110009"
).expanduser()

# Timing constants (ms) – must match constants.py
REST_MS     = 4_000
FIX_MS      = 1_500
OFFSET_MS   = 500      # DELAY_POST_STIMULUS
WINDOW_MS   = 4_000    # TRIAL_DURATION

SEARCH_MS   = 2_000    # search ±2 s (plenty when SR = 500 Hz)
CORR_THRESH = 0.95     # ≥ 0.95 = good match across all channels

# ────────────────────────────────────────────────────────────────────────
def load_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def flatten(a):
    """Return 1‑D float32 view (saves RAM for corr)."""
    return a.astype(np.float32, copy=False).ravel()

def best_corr(raw, seg, exp_start, half_band):
    """Slide seg across raw ±half_band samples, return (max_corr, best_shift)."""
    win = seg.shape[0]
    seg_vec = flatten(seg)
    best, shift_best = -1.0, None
    for sh in range(-half_band, half_band + 1):
        s0, s1 = exp_start + sh, exp_start + sh + win
        if s0 < 0 or s1 > raw.shape[0]:
            continue
        raw_vec = flatten(raw[s0:s1, :])
        # Pearson r between the two flattened vectors
        r = np.corrcoef(raw_vec, seg_vec)[0, 1]
        if r > best:
            best, shift_best = r, sh
    return best, shift_best

def verify_pair(raw_file):
    trial_stem   = raw_file.stem              # "trial_3_raw"
    trial_prefix = trial_stem[:-4]            # "trial_3"
    seg_file     = raw_file.with_name(f"{trial_prefix}_fixation_raw.npy")
    raw_meta_f   = raw_file.with_name(f"{raw_file.stem}_meta.json")
    seg_meta_f   = seg_file.with_name(f"{seg_file.stem}_meta.json")

    if not seg_file.exists() or not raw_meta_f.exists() or not seg_meta_f.exists():
        return trial_prefix, "segment_or_meta_missing"

    sr = load_json(raw_meta_f)["sampling_rate_hz"]          # 500 Hz
    raw = np.load(raw_file, mmap_mode="r")
    seg = np.load(seg_file)

    # quick sanity on shapes
    if raw.shape[1] != seg.shape[1] or seg.shape[0] != int(round(WINDOW_MS/1000*sr)):
        return trial_prefix, "shape_mismatch"

    exp_ms      = REST_MS + FIX_MS + OFFSET_MS              # 6000 ms
    exp_start   = int(round(exp_ms/1000*sr))                # samples
    half_band   = int(round(SEARCH_MS/1000*sr))             # samples to either side

    r, sh = best_corr(raw, seg, exp_start, half_band)

    if r >= CORR_THRESH:
        if sh == 0:
            status = "OK"
        else:
            ms = sh * 1000 / sr
            status = f"OK_shifted({ms:+.0f}ms)"
    else:
        status = f"no_match(max_r={r:.2f})"
    return trial_prefix, status

def main():
    if not ROOT_DIR.exists():
        sys.exit(f"[ERROR] Folder not found: {ROOT_DIR}")
    results = [
        verify_pair(f)
        for f in ROOT_DIR.rglob("trial_*_raw.npy")
        if "_fixation_" not in f.name
    ]
    if not results:
        sys.exit("[ERROR] No trial_*_raw.npy files found.")
    width = max(len(n) for n, _ in results)
    for n, s in sorted(results):
        print(f"{n:<{width}}  {s}")

if __name__ == "__main__":
    main()
