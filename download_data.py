import os
import time
import glob
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, iirnotch
import pandas as pd
import numpy as np
import moabb

from moabb.datasets import BNCI2014001, BNCI2014002, BNCI2015001
from moabb.paradigms import MotorImagery, P300

# retry settings
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds

# slicing/constants
IGNORE_START_SECONDS_CUSTOM = 0
IGNORE_END_SECONDS_CUSTOM   = 1
SAMPLE_RATE_CUSTOM          = 200

# define your fixed augmentation windows (start_s, end_s)
AUG_WINDOWS_S = [
    (1.5, 5.5),
    (2.0, 6.0),
    (2.5, 6.5)
]

CH_NAMES = [
    'FP1','FZ','F3','F7','FT7','FC5','FC1','C3','T7','TP7','CP5','CP1','PZ',
    'P3','P7','O1','O2','P4','P8','TP8','CP6','CP2','CZ','C4','T8','FT8',
    'FC6','FC2','F4','F8','FP2'
]
KEEP_CHANNELS = [
    'FP1','FZ','F3','F7','FT7','FC5','FC1','C3','T7','TP7','CP5','CP1','PZ',
    'P3','P7','O1','O2','P4','P8','TP8','CP6','CP2','CZ','C4','T8','FT8',
    'FC6','FC2','F4','F8','FP2'
]
_KEEP_IDX = [CH_NAMES.index(ch) for ch in KEEP_CHANNELS]


def augment_defined_windows(X, y, windows_s, fs):
    """
    X: (n_trials, n_ch, n_samps)
    y: (n_trials,)
    windows_s: list of (start_s, end_s) tuples
    fs: sample rate
    returns: X_aug, y_aug where each trial is cropped at each window
    """
    X_list, y_list = [], []
    for Xi, yi in zip(X, y):
        n_samps = Xi.shape[1]              # <-- fixed here
        for start_s, end_s in windows_s:
            i0 = int(start_s * fs)
            i1 = int(end_s   * fs)
            if i1 <= n_samps:
                X_list.append(Xi[:, i0:i1])
                y_list.append(yi)
    X_aug = np.stack(X_list, axis=0)
    y_aug = np.array(y_list, dtype=y.dtype)
    return X_aug, y_aug


def dataset_to_file(dataset_name, data_save):
    moabb.set_log_level("ERROR")

    # ---- load raw data per dataset ----
    if dataset_name == 'BNCI2014001':
        dataset  = BNCI2014001()
        paradigm = MotorImagery(n_classes=4)
    elif dataset_name == 'BNCI2014002':
        dataset  = BNCI2014002()
        paradigm = MotorImagery(n_classes=2)
    elif dataset_name == 'BNCI2015001':
        dataset  = BNCI2015001()
        paradigm = MotorImagery(n_classes=2)
    elif dataset_name == 'CustomEpoch':
        root_dir = r"E:\Exoskeleton_DL\XK_work\Data_Epoch"
        mat_files = sorted(glob.glob(os.path.join(root_dir, "*", "*.mat")))
        if not mat_files:
            raise ValueError(f"No .mat files found in {root_dir}")

        all_X, all_y, meta_rows = [], [], []
        for fn in mat_files:
            mat   = loadmat(fn)
            raw_X = mat['MyEpoch']               # (n_trials, samples, channels)
            X     = raw_X.transpose(0, 2, 1)     # -> (n_trials, channels, samples)

            # select channels
            X = X[:, _KEEP_IDX, :]

            if not all_X:
                vals = X[0, :, 0]
                print("Sanity check (first trial, first sample) per kept channel:")
                for ch, v in zip(KEEP_CHANNELS, vals):
                    print(f"  {ch}: {v}")

            y = mat['MyLabel'].flatten().astype(int)
            y = y - y.min()
            all_X.append(X)
            all_y.append(y)
            meta_rows.append({'file': os.path.basename(fn), 'n_trials': X.shape[0]})

        X      = np.concatenate(all_X, axis=0)
        labels = np.concatenate(all_y, axis=0)
        meta   = pd.DataFrame(meta_rows)

    else:
        raise ValueError(f"Unknown dataset {dataset_name}")

    # ---- preprocess and save branch ----
    if data_save:
        print(f'preparing {dataset_name} data...')

        if dataset_name.startswith('BNCI'):
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    X, labels, meta = paradigm.get_data(
                        dataset=dataset,
                        subjects=dataset.subject_list[:]
                    )
                    break
                except Exception as e:
                    print(f"Attempt {attempt} failed: {e}")
                    if attempt == MAX_RETRIES:
                        raise
                    time.sleep(RETRY_DELAY)

        if dataset_name == 'CustomEpoch':
            # notch & bandpass
            nyq     = SAMPLE_RATE_CUSTOM / 2
            bn, an  = iirnotch(50.0/nyq, 30.0)
            b, a    = butter(5, [8/nyq, 32/nyq], btype='band')
            X       = filtfilt(bn, an, X, axis=2)
            X       = filtfilt(b, a,   X, axis=2)
            # z-score
            X       = (X - X.mean(axis=2, keepdims=True)) / (X.std(axis=2, keepdims=True) + 1e-8)

            # drop 1 s front AND 3 s end
            i0 = int(IGNORE_START_SECONDS_CUSTOM * SAMPLE_RATE_CUSTOM)
            i1 = int(IGNORE_END_SECONDS_CUSTOM   * SAMPLE_RATE_CUSTOM)
            X  = X[:, :, i0:-i1]

            # apply fixed-window augmentation
            X, labels = augment_defined_windows(
                X, labels,
                windows_s=AUG_WINDOWS_S,
                fs=SAMPLE_RATE_CUSTOM
            )

            # update meta for augmented counts
            meta['n_trials'] = meta['n_trials'] * len(AUG_WINDOWS_S)

            win_len = int((AUG_WINDOWS_S[0][1] - AUG_WINDOWS_S[0][0]) * SAMPLE_RATE_CUSTOM)
            print(f"Augmented to {X.shape[0]} trials, each window {win_len} samples long")

        # final counts & save
        ar_u, cnts = np.unique(labels, return_counts=True)
        print("labels:", ar_u, "counts:", cnts)
        print("final shapes:", X.shape, labels.shape)

        outdir = os.path.join('.', 'data', dataset_name)
        os.makedirs(outdir, exist_ok=True)
        np.save(os.path.join(outdir, 'X.npy'), X)
        np.save(os.path.join(outdir, 'labels.npy'), labels)
        meta.to_csv(os.path.join(outdir, 'meta.csv'), index=False)
        print('done!')

    else:
        if isinstance(paradigm, (MotorImagery, P300)):
            epochs, labels, meta = paradigm.get_data(
                dataset=dataset,
                subjects=[dataset.subject_list[0]],
                return_epochs=True
            )
            return epochs.info


if __name__ == '__main__':
    datasets = ['CustomEpoch']
    for ds in datasets:
        dataset_to_file(ds, data_save=True)




    '''
    BNCI2014001
    <Info | 8 non-empty values
     bads: []
     ch_names: 'Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P1', 'Pz', 'P2', 'POz'
     chs: 22 EEG
     custom_ref_applied: False
     dig: 25 items (3 Cardinal, 22 EEG)
     highpass: 8.0 Hz
     lowpass: 32.0 Hz
     meas_date: unspecified
     nchan: 22
     projs: []
     sfreq: 250.0 Hz
    >

    BNCI2014002
    <Info | 7 non-empty values
     bads: []
     ch_names: 'EEG1', 'EEG2', 'EEG3', 'EEG4', 'EEG5', 'EEG6', 'EEG7', 'EEG8', 'EEG9', 'EEG10', 'EEG11', 'EEG12', 'EEG13', 'EEG14', 'EEG15'
     chs: 15 EEG
     custom_ref_applied: False
     highpass: 8.0 Hz
     lowpass: 32.0 Hz
     meas_date: unspecified
     nchan: 15
     projs: []
     sfreq: 512.0 Hz
    >

    BNCI2015001
    <Info | 8 non-empty values
     bads: []
     ch_names: 'FC3', 'FCz', 'FC4', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP3', 'CPz', 'CP4'
     chs: 13 EEG
     custom_ref_applied: False
     dig: 16 items (3 Cardinal, 13 EEG)
     highpass: 8.0 Hz
     lowpass: 32.0 Hz
     meas_date: unspecified
     nchan: 13
     projs: []
     sfreq: 512.0 Hz
    >
    '''