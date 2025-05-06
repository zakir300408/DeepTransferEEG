import os
import time
import glob
from scipy.io import loadmat
import pandas as pd
import numpy as np
import moabb

from moabb.datasets import BNCI2014001, BNCI2014002, BNCI2015001
from moabb.paradigms import MotorImagery, P300


# Add retry settings
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds


def dataset_to_file(dataset_name, data_save):
    moabb.set_log_level("ERROR")
    if dataset_name == 'BNCI2014001':
        dataset = BNCI2014001()
        paradigm = MotorImagery(n_classes=4)
        # (5184, 22, 1001) (5184,) 250Hz 9subjects * 4classes * (72+72)trials for 2sessions
    elif dataset_name == 'BNCI2014002':
        dataset = BNCI2014002()
        paradigm = MotorImagery(n_classes=2)
        # (2240, 15, 2561) (2240,) 512Hz 14subjects * 2classes * (50+30)trials * 2sessions(not namely separately)
    elif dataset_name == 'BNCI2015001':
        dataset = BNCI2015001()
        paradigm = MotorImagery(n_classes=2)
        # (5600, 13, 2561) (5600,) 512Hz 12subjects * 2 classes * (200 + 200 + (200 for Subj 8/9/10/11)) trials * (2/3)sessions
    elif dataset_name == 'CustomEpoch':
        root_dir = r"E:\Exoskeleton_DL\XK_work\Data_Epoch"
        mat_files = sorted(glob.glob(os.path.join(root_dir, "*", "*.mat")))
        if not mat_files:
            raise ValueError(f"No .mat files found in CustomEpoch path {root_dir}")
        all_X, all_y, meta_rows = [], [], []
        for fn in mat_files:
            mat = loadmat(fn)
            X = mat['MyEpoch']                   # (n_trials, samples, channels)
            X = X.transpose(0, 2, 1)             # to (n_trials, channels, samples)
            y = mat['MyLabel'].flatten()
            all_X.append(X)
            all_y.append(y)
            meta_rows.append({
                'file': os.path.basename(fn),
                'n_trials': X.shape[0]
            })
        X = np.concatenate(all_X, axis=0)
        labels = np.concatenate(all_y, axis=0)
        labels = labels.astype(int)
        labels = labels - labels.min()        # ensure classes start at 0
        meta = pd.DataFrame(meta_rows)
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")

    if data_save:
        print(f'preparing {dataset_name} data...')
        if dataset_name.startswith('BNCI'):
            # retry download/get_data on failure
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    X, labels, meta = paradigm.get_data(
                        dataset=dataset,
                        subjects=dataset.subject_list[:]
                    )
                    break
                except Exception as e:
                    print(f"Attempt {attempt}/{MAX_RETRIES} failed: {e}")
                    if attempt == MAX_RETRIES:
                        raise
                    time.sleep(RETRY_DELAY)
        # display counts for all datasets
        ar_unique, cnts = np.unique(labels, return_counts=True)
        print("labels:", ar_unique)
        print("Counts:", cnts)
        print(X.shape, labels.shape)
        # save to disk
        data_dir = os.path.join('.', 'data', dataset_name)
        os.makedirs(data_dir, exist_ok=True)
        np.save(os.path.join(data_dir, 'X'), X)
        np.save(os.path.join(data_dir, 'labels'), labels)
        meta.to_csv(os.path.join(data_dir, 'meta.csv'), index=False)
        print('done!')
    else:
        if isinstance(paradigm, MotorImagery):
            X, labels, meta = paradigm.get_data(dataset=dataset, subjects=[dataset.subject_list[0]], return_epochs=True)
            return X.info
        elif isinstance(paradigm, P300):
            X, labels, meta = paradigm.get_data(dataset=dataset, subjects=[dataset.subject_list[0]], return_epochs=True)
            return X.info


if __name__ == '__main__':

    datasets = [
        # 'BNCI2014001',
        # 'BNCI2014002',
        # 'BNCI2015001',
        'CustomEpoch'                  # <-- add your folder here
    ]
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