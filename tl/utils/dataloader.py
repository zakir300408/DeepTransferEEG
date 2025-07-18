# -*- coding: utf-8 -*-
# @Time    : 2023/7/11
# @Author  : Siyang Li
# @File    : dataloader.py
import numpy as np
import pandas as pd
from sklearn import preprocessing
from scipy.signal import butter, filtfilt, iirnotch, spectrogram
from joblib import Parallel, delayed
from utils.data_utils import traintest_split_cross_subject, traintest_split_domain_classifier, \
                              traintest_split_multisource, traintest_split_domain_classifier_pretest

def data_process(dataset):
    '''
    :param dataset: str, dataset name
    :return: X, y, num_subjects, paradigm, sample_rate, ch_num
    '''
    # --- load data ---
    if dataset == 'BNCI2014001-4':
        X = np.load('./data/BNCI2014001/X.npy')
        y = np.load('./data/BNCI2014001/labels.npy')
    else:
        X = np.load(f'./data/{dataset}/X.npy')
        y = np.load(f'./data/{dataset}/labels.npy')
    print('raw data:', X.shape, y.shape)

    num_subjects = paradigm = sample_rate = ch_num = None

    # --- CustomEpoch branch with full pipeline ---
    if dataset == 'CustomEpoch':
        # load concatenated epochs and labels (already first-session only)
        X = np.load('./data/CustomEpoch/X.npy')
        y = np.load('./data/CustomEpoch/labels.npy')
        print('CustomEpoch data:', X.shape, y.shape)

        # metadata
        meta = pd.read_csv('./data/CustomEpoch/meta.csv')
        num_subjects = len(meta)
        paradigm     = 'MI'
        orig_sr      = 200       # original sampling rate
        sample_rate  = 100       # target downsample rate
        ch_num       = X.shape[1]

        # 1) 50 Hz notch at original rate (skip initial 8–32 Hz bandpass)
        nyq = orig_sr / 2
        bn, an = iirnotch(50.0/nyq, 30.0)
        X = filtfilt(bn, an, X, axis=2)
        # 2) Downsample to 100 Hz
        X = X[:, :, ::2]

        # 3) Multi‐band spectral fusion (θ, α, β, full)
        nyq = sample_rate / 2
        bands = [
            (4.0, 7.0),    # theta
            (7.0, 13.0),   # alpha
            (13.0, 32.0),  # beta
            (4.0, 40.0)    # full
        ]
        X_sum = np.zeros_like(X)
        for low, high in bands:
            b, a = butter(5, [low/nyq, high/nyq], btype='band')
            Xf   = filtfilt(b, a, X, axis=2)
            X_sum += Xf
        X_fused = X_sum / len(bands)

        # 4) Time‐frequency (spectrogram) features
        nperseg, noverlap = 128, 64
        # get dimensions from one channel
        _, _, S0 = spectrogram(X_fused[0, 0], fs=sample_rate,
                               nperseg=nperseg, noverlap=noverlap)
        freq_bins, time_bins = S0.shape

        flat_X = X_fused.reshape(-1, X_fused.shape[2])
        def _compute_sxx(x):
            return spectrogram(x, fs=sample_rate,
                               nperseg=nperseg, noverlap=noverlap)[2]

        sxx_list = Parallel(n_jobs=-1)(
            delayed(_compute_sxx)(flat_X[i])
            for i in range(flat_X.shape[0])
        )
        tf_feats = (np.stack(sxx_list)
                      .reshape(X_fused.shape[0],
                               X_fused.shape[1],
                               freq_bins,
                               time_bins))
        tf_flat = tf_feats.reshape(X_fused.shape[0],
                                   X_fused.shape[1],
                                   -1)

        # 5) Concatenate fused spectral + TF features along time axis
        #    X_fused: (trials, ch, times)
        #    tf_flat:  (trials, ch, freq_bins*time_bins)
        X = np.concatenate([X_fused, tf_flat], axis=2)

        # 6) Label‐encode and normalize per channel
        y = preprocessing.LabelEncoder().fit_transform(y)
        med = np.median(X, axis=2, keepdims=True)
        std = np.std(   X, axis=2, keepdims=True) + 1e-8
        X   = (X - med) / std

        print('processed data:', X.shape, 'labels:', y.shape)
        return X, y, num_subjects, paradigm, sample_rate, ch_num

    if dataset == 'BNCI2014001':
        paradigm = 'MI'
        num_subjects = 9
        sample_rate = 250
        ch_num = 22

        # only use session T, remove session E
        indices = []
        for i in range(num_subjects):
            indices.append(np.arange(288) + (576 * i))
        indices = np.concatenate(indices, axis=0)
        X = X[indices]
        y = y[indices]

        # only use two classes [left_hand, right_hand]
        indices = []
        for i in range(len(y)):
            if y[i] in ['left_hand', 'right_hand']:
                indices.append(i)
        X = X[indices]
        y = y[indices]
    elif dataset == 'BNCI2014002':
        paradigm = 'MI'
        num_subjects = 14
        sample_rate = 512
        ch_num = 15

        # only use session train, remove session test
        indices = []
        for i in range(num_subjects):
            indices.append(np.arange(100) + (160 * i))
        indices = np.concatenate(indices, axis=0)
        X = X[indices]
        y = y[indices]

    elif dataset == 'BNCI2015001':
        paradigm = 'MI'
        num_subjects = 12
        sample_rate = 512
        ch_num = 13

        # only use session 1, remove session 2/3
        indices = []
        for i in range(num_subjects):
            if i in [7, 8, 9, 10]:
                indices.append(np.arange(200) + (400 * 7) + 600 * (i - 7))
            elif i == 11:
                indices.append(np.arange(200) + (400 * 7) + 600 * (i - 7))
            else:
                indices.append(np.arange(200) + (400 * i))

        indices = np.concatenate(indices, axis=0)
        X = X[indices]
        y = y[indices]
    elif dataset == 'BNCI2014001-4':
        paradigm = 'MI'
        num_subjects = 9
        sample_rate = 250
        ch_num = 22

        # only use session T, remove session E
        indices = []
        for i in range(num_subjects):
            indices.append(np.arange(288) + (576 * i))
        indices = np.concatenate(indices, axis=0)
        X = X[indices]
        y = y[indices]

    # after all branches
    le = preprocessing.LabelEncoder()
    y = le.fit_transform(y)
    # normalize each channel of each trial over time
    X = (X - X.mean(axis=2, keepdims=True)) / (X.std(axis=2, keepdims=True) + 1e-8)
    print('data shape:', X.shape, ' labels shape:', y.shape)
    return X, y, num_subjects, paradigm, sample_rate, ch_num


def data_process_secondsession(dataset):
    '''

    :param dataset: str, dataset name
    :return: X, y, num_subjects, paradigm, sample_rate
    '''

    if dataset == 'BNCI2014001-4':
        X = np.load('./data/' + 'BNCI2014001' + '/X.npy')
        y = np.load('./data/' + 'BNCI2014001' + '/labels.npy')
    else:
        X = np.load('./data/' + dataset + '/X.npy')
        y = np.load('./data/' + dataset + '/labels.npy')
    print(X.shape, y.shape)

    num_subjects, paradigm, sample_rate = None, None, None
    ch_num = None

    # Custom dataset
    if dataset == 'CustomEpoch':
        # load concatenated epochs and labels
        X = np.load('./data/CustomEpoch/X.npy')
        y = np.load('./data/CustomEpoch/labels.npy')
        print('CustomEpoch data:', X.shape, y.shape)
        meta = pd.read_csv('./data/CustomEpoch/meta.csv')
        num_subjects = len(meta)
        paradigm     = 'MI'
        orig_sr      = 200        # original sampling rate
        sample_rate  = 100        # target downsample rate
        ch_num       = X.shape[1]

        # apply 50 Hz notch at original rate (skip initial 8–32 Hz bandpass)
        nyq = orig_sr / 2
        bn, an = iirnotch(50.0/nyq, 30.0)
        X = filtfilt(bn, an, X, axis=2)
        # downsample to 100 Hz
        X = X[:, :, ::2]

        # ========== NEW: compute time-frequency features in parallel ==========
        nperseg, noverlap = 128, 64
        _, _, S0 = spectrogram(X[0,0], fs=sample_rate, nperseg=nperseg, noverlap=noverlap)
        freq_bins, time_bins = S0.shape
        flat_X = X.reshape(-1, X.shape[2])
        def _compute_sxx(x):
            return spectrogram(x, fs=sample_rate, nperseg=nperseg, noverlap=noverlap)[2]
        sxx_list = Parallel(n_jobs=-1)(delayed(_compute_sxx)(flat_X[k])
                                       for k in range(flat_X.shape[0]))
        tf_feats = np.stack(sxx_list).reshape(X.shape[0], X.shape[1], freq_bins, time_bins)
        tf_flat = tf_feats.reshape(X.shape[0], X.shape[1], -1)
        X = np.concatenate([X, tf_flat], axis=2)
        # ===========================================================

        # skip other branches
        y = preprocessing.LabelEncoder().fit_transform(y)
        # normalize each channel of each trial over time
        X = (X - X.mean(axis=2, keepdims=True)) / (X.std(axis=2, keepdims=True) + 1e-8)
        print('data shape:', X.shape, ' labels shape:', y.shape)
        return X, y, num_subjects, paradigm, sample_rate, ch_num

    if dataset == 'BNCI2014001':
        paradigm = 'MI'
        num_subjects = 9
        sample_rate = 250
        ch_num = 22

        # only use session T, remove session E
        indices = []
        for i in range(num_subjects):
            indices.append(np.arange(288) + (576 * i) + 288) # use second sessions
        indices = np.concatenate(indices, axis=0)
        X = X[indices]
        y = y[indices]

        # only use two classes [left_hand, right_hand]
        indices = []
        for i in range(len(y)):
            if y[i] in ['left_hand', 'right_hand']:
                indices.append(i)
        X = X[indices]
        y = y[indices]
    elif dataset == 'BNCI2014002':
        paradigm = 'MI'
        num_subjects = 14
        sample_rate = 512
        ch_num = 15

        # only use session train, remove session test
        indices = []
        for i in range(num_subjects):
            #indices.append(np.arange(100) + (160 * i))
            indices.append(np.arange(60) + (160 * i) + 100) # use second sessions
        indices = np.concatenate(indices, axis=0)
        X = X[indices]
        y = y[indices]

    elif dataset == 'BNCI2015001':
        paradigm = 'MI'
        num_subjects = 12
        sample_rate = 512
        ch_num = 13

        # only use session 1, remove session 2/3
        indices = []
        for i in range(num_subjects):
            # use second sessions
            if i in [7, 8, 9, 10]:
                indices.append(np.arange(200) + (400 * 7) + 600 * (i - 7))
            elif i == 11:
                indices.append(np.arange(200) + (400 * 7) + 600 * (i - 7))
            else:
                indices.append(np.arange(200) + (400 * i))

        indices = np.concatenate(indices, axis=0)
        X = X[indices]
        y = y[indices]
    elif dataset == 'BNCI2014001-4':
        paradigm = 'MI'
        num_subjects = 9
        sample_rate = 250
        ch_num = 22

        # only use session T, remove session E
        indices = []
        for i in range(num_subjects):
            indices.append(np.arange(288) + (576 * i))
        indices = np.concatenate(indices, axis=0)
        X = X[indices]
        y = y[indices]

    # after all branches
    le = preprocessing.LabelEncoder()
    y = le.fit_transform(y)
    # normalize each channel of each trial over time
    X = (X - X.mean(axis=2, keepdims=True)) / (X.std(axis=2, keepdims=True) + 1e-8)
    print('data shape:', X.shape, ' labels shape:', y.shape)
    return X, y, num_subjects, paradigm, sample_rate, ch_num


def read_mi_combine_tar(args):
    # load full data
    if 'ontinual' in args.method:
        X, y, num_subjects, paradigm, sample_rate, ch_num = data_process_secondsession(args.data)
    else:
        X, y, num_subjects, paradigm, sample_rate, ch_num = data_process(args.data)

    # special handling for CustomEpoch: treat each .mat (row in meta.csv) as one subject
    if args.data == 'CustomEpoch':
        meta = pd.read_csv('./data/CustomEpoch/meta.csv')
        counts = meta['n_trials'].values
        starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
        ends   = np.cumsum(counts)
        idts = args.idt if isinstance(args.idt, (list, tuple)) else [args.idt]
        # gather all target sessions
        tar_data  = np.concatenate([X[starts[i]:ends[i]] for i in idts], axis=0)
        tar_label = np.concatenate([y[starts[i]:ends[i]] for i in idts], axis=0)
        # the rest become source
        src_idxs = [i for i in range(len(counts)) if i not in idts]
        src_data  = np.concatenate([X[starts[i]:ends[i]] for i in src_idxs], axis=0)
        src_label = np.concatenate([y[starts[i]:ends[i]] for i in src_idxs], axis=0)
        return src_data, src_label, tar_data, tar_label

    # default cross‐subject split
    src_data, src_label, tar_data, tar_label = traintest_split_cross_subject(
        args.data, X, y, num_subjects, args.idt
    )
    return src_data, src_label, tar_data, tar_label


def read_mi_combine_domain(args):

    X, y, num_subjects, paradigm, sample_rate, ch_num = data_process(args.data)

    src_data, src_label, tar_data, tar_label = traintest_split_domain_classifier(args.data, X, y, num_subjects, args.idt)

    return src_data, src_label, tar_data, tar_label


def read_mi_combine_domain_split(args):

    X, y, num_subjects, paradigm, sample_rate, ch_num = data_process(args.data)

    src_data, src_label, tar_data, tar_label = traintest_split_domain_classifier_pretest(args.data, X, y, num_subjects, args.ratio)

    return src_data, src_label, tar_data, tar_label


def read_mi_multi_source(args):
    X, y, num_subjects, paradigm, sample_rate, ch_num = data_process(args.data)

    src_data, src_label, tar_data, tar_label = traintest_split_multisource(args.data, X, y, num_subjects, args.idt)

    return src_data, src_label, tar_data, tar_label


def data_normalize(fea_de, norm_type):
    if norm_type == 'zscore':
        zscore = preprocessing.StandardScaler()
        fea_de = zscore.fit_transform(fea_de)

    return fea_de
def data_normalize(fea_de, norm_type):
    if norm_type == 'zscore':
        zscore = preprocessing.StandardScaler()
        fea_de = zscore.fit_transform(fea_de)

    return fea_de
