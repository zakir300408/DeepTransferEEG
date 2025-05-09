# -*- coding: utf-8 -*-
# @Time    : 2023/7/11
# @Author  : Siyang Li
# @File    : dataloader.py
import numpy as np
import pandas as pd
from sklearn import preprocessing
from scipy.signal import butter, filtfilt
from utils.data_utils import traintest_split_cross_subject, traintest_split_domain_classifier, traintest_split_multisource, traintest_split_domain_classifier_pretest, traintest_split_multisource


def data_process(dataset):
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
        # load concatenated epochs and labels (already first-session only)
        X = np.load('./data/CustomEpoch/X.npy')
        y = np.load('./data/CustomEpoch/labels.npy')
        print('CustomEpoch data:', X.shape, y.shape)
        # count subjects from meta.csv
        meta = pd.read_csv('./data/CustomEpoch/meta.csv')
        num_subjects = len(meta)
        paradigm     = 'MI'
        sample_rate  = 200
        ch_num       = X.shape[1]
        # apply 8–32 Hz bandpass
        nyq = sample_rate / 2
        b, a = butter(5, [8/nyq, 32/nyq], btype='band')
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                X[i, j, :] = filtfilt(b, a, X[i, j, :])
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
        # dynamic count from meta.csv
        meta = pd.read_csv('./data/CustomEpoch/meta.csv')
        num_subjects = len(meta)       # number of sessions
        paradigm     = 'MI'
        sample_rate  = 200             # Hz, as set in dnn.py
        ch_num       = X.shape[1]      # channels
        # apply 8–32 Hz bandpass
        nyq = sample_rate / 2
        b, a = butter(5, [8/nyq, 32/nyq], btype='band')
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                X[i, j, :] = filtfilt(b, a, X[i, j, :])
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
