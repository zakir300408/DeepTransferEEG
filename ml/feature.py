# -*- coding: utf-8 -*-
# @Time    : 2023/07/07
# @Author  : Siyang Li
# @File    : feature.py
import mne
import numpy as np
import torch
from sklearn import preprocessing
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

import random
import sys
import os
import importlib.util

# load load_subject_data from tl.py
tl_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'tl.py')
spec = importlib.util.spec_from_file_location('tl_file', tl_path)
tl_file = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tl_file)
load_subject_data = tl_file.load_subject_data

from utils.alg_utils import EA
from utils.data_utils import traintest_split_cross_subject


def apply_zscore(train_x, test_x, num_subjects):
    # train split into subjects
    train_z = []
    trial_num = int(train_x.shape[0] / (num_subjects - 1))
    for j in range(num_subjects - 1):
        scaler = preprocessing.StandardScaler()
        train_x_tmp = scaler.fit_transform(train_x[trial_num * j: trial_num * (j + 1), :])
        train_z.append(train_x_tmp)
    train_x = np.concatenate(train_z, axis=0)
    # test subject
    scaler = preprocessing.StandardScaler()
    test_x = scaler.fit_transform(test_x)
    return train_x, test_x


def data_process(dataset):
    """
    :param dataset: str, full path to root folder of your MAT datasets
    :return: X, y, num_subjects, paradigm, sample_rate, ch_num
    """
    mne.set_log_level('warning')
    root_dir = dataset
    print(f"[DATA_LOAD] Root directory: {root_dir}")
    # list all subject subfolders
    subjects = [d for d in os.listdir(root_dir)
                if os.path.isdir(os.path.join(root_dir, d))]
    print(f"[DATA_LOAD] Found {len(subjects)} subject(s): {subjects!r}")
    X_list, y_list = [], []
    subject_counts = []
    for s in subjects:
        print(f"[DATA_LOAD] Loading subject '{s}'...")
        trials, labels = load_subject_data(os.path.join(root_dir, s), Fs=200)
        print(f"[DATA_LOAD]  -> {len(trials)} trial(s), {len(labels)} label(s) loaded")
        for tr in trials:
            X_list.append(tr)           # tr: channels x samples
        y_list.extend(labels.tolist()) # labels as array
        subject_counts.append(len(trials))
    X = np.stack(X_list, axis=0)      # shape: [n_trials, channels, samples]
    print(f"[DATA_LOAD] Total assembled trials: {X.shape[0]}")
    y = np.array(y_list)
    num_subjects = len(subjects)
    ch_num = X.shape[1]
    sample_rate = 200
    paradigm = 'MI'
    print(f"[DATA_LOAD] Completed: {num_subjects} subject(s), {X.shape[0]} trials, "
          f"{ch_num} channels @ {sample_rate}Hz")
    # also return subject names
    return X, y, num_subjects, paradigm, sample_rate, ch_num, subject_counts, subjects


def data_alignment(X, num_subjects):
    '''
    :param X: np array, EEG data
    :param num_subjects: int, number of total subjects in X
    :return: np array, aligned EEG data
    '''
    # subject-wise EA
    print('before EA:', X.shape)
    out = []
    for i in range(num_subjects):
        tmp_x = EA(X[X.shape[0] // num_subjects * i:X.shape[0] // num_subjects * (i + 1), :, :])
        out.append(tmp_x)
    X = np.concatenate(out, axis=0)
    print('after EA:', X.shape)
    return X


def traintest_split_within_subject(dataset, X, y, num_subjects, test_subject_id, num, shuffle):
    data_subjects = np.split(X, indices_or_sections=num_subjects, axis=0)
    labels_subjects = np.split(y, indices_or_sections=num_subjects, axis=0)
    subj_data = data_subjects.pop(test_subject_id)
    subj_label = labels_subjects.pop(test_subject_id)
    class_out = len(np.unique(subj_label))
    if shuffle:
        inds = np.arange(len(subj_data))
        np.random.shuffle(inds)
        subj_data = subj_data[inds]
        subj_label = subj_label[inds]
    if num < 1:  # percentage
        num_int = int(len(subj_data) * num / class_out)
    else:  # numbers
        num_int = int(num)

    inds_all_train = []
    inds_all_test = []
    for class_num in range(class_out):
        inds_class = np.where(subj_label == class_num)[0]
        inds_all_train.append(inds_class[:num_int])
        inds_all_test.append(inds_class[num_int:])
    inds_all_train = np.concatenate(inds_all_train)
    inds_all_test = np.concatenate(inds_all_test)

    train_x = subj_data[inds_all_train]
    train_y = subj_label[inds_all_train]
    test_x = subj_data[inds_all_test]
    test_y = subj_label[inds_all_test]

    print('Within subject s' + str(test_subject_id))
    print('Training/Test split:', train_x.shape, test_x.shape)
    return train_x, train_y, test_x, test_y


def ml_classifier(approach, output_probability, train_x, train_y, test_x, return_model=None, weight=None):
    if approach == 'LDA':
        clf = LinearDiscriminantAnalysis()
    elif approach == 'LR':
        clf = LogisticRegression(max_iter=1000)
    elif approach == 'AdaBoost':
        clf = AdaBoostClassifier()
    elif approach == 'GradientBoosting':
        clf = GradientBoostingClassifier()
    elif approach == 'xgb':
        clf = XGBClassifier()
        if weight:
            print('XGB weight:', weight)
            clf = XGBClassifier(scale_pos_weight=weight)
            # clf = imb_xgb(special_objective='focal', focal_gamma=2.0)
    # clf = LinearDiscriminantAnalysis()
    # clf = SVC()
    # clf = LinearSVC()
    # clf = KNeighborsClassifier()
    clf.fit(train_x, train_y)

    if output_probability:
        pred = clf.predict_proba(test_x)
    else:
        pred = clf.predict(test_x)
    if return_model:
        return pred, clf
    else:
        print(pred)
        return pred


def ml_cross(dataset, info, align, approach):
    X, y, num_subjects, paradigm, sample_rate, ch_num, subject_counts, subjects = data_process(dataset)
    print('X, y, num_subjects, paradigm, sample_rate:', X.shape, y.shape, num_subjects, paradigm, sample_rate)

    if align:
        X = data_alignment(X, num_subjects)
        y = y[: X.shape[0]]

    scores_arr = []
    for i in range(num_subjects):
        # verbose subject name
        print(f"[ML_CROSS] Leave‐one‐out: training on all except '{subjects[i]}', testing on '{subjects[i]}'")
        train_x, train_y, test_x, test_y = traintest_split_cross_subject(X, y, subject_counts, i)
        print('train_x, train_y, test_x, test_y.shape', train_x.shape, train_y.shape, test_x.shape, test_y.shape)

        if paradigm == 'MI':
            # CSP
            csp = mne.decoding.CSP(n_components=6)
            train_x_csp = csp.fit_transform(train_x, train_y)
            test_x_csp = csp.transform(test_x)

            # classifier
            pred, model = ml_classifier(approach, False, train_x_csp, train_y, test_x_csp, return_model=True)
            score = np.round(accuracy_score(test_y, pred), 5)
            print('score', np.round(score, 5))

        scores_arr.append(score)
    print('#' * 30)
    for i in range(len(scores_arr)):
        scores_arr[i] = np.round(scores_arr[i] * 100, 5)
    print('sbj scores', scores_arr)
    print('avg', np.round(np.average(scores_arr), 5))

    return scores_arr


def ml_within(dataset, info, align, approach, cuda_device_id):
    X, y, num_subjects, paradigm, sample_rate, ch_num, subject_counts, subjects = data_process(dataset)
    print('X, y, num_subjects, paradigm, sample_rate:', X.shape, y.shape, num_subjects, paradigm, sample_rate)

    print('sample rate:', sample_rate)

    if align:
        X = data_alignment(X, num_subjects)

    scores_arr = []

    for i in range(num_subjects):
        train_x, train_y, test_x, test_y = traintest_split_within_subject(dataset, X, y, num_subjects, i, 0.5, True)
        print('train_x, train_y, test_x, test_y.shape', train_x.shape, train_y.shape, test_x.shape, test_y.shape)

        if paradigm == 'MI':
            # CSP
            csp = mne.decoding.CSP(n_components=10)
            train_x_csp = csp.fit_transform(train_x, train_y)
            test_x_csp = csp.transform(test_x)

            # classifier
            pred, model = ml_classifier(approach, False, train_x_csp, train_y, test_x_csp, return_model=True)
            score = np.round(accuracy_score(test_y, pred), 5)
            print('score', np.round(score, 5))
        scores_arr.append(score)
    print('#' * 30)
    for i in range(len(scores_arr)):
        scores_arr[i] = np.round(scores_arr[i] * 100)
    print('sbj scores', scores_arr)
    print('avg', np.round(np.average(scores_arr), 5))

    return scores_arr


# toggle Euclidean Alignment on/off
USE_EA = True

if __name__ == '__main__':
    # Example invocation:
    root_dir = r"E:\Exoskeleton_DL\XK_work\Data_Epoch"
    print(f"ML cross‐subject run (Euclidean Alignment = {USE_EA})")
    ml_cross(root_dir, None, align=USE_EA, approach='LDA')
