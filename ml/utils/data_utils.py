# -*- coding: utf-8 -*-
# @Time    : 2023/01/14
# @Author  : Siyang Li
# @File    : data_utils.py
import numpy as np


def split_data(data, axis, times):
    # Splitting data into multiple sections. data: (trials, channels, time_samples)
    data_split = np.split(data, indices_or_sections=times, axis=axis)
    return data_split


def convert_label(labels, axis, threshold):
    # Converting labels to 0 or 1, based on a certain threshold
    label_01 = np.where(labels > threshold, 1, 0)
    #print(label_01)
    return label_01


def time_cut(data, cut_percentage):
    # Time Cutting: cut at a certain percentage of the time. data: (..., ..., time_samples)
    data = data[:, :, :int(data.shape[2] * cut_percentage)]
    return data


def traintest_split_cross_subject(X, y, subject_counts, test_subject_id):
    """
    Leave‐one‐subject‐out split using actual per‐subject trial counts.
    """
    offsets = np.cumsum(subject_counts)[:-1]
    data_subjects = np.split(X, offsets, axis=0)
    labels_subjects = np.split(y, offsets, axis=0)

    subj_data = data_subjects.pop(test_subject_id)
    subj_label = labels_subjects.pop(test_subject_id)
    train_x = np.concatenate(data_subjects, axis=0)
    train_y = np.concatenate(labels_subjects, axis=0)
    return train_x, train_y, subj_data, subj_label

