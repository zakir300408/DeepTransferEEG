# file: plot_box_tta_pre_tta.py

import json
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from collections import defaultdict

# — helper to extract numeric session index from filename —
def get_session_index(session_name):
    """
    Pull the first integer you find in the session_name.
    Tweak the regex if your filenames differ.
    """
    m = re.search(r'(\d+)', session_name)
    return int(m.group(1)) if m else float('inf')

# — load your updated JSON —
JSON_PATH = r'E:\Exoskeleton_DL\DeepTransferEEG\logs\CustomEpoch\mtta8_str1_t1.5_lr0.0001_st1\results_mtta8_str1_t1.5_lr0.0001_st1.json'
with open(JSON_PATH, 'r') as f:
    data = json.load(f)

# — group session data by subject —
session_tta = data['session_tta_breakdowns']
session_pre = data['session_pre_tta_breakdowns']

tta_by_subject = defaultdict(list)
pre_by_subject = defaultdict(list)

for subj, sess, mean_acc, med_acc, sml_acc in session_tta:
    tta_by_subject[subj].append((sess, mean_acc, med_acc, sml_acc))

for subj, sess, mean_acc, med_acc, sml_acc in session_pre:
    pre_by_subject[subj].append((sess, mean_acc, med_acc, sml_acc))

# — get a sorted list of subjects —
subjects = sorted(tta_by_subject.keys())

# — for each subject, sort its sessions by numeric index and collect the mean‐prob accuracies —
session_vals_post = []
session_vals_pre  = []
subject_means_post = []
subject_means_pre  = []

for subj in subjects:
    # sort sessions by the number in their filename
    ordered_tta = sorted(tta_by_subject[subj], key=lambda x: get_session_index(x[0]))
    ordered_pre = sorted(pre_by_subject[subj], key=lambda x: get_session_index(x[0]))

    # pull out the mean-prob ensemble accuracy (first metric in each tuple)
    vals_post = [mean_acc for _, mean_acc, _, _ in ordered_tta]
    vals_pre  = [mean_acc for _, mean_acc, _, _ in ordered_pre]

    session_vals_post.append(vals_post)
    session_vals_pre.append(vals_pre)
    subject_means_post.append(np.mean(vals_post))
    subject_means_pre.append(np.mean(vals_pre))

# — compute overall means —
overall_mean_post = np.mean(subject_means_post)
overall_mean_pre  = np.mean(subject_means_pre)

# — colors —
pre_color  = "#0072B2"   # blue
post_color = "#D55E00"   # vermillion

# — plot 1: Post-TTA only —
fig, ax = plt.subplots(figsize=(14, 6))
n = len(subjects)
positions = np.arange(n) * 2 + 1
width = 0.6

bp_post = ax.boxplot(
    session_vals_post,
    positions=positions,
    widths=width,
    patch_artist=True,
    showmeans=True,
    boxprops=dict(facecolor=post_color, edgecolor=post_color, alpha=0.6),
    medianprops=dict(color='black', linewidth=1.5),
    meanprops=dict(marker='D', markerfacecolor=post_color, markeredgecolor=post_color, markersize=6),
    whiskerprops=dict(color=post_color),
    capprops=dict(color=post_color),
    flierprops=dict(marker='o', markerfacecolor=post_color, markeredgecolor=post_color, markersize=4)
)

line_post = ax.axhline(
    overall_mean_post,
    color=post_color, linestyle='-.', linewidth=2,
    label=f'Overall Mean Post-TTA: {overall_mean_post:.2f}%'
)

handles = [
    Patch(facecolor=post_color, edgecolor=post_color, alpha=0.6, label='Post-TTA'),
    Line2D([0], [0], color='black', linewidth=1.5, label='Median'),
    Line2D([0], [0], marker='D', color=post_color, markerfacecolor=post_color, markersize=6, linestyle='None', label='Mean'),
    line_post
]

ax.legend(
    handles=handles,
    loc='upper left',
    bbox_to_anchor=(1.02, 1),
    frameon=False
)

ax.set_xlabel('Subject', fontsize=12)
ax.set_ylabel('Session-wise Ensemble Accuracy (%)', fontsize=12)
ax.set_title('Post-TTA\nSession-wise Ensemble Accuracy per Subject',
             fontsize=14, fontweight='bold')
ax.set_xticks(positions)
ax.set_xticklabels(subjects, fontsize=11)
ax.grid(axis='y', linestyle=':', linewidth=0.5)

plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.show()


# — plot 2: Pre-TTA vs Post-TTA side by side —
fig, ax = plt.subplots(figsize=(14, 6))

# offset pre and post so they sit next to each other
offset = width / 2
pos_pre  = positions - offset
pos_post = positions + offset

bp_pre = ax.boxplot(
    session_vals_pre,
    positions=pos_pre,
    widths=width,
    patch_artist=True,
    showmeans=True,
    boxprops=dict(facecolor=pre_color, edgecolor=pre_color, alpha=0.6),
    medianprops=dict(color='black', linewidth=1.5),
    meanprops=dict(marker='D', markerfacecolor=pre_color, markeredgecolor=pre_color, markersize=6),
    whiskerprops=dict(color=pre_color),
    capprops=dict(color=pre_color),
    flierprops=dict(marker='o', markerfacecolor=pre_color, markeredgecolor=pre_color, markersize=4)
)

bp_post = ax.boxplot(
    session_vals_post,
    positions=pos_post,
    widths=width,
    patch_artist=True,
    showmeans=True,
    boxprops=dict(facecolor=post_color, edgecolor=post_color, alpha=0.6),
    medianprops=dict(color='black', linewidth=1.5),
    meanprops=dict(marker='D', markerfacecolor=post_color, markeredgecolor=post_color, markersize=6),
    whiskerprops=dict(color=post_color),
    capprops=dict(color=post_color),
    flierprops=dict(marker='o', markerfacecolor=post_color, markeredgecolor=post_color, markersize=4)
)

line_pre = ax.axhline(
    overall_mean_pre,
    color=pre_color, linestyle='--', linewidth=2,
    label=f'Overall Mean Pre-TTA: {overall_mean_pre:.2f}%'
)
line_post = ax.axhline(
    overall_mean_post,
    color=post_color, linestyle='-.', linewidth=2,
    label=f'Overall Mean Post-TTA: {overall_mean_post:.2f}%'
)

handles = [
    Patch(facecolor=pre_color, edgecolor=pre_color, alpha=0.6, label='Pre-TTA'),
    Patch(facecolor=post_color, edgecolor=post_color, alpha=0.6, label='Post-TTA'),
    Line2D([0], [0], color='black', linewidth=1.5, label='Median'),
    Line2D([0], [0], marker='D', color=pre_color, markerfacecolor=pre_color, markersize=6, linestyle='None', label='Mean (Pre)'),
    Line2D([0], [0], marker='D', color=post_color, markerfacecolor=post_color, markersize=6, linestyle='None', label='Mean (Post)'),
    line_pre,
    line_post
]

ax.legend(
    handles=handles,
    loc='upper left',
    bbox_to_anchor=(1.02, 1),
    frameon=False
)

ax.set_xlabel('Subject', fontsize=12)
ax.set_ylabel('Session-wise Ensemble Accuracy (%)', fontsize=12)
ax.set_title('Pre-TTA vs Post-TTA\nSession-wise Ensemble Accuracy per Subject',
             fontsize=14, fontweight='bold')
ax.set_xticks(positions)
ax.set_xticklabels(subjects, fontsize=11)
ax.grid(axis='y', linestyle=':', linewidth=0.5)

plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.show()
