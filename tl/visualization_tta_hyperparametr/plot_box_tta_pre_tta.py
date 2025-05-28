import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# Load JSON results
JSON_PATH = r'E:\Exoskeleton_DL\DeepTransferEEG\logs\CustomEpoch\mtta8_str1_t1.5_lr0.0001_st1\results_mtta8_str1_t1.5_lr0.0001_st1.json'
with open(JSON_PATH, 'r') as f:
    data = json.load(f)

# Define and sort session order numerically
session_order = [10, 11, 12] + list(range(1, 10))
sorted_indices = sorted(range(len(session_order)), key=lambda i: session_order[i])

# Collect only Post-TTA per subject, skipping duplicates
seen_subjects = set()
subjects = []
session_vals_post = []
subject_means_post = []
for subject, sess_post in data['session_tta_breakdowns']:
    if subject in seen_subjects:
        continue
    seen_subjects.add(subject)
    # reorder sessions numerically
    ordered_post = [sess_post[i] for i in sorted_indices]
    # extract mean-probability ensemble accuracy (index 1)
    vals_post = [s[1] for s in ordered_post]
    subjects.append(subject)
    session_vals_post.append(vals_post)
    subject_means_post.append(np.mean(vals_post))

# Compute overall mean for Post-TTA
overall_mean_post = np.mean(subject_means_post)

# Colors (color-blind–safe)
post_color = "#D55E00"   # vermillion

# Create the plot with increased horizontal spacing
fig, ax = plt.subplots(figsize=(14, 6))
n = len(subjects)
# space subjects by 2 units for clarity
positions = np.arange(n) * 2 + 1
width = 0.6

# Plot only Post-TTA boxplots
bp_post = ax.boxplot(
    session_vals_post,
    positions=positions,  # center
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

# Plot overall Post-TTA mean line
line_post = ax.axhline(
    overall_mean_post,
    color=post_color, linestyle='-.', linewidth=2,
    label=f'Overall Mean Post-TTA: {overall_mean_post:.2f}%'
)

# Construct legend with only Post-TTA
handles = [
    Patch(facecolor=post_color, edgecolor=post_color, alpha=0.6, label='Post-TTA'),
    Line2D([0], [0], color='black', linewidth=1.5, label='Median'),
    Line2D([0], [0], marker='D', color=post_color, markerfacecolor=post_color, markersize=6, linestyle='None', label='Mean'),
    line_post
]

# Place legend outside to the right
ax.legend(
    handles=handles,
    loc='upper left',
    bbox_to_anchor=(1.02, 1),
    borderaxespad=0.,
    frameon=False
)

# Labels, title, ticks, and grid
ax.set_xlabel('Subject', fontsize=12)
ax.set_ylabel('Session-wise Ensemble Accuracy (%)', fontsize=12)
ax.set_title(
    'Post-TTA\nSession-wise Ensemble Accuracy per Subject',
    fontsize=14, fontweight='bold'
)
ax.set_xticks(positions)
ax.set_xticklabels(subjects, fontsize=11)
ax.tick_params(axis='y', labelsize=11)
ax.grid(axis='y', linestyle=':', linewidth=0.5)

plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.show()
# Construct legend handles
handles = [
    Patch(facecolor=pre_color,  edgecolor=pre_color,  alpha=0.6, label='Pre-TTA'),
    Patch(facecolor=post_color, edgecolor=post_color, alpha=0.6, label='Post-TTA'),
    Line2D([0], [0], color='black', linewidth=1.5, label='Median'),
    Line2D([0], [0], marker='D', color=pre_color,  markerfacecolor=pre_color,  markersize=6, linestyle='None', label='Mean (Pre)'),
    Line2D([0], [0], marker='D', color=post_color, markerfacecolor=post_color, markersize=6, linestyle='None', label='Mean (Post)'),
    line_pre,
    line_post
]

# Place legend outside to the right
ax.legend(
    handles=handles,
    loc='upper left',
    bbox_to_anchor=(1.02, 1),
    borderaxespad=0.,
    frameon=False
)

# Labels, title, ticks, and grid
ax.set_xlabel('Subject', fontsize=12)
ax.set_ylabel('Session-wise Ensemble Accuracy (%)', fontsize=12)
ax.set_title(
    'Pre-TTA vs Post-TTA\nSession-wise Ensemble Accuracy per Subject',
    fontsize=14, fontweight='bold'
)
ax.set_xticks(positions)
ax.set_xticklabels(subjects, fontsize=11)
ax.tick_params(axis='y', labelsize=11)
ax.grid(axis='y', linestyle=':', linewidth=0.5)

plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.show()
