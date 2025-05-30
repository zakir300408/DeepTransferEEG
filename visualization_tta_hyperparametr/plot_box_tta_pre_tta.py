# file: plot_sml_pre_post_modular_larger.py

import json
import re
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from collections import defaultdict

# — CONFIGURATION —
JSON_PATH = (
    r"E:\Exoskeleton_DL\DeepTransferEEG\logs\CustomEpoch"
    r"\mtta8_str1_t1.5_lr0.0001_st1"
    r"\results_mtta8_str1_t1.5_lr0.0001_st1.json"
)
PRE_COLOR  = "#0072B2"   # blue
POST_COLOR = "#D55E00"   # vermillion
SESSION_PLOTS_COLS = 4

# — HELPERS —
def get_session_index(name):
    """Extract first integer from session filename, or inf if none."""
    m = re.search(r"(\d+)", name)
    return int(m.group(1)) if m else float("inf")

def load_data(path):
    with open(path, "r") as f:
        return json.load(f)

def group_sml_by_subject(breakdowns):
    """Subj → list of (sess_name, sml_value)."""
    d = defaultdict(list)
    for subj, sess, _, _, sml in breakdowns:
        d[subj].append((sess, sml))
    return d

def sort_by_session(d):
    """
    From subj→[(sess, val)...],
    return (subjects, [vals_sorted_by_session...]).
    """
    subjects = sorted(d.keys())
    all_vals = []
    for subj in subjects:
        ordered = sorted(d[subj], key=lambda x: get_session_index(x[0]))
        all_vals.append([v for _, v in ordered])
    return subjects, all_vals

# — PLOTTING —
def plot_pre_post_boxplots(subjects, pre_vals, post_vals,
                           global_pre, global_sd_pre,
                           global_post, global_sd_post,
                           pre_color, post_color):
    """Boxplots of Pre vs Post-TTA SML with global μ±σ and per-subject μ±σ."""
    n = len(subjects)
    positions = np.arange(n) * 2 + 1
    width = 0.6

    # subject-level means
    subj_means_pre  = [np.mean(v) for v in pre_vals]
    subj_means_post = [np.mean(v) for v in post_vals]

    # enlarge figure
    fig, ax = plt.subplots(figsize=(16, 8))

    # pre box
    ax.boxplot(
        pre_vals,
        positions=positions - width/2, widths=width,
        patch_artist=True, showmeans=True,
        boxprops=dict(facecolor=pre_color, edgecolor=pre_color, alpha=0.6),
        medianprops=dict(color="black", linewidth=1.5),
        meanprops=dict(marker="D", markerfacecolor=pre_color, markeredgecolor=pre_color, markersize=6),
        whiskerprops=dict(color=pre_color), capprops=dict(color=pre_color),
        flierprops=dict(marker="o", markerfacecolor=pre_color, markeredgecolor=pre_color, markersize=4),
    )
    # post box
    ax.boxplot(
        post_vals,
        positions=positions + width/2, widths=width,
        patch_artist=True, showmeans=True,
        boxprops=dict(facecolor=post_color, edgecolor=post_color, alpha=0.6),
        medianprops=dict(color="black", linewidth=1.5),
        meanprops=dict(marker="D", markerfacecolor=post_color, markeredgecolor=post_color, markersize=6),
        whiskerprops=dict(color=post_color), capprops=dict(color=post_color),
        flierprops=dict(marker="o", markerfacecolor=post_color, markeredgecolor=post_color, markersize=4),
    )

    # global lines with μ±σ in legend
    line_pre = ax.axhline(
        global_pre, color=PRE_COLOR, linestyle="--", linewidth=2,
        label=f"Global Pre-TTA SML: {global_pre:.2f}±{global_sd_pre:.2f}%"
    )
    line_post = ax.axhline(
        global_post, color=POST_COLOR, linestyle="-.", linewidth=2,
        label=f"Global Post-TTA SML: {global_post:.2f}±{global_sd_post:.2f}%"
    )

    # annotate per-subject μ±σ
    for i in range(n):
        m_pre,  s_pre  = subj_means_pre[i],  np.std(pre_vals[i])
        m_post, s_post = subj_means_post[i], np.std(post_vals[i])
        ypos = max(max(pre_vals[i]), max(post_vals[i])) + 1.0
        txt = f"P: {m_pre:.1f}±{s_pre:.1f}%\nT: {m_post:.1f}±{s_post:.1f}%"
        ax.text(positions[i], ypos, txt, ha="center", va="bottom", fontsize=8,
                bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"))

    # legend & labels
    handles = [
        Patch(facecolor=PRE_COLOR, edgecolor=PRE_COLOR, alpha=0.6, label="Pre-TTA (SML)"),
        Patch(facecolor=POST_COLOR, edgecolor=POST_COLOR, alpha=0.6, label="Post-TTA (SML)"),
        Line2D([0], [0], color="black", linewidth=1.5, label="Median"),
        Line2D([0], [0], marker="D", color=PRE_COLOR,   markerfacecolor=PRE_COLOR,   markersize=6, linestyle="None", label="Mean (Pre)"),
        Line2D([0], [0], marker="D", color=POST_COLOR, markerfacecolor=POST_COLOR, markersize=6, linestyle="None", label="Mean (Post)"),
        line_pre, line_post
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False)
    ax.set_xticks(positions)
    ax.set_xticklabels(subjects)
    ax.set_xlabel("Subject")
    ax.set_ylabel("Session-wise Ensemble SML Accuracy (%)")
    ax.set_title("Pre-TTA vs Post-TTA (SML)\nSession-wise Ensemble SML Accuracy per Subject", fontweight="bold")
    ax.grid(axis="y", linestyle=":", linewidth=0.5)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.show()

def plot_session_trends(subjects, pre_dict, post_dict,
                        pre_color, post_color, cols,
                        global_pre, global_sd_pre,
                        global_post, global_sd_post):
    """Per-subject session-wise line plots with mean±sd and global lines."""
    n = len(subjects)
    rows = math.ceil(n / cols)
    # enlarge grid figure
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows), sharey=True)
    axes = axes.flatten()

    for ax, subj in zip(axes, subjects):
        tta = sorted(post_dict[subj], key=lambda x: get_session_index(x[0]))
        pre = sorted(pre_dict[subj], key=lambda x: get_session_index(x[0]))
        idx_tta = [get_session_index(s) for s, _ in tta]
        idx_pre = [get_session_index(s) for s, _ in pre]
        vals_tta = [v for _, v in tta]
        vals_pre = [v for _, v in pre]

        # session lines
        ax.plot(idx_pre, vals_pre, marker="o", linestyle="-", label="Pre-TTA")
        ax.plot(idx_tta, vals_tta, marker="o", linestyle="--", label="Post-TTA")

        # global horizontal lines
        ax.axhline(global_pre,  color=pre_color,  linestyle="--", linewidth=1)
        ax.axhline(global_post, color=post_color, linestyle="-.", linewidth=1)

        # mean±sd annotation
        m_pre, sd_pre   = np.mean(vals_pre),  np.std(vals_pre)
        m_post, sd_post = np.mean(vals_tta), np.std(vals_tta)
        txt = f"Pre: {m_pre:.2f}±{sd_pre:.2f}%\nPost: {m_post:.2f}±{sd_post:.2f}%"
        ax.text(0.05, 0.95, txt, transform=ax.transAxes,
                va="top", ha="left", fontsize=8,
                bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"))

        ax.set_title(subj)
        ax.set_xlabel("Session #")
        if (axes.tolist().index(ax) % cols) == 0:
            ax.set_ylabel("SML Accuracy (%)")
        ax.grid(True, linestyle=":", linewidth=0.5)

    # remove empty axes
    for ax in axes[n:]:
        fig.delaxes(ax)

    # shared legend
    lines = [
        Line2D([], [], color=PRE_COLOR,   marker="o", linestyle="-",  label="Pre-TTA"),
        Line2D([], [], color=POST_COLOR, marker="o", linestyle="--", label="Post-TTA"),
        Line2D([], [], color=PRE_COLOR,   linestyle="--",      label=f"Global Pre: {global_pre:.2f}±{global_sd_pre:.2f}%"),
        Line2D([], [], color=POST_COLOR, linestyle="-.",      label=f"Global Post: {global_post:.2f}±{global_sd_post:.2f}%")
    ]
    fig.suptitle("Session-wise SML Accuracy per Subject", fontweight="bold")
    fig.legend(
        handles=lines,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.90),
        ncol=4
    )
    plt.tight_layout(rect=[0, 0, 1, 0.88])
    plt.show()

# — MAIN EXECUTION —
def main():
    data = load_data(JSON_PATH)
    tta_dict = group_sml_by_subject(data["session_tta_breakdowns"])
    pre_dict = group_sml_by_subject(data["session_pre_tta_breakdowns"])
    subjects, post_vals = sort_by_session(tta_dict)
    _,       pre_vals  = sort_by_session(pre_dict)

    subj_means_pre  = [np.mean(v) for v in pre_vals]
    subj_means_post = [np.mean(v) for v in post_vals]
    global_pre      = np.mean(subj_means_pre)
    global_post     = np.mean(subj_means_post)
    global_sd_pre   = np.std(subj_means_pre)
    global_sd_post  = np.std(subj_means_post)

    plot_pre_post_boxplots(
        subjects, pre_vals, post_vals,
        global_pre, global_sd_pre,
        global_post, global_sd_post,
        PRE_COLOR, POST_COLOR
    )
    plot_session_trends(
        subjects, pre_dict, tta_dict,
        PRE_COLOR, POST_COLOR,
        SESSION_PLOTS_COLS,
        global_pre, global_sd_pre,
        global_post, global_sd_post
    )

if __name__ == "__main__":
    main()
