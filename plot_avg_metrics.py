import re
import logging
import pandas as pd
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")

# disable matplotlib debug logs (especially font_manager spam)
logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)
# also silence other matplotlib and PIL debug streams
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("matplotlib.backends").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)

log_file = r"e:\Exoskeleton_DL\DeepTransferEEG\classification_results\logs_20250427_183811_TL_run_WU_40_trials\log.txt"

# parse log
subject_metrics = {}
current_subject = None
current_session = None
with open(log_file, "r") as f:
    for line in f:
        m_sub = re.match(r"Subject (\w+) source labels", line)
        if m_sub:
            current_subject = m_sub.group(1)
            subject_metrics[current_subject] = []
            logging.debug(f"Found subject {current_subject}")
            continue
        m_sess = re.match(r"\s*Session (\w+\.mat)", line)
        if m_sess and current_subject:
            current_session = m_sess.group(1)
            logging.debug(f"  Found session {current_session}")
            continue
        m2 = re.search(r"avg transfer:\s*([\d.]+)%\s*avg baseline:\s*([\d.]+)%", line)
        if m2 and current_subject and current_session:
            t = float(m2.group(1))
            b = float(m2.group(2))
            subject_metrics[current_subject].append((current_session, t, b))
            logging.debug(f"    {current_session} → transfer={t}%, baseline={b}%")
            continue

# compute per‐subject averages
avg_metrics = {
    subj: (
        sum(t for _, t, _ in vals) / len(vals),
        sum(b for _, _, b in vals) / len(vals)
    )
    for subj, vals in subject_metrics.items() if vals
}

# print summary table
df_summary = pd.DataFrame([
    {"Subject": subj, "AvgTransfer": tr, "AvgBaseline": bl}
    for subj, (tr, bl) in avg_metrics.items()
])
print("\nOverall subject averages:")
print(df_summary.to_string(index=False))

# print per‑subject detail tables
for subj, sessions in subject_metrics.items():
    if not sessions:
        continue
    df_sub = pd.DataFrame(sessions, columns=["Session", "Transfer", "Baseline"])
    print(f"\nDetails for subject {subj}:")
    print(df_sub.to_string(index=False))

# prepare plotting
subjects = list(avg_metrics.keys())
transfers = [avg_metrics[s][0] for s in subjects]
baselines = [avg_metrics[s][1] for s in subjects]
x = range(len(subjects))

plt.figure(figsize=(12, 6))
plt.bar([i - 0.2 for i in x], transfers, width=0.4, label="Avg Transfer")
plt.bar([i + 0.2 for i in x], baselines, width=0.4, label="Avg Baseline")
plt.xticks(x, subjects)
plt.ylabel("Accuracy (%)")
plt.title("Per‑Subject Average Transfer vs Baseline Accuracy")
plt.legend()
plt.tight_layout()
plt.show()
