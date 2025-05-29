import re
from pathlib import Path

def parse_epoch_logs(base_dir):
    """
    Parse all CustomEpoch log subfolders to extract Ensemble TTA MeanProb for each
    hyperparameter combination.
    """
    results = []
    # match folder names like mtta8_str1_t1.5_lr0.0001_st1
    folder_re = re.compile(
        r"mtta(?P<tta>\d+)_str(?P<stride>\d+)_t(?P<t>[\d\.]+)"
        r"_lr(?P<lr>[\d\.]+)_st(?P<steps>\d+)"
    )

    # look recursively for any folder matching our hyperparam pattern
    for subdir in Path(base_dir).rglob('mtta*_str*_t*_lr*_st*'):
        if not subdir.is_dir():
            continue
        m = folder_re.fullmatch(subdir.name)
        if not m:
            continue

        # parse hyperparameters
        params = {
            "tta":    int(m.group("tta")),
            "stride": int(m.group("stride")),
            "t":      float(m.group("t")),
            "lr":     float(m.group("lr")),
            "steps":  int(m.group("steps"))
        }

        # find the log file
        logs = list(subdir.glob("log_T-TIME_CustomEpoch_*.txt"))
        if not logs:
            print(f"⚠️  No log file found in {subdir}")
            continue
        log_path = logs[0]

        # read lines (adjust encoding if needed)
        with log_path.open("r", encoding="cp1252", errors="ignore") as f:
            lines = f.readlines()

        # find all hyperparameter-header lines
        hp_idxs = [
            i for i, line in enumerate(lines)
            if "Running hyperparameters: max_tta" in line
        ]
        if not hp_idxs:
            print(f"⚠️  No hyperparameter header in {log_path}")
            continue

        # define blocks between successive headers (or to EOF)
        blocks = []
        if len(hp_idxs) >= 2:
            blocks = list(zip(hp_idxs, hp_idxs[1:]))
        else:
            blocks = [(hp_idxs[0], len(lines))]

        for start, end in blocks:
            block = lines[start:end]
            # extract all Ensemble TTA MeanProb lines
            meanprobs = [
                float(re.search(r"Ensemble TTA: MeanProb=([\d\.]+)%", line).group(1))
                for line in block
                if "Ensemble TTA: MeanProb=" in line
            ]
            if len(meanprobs) != 9:
                print(f"⚠️  Expected 9 MeanProb entries, got {len(meanprobs)} in {subdir.name}")
                continue
            avg = sum(meanprobs) / len(meanprobs)
            entry = params.copy()
            entry["meanprobs"] = meanprobs
            entry["avg_meanprob"] = avg
            results.append(entry)

    return results

if __name__ == "__main__":
    base_dir = r"E:\Exoskeleton_DL\DeepTransferEEG\logs\CustomEpoch"
    results = parse_epoch_logs(base_dir)
    # pretty-print
    for r in results:
        print(
            f"tta={r['tta']}, stride={r['stride']}, t={r['t']}, "
            f"lr={r['lr']}, steps={r['steps']}  →  MeanProbs: {r['meanprobs']}  "
            f"Avg: {r['avg_meanprob']:.2f}"
        )
