import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import accuracy_score

# 1) load data + meta
X = np.load('./data/CustomEpoch/X.npy')           # (total_trials, n_ch, n_times)
y = np.load('./data/CustomEpoch/labels.npy')
meta = pd.read_csv('./data/CustomEpoch/meta.csv') # columns: file, n_trials

# 2) compute session boundaries
counts = meta['n_trials'].values
starts = np.concatenate(([0], counts.cumsum()[:-1]))
ends   = counts.cumsum()

# 3) design filters at 500 Hz
fs_orig, fs_target = 200, 100
ds_factor = fs_orig // fs_target  # == 2
nyq = fs_orig / 2                # == 100
b_notch, a_notch = iirnotch(50/nyq, 30)
b_bp,    a_bp    = butter(4, [8/nyq, 32/nyq], btype='band')

results = []  # collect (file, accuracy) tuples

# 4) loop over sessions
for sess_idx, (s,e) in enumerate(zip(starts, ends)):
    Xs = X[s:e]       # (n_trials_sess, n_ch, t0)
    ys = y[s:e]
    # preprocess: notch → bandpass → downsample by 5
    Xp = []
    for tr in Xs:
        d = filtfilt(b_notch, a_notch, tr, axis=1)
        d = filtfilt(b_bp,    a_bp,    d,  axis=1)
        d = d[:, ::ds_factor]  # correct 200→100 Hz
        Xp.append(d)
    Xp = np.stack(Xp)  # (n_trials_sess, n_ch, t1)

    # 5) split: first 20 for train
    X_train, y_train = Xp[:20], ys[:20]
    X_test,  y_test  = Xp[20:], ys[20:]

    # 6) fit CSP+LDA
    csp = CSP(n_components=6, reg=None, log=False, norm_trace=False)
    csp.fit(X_train, y_train)
    Xtr_feat = csp.transform(X_train)
    lda = LDA().fit(Xtr_feat, y_train)

    # 7) eval
    Xte_feat = csp.transform(X_test)
    acc = accuracy_score(y_test, lda.predict(Xte_feat))
    # record train/test counts too
    n_train, n_test = X_train.shape[0], X_test.shape[0]
    results.append((meta.loc[sess_idx, 'file'], n_train, n_test, acc))

# ==== clean summary report ====
print("Session,#Train,#Test,Accuracy")
for fname, n_train, n_test, acc in results:
    print(f"{fname},{n_train},{n_test},{acc:.3f}")

# ==== accuracy brackets ====
buckets = {'<0.5':0, '0.5-0.6':0, '0.6-0.7':0, '0.7-0.8':0, '0.8-0.9':0, '0.9-1.0':0}
for _, _, _, acc in results:
    if acc < 0.5:
        buckets['<0.5'] += 1
    elif acc < 0.6:
        buckets['0.5-0.6'] += 1
    elif acc < 0.7:
        buckets['0.6-0.7'] += 1
    elif acc < 0.8:
        buckets['0.7-0.8'] += 1
    elif acc < 0.9:
        buckets['0.8-0.9'] += 1
    else:
        buckets['0.9-1.0'] += 1
print("Accuracy brackets:")
for br, cnt in buckets.items():
    print(f"{br}: {cnt}")

# ==== average accuracy across sessions ====
avg_acc = np.mean([acc for _, _, _, acc in results])
print(f"Average accuracy across sessions: {avg_acc:.3f}")
