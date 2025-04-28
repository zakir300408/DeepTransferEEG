import os
import glob
import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, filtfilt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from tqdm import tqdm
from datetime import datetime
from scipy.linalg import fractional_matrix_power

###############################################################################
# Euclidean Alignment (EA) Class — arithmetic mean only, SPD-safe
###############################################################################
class EuclideanAlignment:
    @staticmethod
    def compute_reference_covariance(raw_trials, eps=1e-6):
        n = len(raw_trials)
        C = np.zeros((raw_trials[0].shape[0], raw_trials[0].shape[0]))
        for X in raw_trials:
            C += X @ X.T
        R_bar = C / n
        trace = np.trace(R_bar)
        R_bar += eps * (trace / R_bar.shape[0]) * np.eye(R_bar.shape[0])
        return R_bar

    @staticmethod
    def compute_transformation(raw_trials):
        R_bar = EuclideanAlignment.compute_reference_covariance(raw_trials)
        eigvals, eigvecs = np.linalg.eigh(R_bar)
        D_inv_sqrt = np.diag(1.0 / np.sqrt(eigvals))
        return eigvecs @ D_inv_sqrt @ eigvecs.T

    @staticmethod
    def apply_to_raw_trial(T, X):
        return T @ X

###############################################################################
# CSPClassifier: Compute CSP filters and extract log-variance features
###############################################################################
class CSPClassifier:
    def __init__(self, n_channels, n_components):
        self.n_channels = n_channels
        self.n_components = n_components
        self.filters_ = None

    def compute_csp(self, covariances, labels, eps=1e-6):
        classes = np.unique(labels)
        if len(classes) != 2:
            raise ValueError("CSP requires exactly 2 classes.")
        R = {c: np.zeros((self.n_channels, self.n_channels)) for c in classes}
        for cov, y in zip(covariances, labels):
            R[y] += cov
        for c in classes:
            tr = np.trace(R[c])
            R[c] /= (tr if tr > 0 else 1.0)
        R_sum = R[classes[0]] + R[classes[1]]
        trace = np.trace(R_sum)
        R_sum += eps * (trace / self.n_channels) * np.eye(self.n_channels)
        eigvals, U = np.linalg.eigh(R_sum)
        P = np.diag(1.0 / np.sqrt(eigvals)) @ U.T
        S = P @ R[classes[0]] @ P.T
        eig_s, U_s = np.linalg.eigh(S)
        idx = np.argsort(eig_s)
        sel = np.hstack([idx[:self.n_components], idx[-self.n_components:]])
        self.filters_ = U.T @ P.T @ U_s[:, sel]

    def extract_features(self, covariances):
        feats = []
        for cov in covariances:
            v = [np.log(self.filters_[:, i].T @ cov @ self.filters_[:, i])
                 for i in range(self.filters_.shape[1])]
            feats.append(v)
        return np.array(feats)

###############################################################################
# OwAR: Online weighted Adaptation Regularization
###############################################################################
def OwAR(Xs, ys, Xt, yt, options=None, K=None):
    if options is None:
        options = {'sigma': 0.1, 'lambda': 10, 'wt': 2.0, 'src_wt': 1.0}
    sigma, lam, wt, src_wt = (options[k] for k in ('sigma','lambda','wt','src_wt'))
    n, m = len(ys), len(yt)
    X = np.vstack((Xs, Xt))
    Y = np.concatenate((ys, yt))

    # source weights
    Ws = np.ones(n) * src_wt
    # target weights
    Wt = np.ones(m)
    for arr, W in [(ys, Ws), (yt, Wt)]:
        uniq, cnt = np.unique(arr, return_counts=True)
        if len(uniq) == 2:
            if arr is yt:
                W[arr == uniq[1]] = (cnt[0]/cnt[1]) * wt
            else:
                W[arr == uniq[1]] = cnt[0]/cnt[1]

    W = np.concatenate((Ws, Wt))
    E = np.diag(W)
    e = np.concatenate((np.ones(n)/n, -np.ones(m)/m))
    M = np.outer(e, e) * 2

    if K is None:
        K = X @ X.T
    K_train = K[:n+m, :n+m]
    A = (E + lam*M) @ K_train + sigma * np.eye(n+m)
    b = E @ Y
    return np.linalg.solve(A, b)

###############################################################################
# Data Loading Functions
###############################################################################
def load_segmented_data(mat_file, Fs=200):
    mat = loadmat(mat_file)
    E = mat['MyEpoch']        # trials x time x channels
    L = mat['MyLabel'].flatten()
    b, a = butter(4, [8/(Fs/2), 30/(Fs/2)], btype='band')
    start, end = Fs, 4*Fs
    trials = []
    for i in range(E.shape[0]):
        x = filtfilt(b, a, E[i], axis=0)[start:end]
        trials.append(x.T)
    return trials, L

def load_subject_data(folder, Fs=200):
    files = glob.glob(os.path.join(folder, "*.mat"))
    alls, alll = [], []
    for f in files:
        tr, lab = load_segmented_data(f, Fs)
        alls.extend(tr)
        alll.extend(lab.tolist())
    return alls, np.array(alll)

def load_source_data(root, target, Fs=200, use_ea=False):
    subs = [d for d in os.listdir(root)
            if os.path.isdir(os.path.join(root, d)) and d != target]
    trials, labels = [], []
    for s in tqdm(subs, desc="Loading source"):
        tr, lab = load_subject_data(os.path.join(root, s), Fs)
        trials.extend(tr)
        labels.extend(lab.tolist())
    return trials, np.array(labels)

###############################################################################
# Main Evaluation Routine
###############################################################################
def main():
    np.random.seed(0)
    root = r"E:\Exoskeleton_DL\XK_work\Data_Epoch"
    Fs = 200
    n_ch = 31
    n_csp = 6
    win_size = 40
    min_test = 5
    n_windows = 5
    options = {'sigma':0.01, 'lambda':1, 'wt':1.0, 'src_wt':1}
    use_ea = True
    eps_ridge = 1e-6

    logs = []
    def log(msg):
        print(msg)
        logs.append(msg)

    subjects = [d for d in os.listdir(root)
                if os.path.isdir(os.path.join(root, d))]
    log(f"Subjects: {subjects!r}")

    results = {}

    for subj in tqdm(subjects, desc="Eval subjects"):
        # load & map source (with per‐subject EA if enabled)
        S_raw, S_lab = load_source_data(root, subj, Fs, use_ea)
        log(f"Subject {subj} source labels: {np.unique(S_lab)!r}")
        classes = np.unique(S_lab)
        label_map = {classes[0]: -1, classes[1]: 1}

        # compute & ridge source covs
        S_cov = []
        for X in S_raw:
            cov = X @ X.T
            cov += eps_ridge * (np.trace(cov)/n_ch) * np.eye(n_ch)
            S_cov.append(cov)

        # train CSP on source
        csp_source = CSPClassifier(n_ch, n_csp)
        csp_source.compute_csp(S_cov, S_lab)
        Xs = csp_source.extract_features(S_cov)
        ys = np.array([label_map[y] for y in S_lab])

        # iterate sessions
        sesss = glob.glob(os.path.join(root, subj, "*.mat"))
        for sess in sesss:
            T_raw, T_lab = load_segmented_data(sess, Fs)
            log(f"Session {os.path.basename(sess)} labels: {np.unique(T_lab)!r}")

            max_start = len(T_raw) - win_size - min_test
            if max_start < 0:
                log(f"  skipping ({len(T_raw)} trials)")
                continue

            starts = np.random.choice(max_start+1,
                                      size=min(n_windows, max_start+1),
                                      replace=False)
            log(f"  windows: {starts.tolist()}")

            accs_tl = []
            accs_base = []

            for st in starts:
                idx_tr = list(range(st, st+win_size))
                idx_te = [i for i in range(len(T_raw)) if i not in idx_tr]
                if len(idx_te) < min_test:
                    continue

                ytr_raw = T_lab[idx_tr]
                yte_raw = T_lab[idx_te]
                if len(np.unique(ytr_raw))<2 or len(np.unique(yte_raw))<2:
                    continue

                # map labels
                yt = np.array([label_map[y] for y in ytr_raw])
                ytrue = np.array([label_map[y] for y in yte_raw])

                # TODO: did not use EuclideanAlignment.compute_reference_covariance implementation, since that would involve re-computing the average over all previous covariance matrices
                # a better alternative is to UPDATE the transformation T matrix, instead of RE-calculating it from scratch
                # extract trials by indices
                Xtr = [T_raw[i] for i in idx_tr]
                Xte = [T_raw[i] for i in idx_te]
                if use_ea:
                    # EA using only target training trials (no leakage)
                    T_t = EuclideanAlignment.compute_transformation(Xtr)
                    Xtr = [EuclideanAlignment.apply_to_raw_trial(T_t, X) for X in Xtr]
                    Xte = [EuclideanAlignment.apply_to_raw_trial(T_t, X) for X in Xte]

                # cov + ridge
                cov_tr, cov_te = [], []
                for X in Xtr:
                    cov = X @ X.T
                    cov += eps_ridge * (np.trace(cov)/n_ch) * np.eye(n_ch)
                    cov_tr.append(cov)
                for X in Xte:
                    cov = X @ X.T
                    cov += eps_ridge * (np.trace(cov)/n_ch) * np.eye(n_ch)
                    cov_te.append(cov)

                # === Baseline CSP+LDA ===
                csp_t = CSPClassifier(n_ch, n_csp)
                csp_t.compute_csp(cov_tr, ytr_raw)  # labels in original 0/1
                Xtr_base = csp_t.extract_features(cov_tr)
                Xte_base = csp_t.extract_features(cov_te)
                lda = LinearDiscriminantAnalysis()
                lda.fit(Xtr_base, ytr_raw)
                ypred_base = lda.predict(Xte_base)
                acc_base = np.mean(ypred_base == yte_raw)
                accs_base.append(acc_base)
                log(f"    baseline win {st}-{st+win_size-1}: {acc_base*100:.1f}%")

                # === Transfer OwAR ===
                Xt = csp_source.extract_features(cov_tr)
                Xtest = csp_source.extract_features(cov_te)
                X_comb = np.vstack((Xs, Xt))
                K_train = X_comb @ X_comb.T
                alpha = OwAR(Xs, ys, Xt, yt, options=options, K=K_train)
                K_test = Xtest @ X_comb.T
                ypred_tl = np.sign(K_test @ alpha)
                acc_tl = np.mean(ypred_tl == ytrue)
                accs_tl.append(acc_tl)
                log(f"    transfer win {st}-{st+win_size-1}: {acc_tl*100:.1f}%")

            if accs_tl:
                mean_tl = np.mean(accs_tl)
                mean_base = np.mean(accs_base)
                log(f"  avg transfer: {mean_tl*100:.1f}%  avg baseline: {mean_base*100:.1f}%")
                results[(subj, os.path.basename(sess))] = (mean_tl, mean_base)

    # save logs & summary
    out = "classification_results"
    os.makedirs(out, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(out, f"logs_{timestamp}")
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "log.txt"), "w") as f:
        f.write("\n".join(logs))
    with open(os.path.join(path, "summary.txt"), "w") as f:
        f.write("subject,session,transfer_acc,baseline_acc\n")
        for (s, sess), (tl, base) in results.items():
            f.write(f"{s},{sess},{tl:.4f},{base:.4f}\n")

    print(f"Done. Logs in {path}")

if __name__ == "__main__":
    main()
