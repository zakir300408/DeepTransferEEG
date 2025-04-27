import os
import glob
import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, filtfilt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
from datetime import datetime

###############################################################################
# Euclidean Alignment (EA) Class
###############################################################################
class EuclideanAlignment:
    """
    Class to perform Euclidean Alignment (EA) on EEG trials.
    """
    @staticmethod
    def compute_reference_covariance(raw_trials):
        n = len(raw_trials)
        R_bar = np.zeros((raw_trials[0].shape[0], raw_trials[0].shape[0]))
        for X in raw_trials:
            R_bar += np.dot(X, X.T)
        return R_bar / n

    @staticmethod
    def compute_transformation(raw_trials):
        """
        Compute the whitening transformation (sqrtRefEA) from the arithmetic mean covariance.
        Equivalent to MATLAB's: sqrtRefEA = refEA^(-1/2)
        """
        R_bar = EuclideanAlignment.compute_reference_covariance(raw_trials)
        eigvals, U = np.linalg.eig(R_bar)
        T = U @ np.diag(1.0 / np.sqrt(np.real(eigvals))) @ U.T
        return T

    @staticmethod
    def apply_to_raw_trial(T, X):
        """
        Apply the EA transformation T to a raw trial X.
        This mimics MATLAB's: XEA = sqrtRefEA * X.
        """
        return T @ X

    @staticmethod
    def aligned_covariance_from_raw(aligned_X):
        """
        Given an aligned raw trial signal, compute its covariance matrix.
        Equivalent to MATLAB's: R_aligned = XEA * XEA'
        """
        return np.dot(aligned_X, aligned_X.T)

    @staticmethod
    def align_trials(raw_trials):
        """
        Align a list of raw EEG trials and return their aligned covariance matrices.
        """
        T = EuclideanAlignment.compute_transformation(raw_trials)
        aligned_covs = []
        for X in raw_trials:
            X_aligned = EuclideanAlignment.apply_to_raw_trial(T, X)
            aligned_covs.append(EuclideanAlignment.aligned_covariance_from_raw(X_aligned))
        return aligned_covs

###############################################################################
# CSPClassifier: Compute CSP filters, extract log-variance features, and train a classifier
###############################################################################
class CSPClassifier:
    """
    Compute Common Spatial Patterns (CSP) filters, extract log-variance features,
    and train a classifier (options: LDA, SVM, Logistic Regression, or RegLDA).
    """
    def __init__(self, data_channel, N, classifier_type='LDA'):
        self.data_channel = data_channel  # number of channels
        self.N = N                        # number of CSP pairs per class
        self.classifier_type = classifier_type
        self.F = None                     # CSP spatial filters
        self.model = None                 # Trained classifier

    def compute_csp(self, trainR, trainType):
        R1 = np.zeros((self.data_channel, self.data_channel))
        R2 = np.zeros((self.data_channel, self.data_channel))
        for i in range(len(trainR)):
            if trainType[i] == 0:
                R1 += trainR[i]
            else:
                R2 += trainR[i]
        R1 /= np.trace(R1)
        R2 /= np.trace(R2)
        R3 = R1 + R2

        eigvals, U0 = np.linalg.eig(R3)
        P = np.dot(np.diag(1.0 / np.sqrt(np.real(eigvals))), U0.T)
        
        YL = P @ R1 @ P.T
        eigvals_L, UL = np.linalg.eig(YL)
        I = np.argsort(np.real(eigvals_L))[::-1]
        sel_indices = np.concatenate((I[:self.N], I[-self.N:]))
        self.F = P.T @ UL[:, sel_indices]

    def extract_features(self, trials_R):
        features = []
        for R_trial in trials_R:
            f = [np.log(np.dot(self.F[:, j].T, R_trial @ self.F[:, j]))
                 for j in range(2 * self.N)]
            features.append(np.array(f))
        return np.array(features)

    def train_classifier(self, X_train, y_train):
        if self.classifier_type == 'LDA':
            self.model = LinearDiscriminantAnalysis()
        elif self.classifier_type == 'SVM':
            self.model = SVC(kernel='linear', probability=True)
        elif self.classifier_type == 'Logistic':
            self.model = LogisticRegression(max_iter=1000)
        elif self.classifier_type == 'RegLDA':
            self.model = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
        else:
            raise ValueError("Unknown classifier type: " + self.classifier_type)
        self.model.fit(X_train, y_train)

    def fit(self, trainR, trainType):
        self.compute_csp(trainR, trainType)
        X_train = self.extract_features(trainR)
        self.train_classifier(X_train, trainType)

    def predict(self, trials_R):
        X = self.extract_features(trials_R)
        return self.model.predict(X)

    def score(self, trials_R, true_labels):
        preds = self.predict(trials_R)
        accuracy = np.mean(preds == true_labels)
        return accuracy, preds

###############################################################################
# Data Loading Functions (no EA flag here)
###############################################################################
def load_segmented_data(mat_file, Fs=200, data_channel=31, return_raw=True):
    """
    Load a MAT file containing pre-segmented EEG data.
    Assumes the file has:
      - 'MyEpoch': shape (n_trials, 1600, 31)
      - 'MyLabel': shape (1, n_trials)
    Each trial is bandpass filtered and a segment is extracted.
    If return_raw is True, return the raw trial matrices; otherwise, return simple covariance matrices.
    """
    mat = loadmat(mat_file)
    print(f"Processing file: {mat_file}")
    MyEpoch = mat['MyEpoch']
    MyLabel = mat['MyLabel']
    n_trials = MyEpoch.shape[0]
    labels = MyLabel.flatten()

    filt_n = 4
    Wn = [8 / (Fs / 2), 30 / (Fs / 2)]
    filter_b, filter_a = butter(filt_n, Wn, btype='band')
    StartTimePoint = Fs      
    EndTimePoint = 4 * Fs    
    
    raw_trials = []
    for i in range(n_trials):
        trial = MyEpoch[i, :, :]
        data_filter = filtfilt(filter_b, filter_a, trial, axis=0)
        segment = data_filter[StartTimePoint:EndTimePoint, :]
        raw_trials.append(segment.T)
    
    if return_raw:
        return raw_trials, labels.tolist()
    else:
        cov_trials = [np.dot(X, X.T) for X in raw_trials]
        return cov_trials, labels.tolist()

def load_subject_data(subject_folder, Fs=200, data_channel=31, return_raw=True):
    mat_files = glob.glob(os.path.join(subject_folder, "*.mat"))
    aggregated_trials = []
    aggregated_labels = []
    for mat_file in tqdm(mat_files, desc=f"Loading files from {os.path.basename(subject_folder)}"):
        trials, labels = load_segmented_data(mat_file, Fs, data_channel, return_raw)
        aggregated_trials.extend(trials)
        aggregated_labels.extend(labels)
    return aggregated_trials, aggregated_labels

def load_all_subjects_data(root_folder, Fs=200, data_channel=31, return_raw=True):
    subjects = [d for d in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, d))]
    all_trials = []
    all_labels = []
    all_subject_ids = []
    for subj in tqdm(subjects, desc="Loading subjects"):
        subject_folder = os.path.join(root_folder, subj)
        trials, labels = load_subject_data(subject_folder, Fs, data_channel, return_raw)
        all_trials.extend(trials)
        all_labels.extend(labels)
        all_subject_ids.extend([subj] * len(trials))
    return all_trials, all_labels, all_subject_ids

###############################################################################
# Evaluation Helper Function
###############################################################################
def perform_ea_alignment(train_raw_trials, test_raw_trials, log_fn=None):
    """
    Compute the EA transformation using the training trials and then
    apply it to both training and testing trials.
    This aligns the raw trials (like MATLAB: XEA = sqrtRefEA * X) and then
    computes their covariance matrices.
    
    If a log function (log_fn) is provided, log the eigenvalues of the EA transformation.
    """
    T = EuclideanAlignment.compute_transformation(train_raw_trials)
    if log_fn:
        eigvals_T = np.linalg.eigvals(T)
        log_fn(f"EA Transformation eigenvalues: {eigvals_T}")
    aligned_train_raw = [EuclideanAlignment.apply_to_raw_trial(T, X) for X in train_raw_trials]
    aligned_test_raw = [EuclideanAlignment.apply_to_raw_trial(T, X) for X in test_raw_trials]
    aligned_train_R = [EuclideanAlignment.aligned_covariance_from_raw(X_aligned) for X_aligned in aligned_train_raw]
    aligned_test_R = [EuclideanAlignment.aligned_covariance_from_raw(X_aligned) for X_aligned in aligned_test_raw]
    return aligned_train_R, aligned_test_R

###############################################################################
# Main Evaluation Routine (Sliding Window Evaluation)
###############################################################################
def main():
    root_folder = r"E:\Exoskeleton_DL\XK_work\Data_Epoch"
    Fs = 200
    data_channel = 31
    N = 6
    classifier_type = 'LDA'
    run_trial_subset = True
    num_trial_sliding_window = 20

    apply_ea = True

    log_lines = []
    def log(msg):
        print(msg)
        log_lines.append(msg)
    
    if apply_ea:
        log("EA flag is enabled: EA will be applied for each sliding window evaluation.")
    else:
        log("EA flag is disabled: Conventional covariance computation will be used.")

    trials_all, labels_all, subject_ids = load_all_subjects_data(root_folder, Fs, data_channel, return_raw=True)
    totalTrials = len(trials_all)
    log(f"Total trials aggregated: {totalTrials}")
    
    unique_subjects = np.unique(subject_ids)
    subj_details = {}
    log("\nSubjects details:")
    for subj in unique_subjects:
        subject_folder = os.path.join(root_folder, subj)
        session_files = glob.glob(os.path.join(subject_folder, "*.mat"))
        trial_count = np.sum(np.array(subject_ids) == subj)
        subj_details[subj] = {'sessions': len(session_files), 'trials': int(trial_count)}
        log(f"Subject {subj}: Sessions = {len(session_files)}, Trials = {trial_count}")
    
    results_summary = {}

    if run_trial_subset:
        trial_subset_accuracies = {}
        log(f"\n{num_trial_sliding_window}-Trial Sliding Window Evaluation (per session):")
        for subj in tqdm(unique_subjects, desc="Subject evaluation"):
            subject_folder = os.path.join(root_folder, subj)
            session_files = glob.glob(os.path.join(subject_folder, "*.mat"))
            for sess_file in session_files:
                raw_trials, trials_Type_sess = load_segmented_data(sess_file, Fs, data_channel, return_raw=True)
                n_trials_sess = len(raw_trials)
                if n_trials_sess < num_trial_sliding_window + 1:
                    log(f"Session {os.path.basename(sess_file)} of subject {subj} has {n_trials_sess} trials (< {num_trial_sliding_window + 1}). Skipping.")
                    continue

                window_accuracies = []
                window_range = range(n_trials_sess - num_trial_sliding_window + 1)
                for start in tqdm(window_range, desc=f"Window evaluation for {os.path.basename(sess_file)}", leave=False):
                    end = start + num_trial_sliding_window
                    train_indices = list(range(start, end))
                    test_indices = sorted(list(set(range(n_trials_sess)) - set(train_indices)))
                    
                    log(f"Subject {subj}, Session {os.path.basename(sess_file)}, Window {start}-{end-1}:")
                    log(f"    Training indices: {train_indices}")
                    log(f"    Testing indices: {test_indices}")

                    train_raw_trials = [raw_trials[idx] for idx in train_indices]
                    test_raw_trials = [raw_trials[idx] for idx in test_indices]
                    train_labels = [trials_Type_sess[idx] for idx in train_indices]
                    test_labels = [trials_Type_sess[idx] for idx in test_indices]

                    try:
                        if apply_ea:
                            log("    EA is applied for this window.")
                            aligned_train_R, aligned_test_R = perform_ea_alignment(train_raw_trials, test_raw_trials, log_fn=log)
                        else:
                            log("    EA is NOT applied for this window; using conventional covariance.")
                            aligned_train_R = [np.dot(X, X.T) for X in train_raw_trials]
                            aligned_test_R  = [np.dot(X, X.T) for X in test_raw_trials]

                        csp_clf = CSPClassifier(data_channel, N, classifier_type=classifier_type)
                        csp_clf.fit(aligned_train_R, train_labels)
                        acc, _ = csp_clf.score(aligned_test_R, test_labels)
                    except Exception as e:
                        log(f"Error for window starting at {start} in subject {subj}, session {os.path.basename(sess_file)}: {e}")
                        acc = 0

                    window_accuracies.append(acc)
                    log(f"    Window {start}-{end-1}: Accuracy = {acc * 100:.2f}%")

                session_avg_acc = np.mean(window_accuracies)
                log(f"Subject {subj}, Session {os.path.basename(sess_file)}: Overall Sliding Window Accuracy = {session_avg_acc * 100:.2f}%")
                trial_subset_accuracies[(subj, os.path.basename(sess_file))] = session_avg_acc

        if trial_subset_accuracies:
            overall_subset_accuracy = np.mean(list(trial_subset_accuracies.values()))
            log(f"Overall {num_trial_sliding_window}-Trial Sliding Window Evaluation Accuracy (all sessions): {overall_subset_accuracy * 100:.2f}%")
            results_summary[f'subset_{num_trial_sliding_window}_trials_sliding_window_accuracies'] = trial_subset_accuracies
            results_summary[f'overall_subset_{num_trial_sliding_window}_trials_sliding_window_accuracy'] = overall_subset_accuracy

    classification_results_folder = "classification_results"
    os.makedirs(classification_results_folder, exist_ok=True)
    log_folder = os.path.join(
        classification_results_folder,
        f"logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}_EA_{classifier_type}"
    )
    os.makedirs(log_folder, exist_ok=True)
    log_file = os.path.join(log_folder, "evaluation_log.txt")
    with open(log_file, "w") as f:
        for line in log_lines:
            f.write(line + "\n")
    print(f"Log saved to {log_file}")

    summary_file = os.path.join(log_folder, "summary_results.txt")
    with open(summary_file, "w") as f:
        if run_trial_subset:
            f.write(f"subset_{num_trial_sliding_window}_trials_sliding_window="
                    f"{results_summary.get(f'subset_{num_trial_sliding_window}_trials_sliding_window_accuracies', 'N/A')}\n")
            f.write(f"overall_subset_{num_trial_sliding_window}_trials_sliding_window_accuracy="
                    f"{results_summary.get(f'overall_subset_{num_trial_sliding_window}_trials_sliding_window_accuracy', 'N/A')}\n")
        f.write(f"subject_details={subj_details}\n")
    print("Summary results saved to summary_results.txt")

if __name__ == '__main__':
    main()
