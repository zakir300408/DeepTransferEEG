import os
import glob
import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, iirnotch
from sklearn.model_selection import KFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import FastICA
from scipy.stats import kurtosis
from tqdm import tqdm

###############################################################################
# Euclidean Alignment (EA) Function (simplified)
###############################################################################
def apply_ea(raw_trials):
    """
    Compute EA transform from raw_trials and return aligned covariances.
    """
    # estimate reference covariance
    R_bar = np.zeros((raw_trials[0].shape[0], raw_trials[0].shape[0]))
    for X in raw_trials:
        R_bar += X @ X.T
    R_bar /= len(raw_trials)
    eigvals, U = np.linalg.eig(R_bar)
    eigvals, U = np.real(eigvals), np.real(U)
    T = U @ np.diag(1.0/np.sqrt(eigvals)) @ U.T
    # align trials
    aligned = []
    for X in raw_trials:
        Xa = T @ X
        aligned.append(Xa @ Xa.T)
    return aligned

###############################################################################
# Preprocessing
###############################################################################
def preprocess_trials(epochs, Fs=200):
    """
    Bandpass filter (8–30Hz), apply notch filter, and perform ICA-based artifact rejection.
    epochs: array (n_trials, timepoints, channels).
    Returns list of arrays (channels, samples).
    """
    filt_n = 4
    Wn = [8/(Fs/2), 30/(Fs/2)]
    b, a = butter(filt_n, Wn, btype='band')
    # Notch filter parameters (50Hz, Q-factor=30)
    notch_freq = 50.0
    Q = 30.0
    b_notch, a_notch = iirnotch(notch_freq, Q, Fs)
    start, end = Fs, 4*Fs
    processed = []
    for trial in epochs:  # trial shape: timepoints x channels
        filt = filtfilt(b, a, trial, axis=0)
        # Apply notch filter
        notch_filtered = filtfilt(b_notch, a_notch, filt, axis=0)
        seg = notch_filtered[start:end, :]
        
        # Perform ICA and artifact rejection
        ica = FastICA(n_components=seg.shape[1], random_state=0, max_iter=1000, tol=1e-4)
        ica_components = ica.fit_transform(seg)  # shape: (samples, components)
        comp_kurt = kurtosis(ica_components, axis=0)
        threshold = 5.0
        bad_components = [i for i, k in enumerate(comp_kurt) if abs(k) > threshold]
        if bad_components:
            ica_components[:, bad_components] = 0  # zero out artifacts
            seg = ica.inverse_transform(ica_components)
        
        processed.append(seg.T)
    return processed

###############################################################################
# Data Loading Functions
###############################################################################
def load_subject_data(subject_folder, Fs=200, data_channel=31):
    mat_files = glob.glob(os.path.join(subject_folder, "*.mat"))
    aggregated_trials_R = []
    aggregated_trials_Type = []
    for mat_file in tqdm(mat_files, desc=f"Loading files from {os.path.basename(subject_folder)}"):
        mat = loadmat(mat_file)
        print(f"Processing file: {mat_file}")
        MyEpoch = mat['MyEpoch']  # shape: (n_trials, time, channels)
        MyLabel = mat['MyLabel']  # shape: (n_trials, 1)
        n_trials = MyEpoch.shape[0]
        
        # extract and preprocess all trials at once
        raw_trials = preprocess_trials(MyEpoch, Fs)
        labels = MyLabel.flatten().tolist()
        
        aggregated_trials_R.extend(raw_trials)
        aggregated_trials_Type.extend(labels)
    return aggregated_trials_R, aggregated_trials_Type

def load_all_subjects_data(root_folder, Fs=200, data_channel=31):
    subjects = [d for d in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, d))]
    all_trials_R, all_trials_Type, all_subject_ids = [], [], []
    subj_details = {}
    for subj in tqdm(subjects, desc="Loading subjects"):
        subject_folder = os.path.join(root_folder, subj)
        mat_files = glob.glob(os.path.join(subject_folder, "*.mat"))
        sess_count = len(mat_files)
        trials_R, trials_Type = load_subject_data(subject_folder, Fs, data_channel)
        trial_count = len(trials_R)
        subj_details[subj] = {'sessions': sess_count, 'trials': trial_count}
        all_trials_R.extend(trials_R)
        all_trials_Type.extend(trials_Type)
        all_subject_ids.extend([subj] * len(trials_R))
    return all_trials_R, all_trials_Type, all_subject_ids, subj_details

###############################################################################
# CSP and Classification
###############################################################################
def apply_csp_and_classify(train_data, train_labels, N, classifier_type):
    from mne.decoding import CSP
    X_train = np.stack(train_data, axis=0)
    y_train = np.array(train_labels)
    csp = CSP(n_components=2*N)
    csp.fit(X_train, y_train)
    X_train_csp = csp.transform(X_train)

    if classifier_type == 'LDA':
        model = LinearDiscriminantAnalysis()
    elif classifier_type == 'SVM':
        model = SVC(kernel='linear', probability=True)
    elif classifier_type == 'Logistic':
        model = LogisticRegression(max_iter=1000)
    elif classifier_type == 'RegLDA':
        model = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    else:
        raise ValueError("Unknown classifier type: " + classifier_type)

    model.fit(X_train_csp, y_train)
    return model, csp

def loso_eval(trials_raw, trials_Type, subject_ids, data_channel, N, classifier_type='LDA', use_ea=False, log=None):
    """
    Leave-One-Subject-Out evaluation that optionally applies EA.
    """
    if log: log(f"\n==== Running LOSO Evaluation (use_ea={use_ea}) ====")
    unique_subjects = np.unique(subject_ids)
    loso_accuracies = {}

    for test_subj in tqdm(unique_subjects, desc="LOSO"):
        test_mask = np.array([s == test_subj for s in subject_ids])
        train_mask = ~test_mask

        # Convert to correct formats - ensure we're working with lists of arrays
        train_raw = [trials_raw[i] for i in range(len(trials_raw)) if train_mask[i]]
        test_raw  = [trials_raw[i] for i in range(len(trials_raw)) if test_mask[i]]
        train_labels = [trials_Type[i] for i in range(len(trials_Type)) if train_mask[i]]
        test_labels  = [trials_Type[i] for i in range(len(trials_Type)) if test_mask[i]]

        if log:
            log(f"Testing on {test_subj} ({len(test_labels)} trials), training on {len(train_labels)} trials")

        # Check data validity
        if len(train_raw) == 0 or len(test_raw) == 0:
            if log: log(f"Warning: No data found for subject {test_subj}")
            continue

        if use_ea:
            train_aligned = apply_ea(train_raw)
            test_aligned  = apply_ea(test_raw)
            final_train_data = train_aligned
            final_test_data  = test_aligned
        else:
            final_train_data = [np.dot(X, X.T) for X in train_raw]
            final_test_data  = [np.dot(X, X.T) for X in test_raw]

        # Call the new helper function
        model, csp = apply_csp_and_classify(final_train_data, train_labels, N, classifier_type)
        X_test = np.stack(final_test_data, axis=0)
        preds = model.predict(csp.transform(X_test))
        acc = np.mean(preds == test_labels)
        loso_accuracies[test_subj] = acc
        if log: log(f"  {test_subj} accuracy={acc*100:.2f}%")

    overall = np.mean(list(loso_accuracies.values()))
    if use_ea:
        print(f"Overall LOSO (with EA) Accuracy: {overall*100:.2f}%")
    else:
        print(f"Overall LOSO Accuracy: {overall*100:.2f}%")

    return loso_accuracies, overall

###############################################################################
# Helpers for main()
###############################################################################
def save_logs_and_summary(log_lines, results_summary, subj_details, apply_ea, classifier_type):
    base = "classification_results"
    os.makedirs(base, exist_ok=True)
    from datetime import datetime
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    status = "EA" if apply_ea else "No_EA"
    folder = os.path.join(base, f"logs_{stamp}_{status}_{classifier_type}")
    os.makedirs(folder, exist_ok=True)
    # save log
    with open(os.path.join(folder, "evaluation_log.txt"), "w") as f:
        f.write("\n".join(log_lines))
    # save summary
    with open(os.path.join(folder, "summary_results.txt"), "w") as f:
        f.write(f"loso={results_summary['loso_accuracies']}\n")
        f.write(f"overall_loso={results_summary['overall_loso_accuracy']}\n")
        f.write(f"subject_details={subj_details}\n")
    print(f"Results saved in {folder}")

###############################################################################
# Simplified main()
###############################################################################
def main():
    # params
    root_folder = r"E:\Exoskeleton_DL\XK_work\Data_Epoch"
    Fs, data_channel, N = 200, 31, 6
    apply_ea, classifier_type = True, 'RegLDA'  # 'LDA', 'SVM', 'Logistic', 'RegLDA'

    log_lines = []
    def log(msg):
        print(msg)
        log_lines.append(msg)

    # Use load_all_subjects_data instead of manually loading .mat files
    trials_raw, trials_Type, subject_ids, subj_details = load_all_subjects_data(
        root_folder, Fs=Fs, data_channel=data_channel
    )
    log(f"Total trials aggregated: {len(trials_raw)}")
    for subj, details in subj_details.items():
        log(f"Subject {subj}: Sessions={details['sessions']}, Trials={details['trials']}")

    # Choose evaluation method based on EA setting
    loso_acc, overall = loso_eval(
        trials_raw, trials_Type, subject_ids,
        data_channel, N, classifier_type,
        use_ea=apply_ea, log=log
    )

    results_summary = {
        'loso_accuracies': loso_acc,
        'overall_loso_accuracy': overall
    }
    save_logs_and_summary(log_lines, results_summary, subj_details, apply_ea, classifier_type)

if __name__ == '__main__':
    main()

