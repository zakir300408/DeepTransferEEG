import os
import glob
import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, filtfilt
from sklearn.model_selection import KFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
import itertools
from collections import defaultdict

###############################################################################
# Euclidean Alignment (EA) Function
###############################################################################
def apply_ea_alignment(raw_trials):
    """
    Apply Euclidean Alignment (EA) to a list of raw EEG trials.
    Each trial X is assumed to be of shape (channels, samples).
    
    EA computes the reference matrix as the arithmetic mean of the
    individual trial covariance matrices and then aligns each trial:
        R_bar = (1/n) * sum_i (X_i X_i^T)
        T = R_bar^(-1/2)
        X_aligned = T @ X_i
        R_aligned = X_aligned @ X_aligned^T
    Returns a list of aligned covariance matrices.
    """
    n = len(raw_trials)
    # Compute the reference matrix as the arithmetic mean of covariance matrices
    R_bar = np.zeros((raw_trials[0].shape[0], raw_trials[0].shape[0]))
    for X in raw_trials:
        R_bar += np.dot(X, X.T)
    R_bar /= n
    # Eigen-decomposition to compute the inverse square root of R_bar
    eigvals, U = np.linalg.eig(R_bar)
    eigvals = np.real(eigvals)
    U = np.real(U)
    inv_sqrt = U @ np.diag(1.0 / np.sqrt(eigvals)) @ U.T
    aligned_covs = []
    for X in raw_trials:
        X_aligned = inv_sqrt @ X
        R_aligned = np.dot(X_aligned, X_aligned.T)
        aligned_covs.append(R_aligned)
    return aligned_covs

def calculate_ea_transform(raw_trials):
    """
    Calculate the EA transformation matrix from a list of raw EEG trials.
    Each trial X is assumed to be of shape (channels, samples).
    
    Returns the transformation matrix.
    """
    n = len(raw_trials)
    R_bar = np.zeros((raw_trials[0].shape[0], raw_trials[0].shape[0]))
    for X in raw_trials:
        R_bar += np.dot(X, X.T)
    R_bar /= n
    eigvals, U = np.linalg.eig(R_bar)
    eigvals = np.real(eigvals)
    U = np.real(U)
    inv_sqrt = U @ np.diag(1.0 / np.sqrt(eigvals)) @ U.T
    return inv_sqrt

def apply_ea_transform(raw_trials, transform_matrix):
    """
    Apply the EA transformation matrix to a list of raw EEG trials.
    Each trial X is assumed to be of shape (channels, samples).
    
    Returns a list of aligned covariance matrices.
    """
    aligned_covs = []
    for X in raw_trials:
        X_aligned = transform_matrix @ X
        R_aligned = np.dot(X_aligned, X_aligned.T)
        aligned_covs.append(R_aligned)
    return aligned_covs

###############################################################################
# CSPClassifier: Compute CSP filters, extract log-variance features, and train a classifier
###############################################################################
class CSPClassifier:
    """
    Class to compute Common Spatial Patterns (CSP) filters,
    extract log-variance features, and then train a classifier.
    The classifier can be chosen from a set of models.
    """
    def __init__(self, data_channel, N, classifier_type='LDA'):
        self.data_channel = data_channel  # number of channels
        self.N = N                        # number of CSP pairs per class
        self.classifier_type = classifier_type
        self.F = None                     # CSP spatial filters
        self.sSP = None                   # CSP patterns (optional)
        self.model = None                 # Trained classifier

    def compute_csp(self, trainR, trainType):
        R1 = np.zeros((self.data_channel, self.data_channel))
        R2 = np.zeros((self.data_channel, self.data_channel))
        for i in range(len(trainR)):
            if trainType[i] == 0:
                R1 += trainR[i]
            else:
                R2 += trainR[i]
        R1 = R1 / np.trace(R1)
        R2 = R2 / np.trace(R2)
        R3 = R1 + R2

        eigvals, U0 = np.linalg.eig(R3)
        eigvals = np.real(eigvals)
        U0 = np.real(U0)
        P = np.dot(np.diag(1.0 / np.sqrt(eigvals)), U0.T)
        
        YL = np.dot(np.dot(P, R1), P.T)
        eigvals_L, UL = np.linalg.eig(YL)
        eigvals_L = np.real(eigvals_L)
        UL = np.real(UL)
        I = np.argsort(eigvals_L)[::-1]
        
        sel_indices = np.concatenate((I[:self.N], I[-self.N:]))
        self.F = np.dot(P.T, UL[:, sel_indices])
        
        SP = np.linalg.inv(np.dot(P.T, UL))
        indices = np.r_[np.arange(0, self.N), np.arange(self.data_channel - self.N, self.data_channel)]
        self.sSP = SP[indices, :]

    def extract_features(self, trials_R):
        features = []
        for R_trial in trials_R:
            f = []
            for j in range(2 * self.N):
                val = np.log(np.dot(self.F[:, j].T, np.dot(R_trial, self.F[:, j])))
                f.append(val)
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
# Data Loading Functions
###############################################################################
def load_segmented_data(mat_file, Fs=200, data_channel=31, apply_ea=False):
    """
    Load a MAT file containing pre-segmented data with keys:
      - 'MyEpoch': shape (n_trials, 1600, 31)
      - 'MyLabel': shape (1, n_trials)
    
    Each trial is filtered and a window is extracted (from Fs to 4*Fs).
    The raw trial is transposed to shape (channels, samples).
    
    If apply_ea is True, EA is applied to the raw trials before computing the covariance.
    Otherwise, covariance matrices are computed directly.
    
    Returns:
        trials_R: List of covariance matrices (possibly aligned)
        trials_Type: List of corresponding labels
    """
    mat = loadmat(mat_file)
    print(f"Processing file: {mat_file}")
    MyEpoch = mat['MyEpoch']  
    MyLabel = mat['MyLabel']  
    n_trials = MyEpoch.shape[0]
    labels = MyLabel.flatten()  
    
    filt_n = 4
    Wn = [8 / (Fs/2), 30 / (Fs/2)]
    filter_b, filter_a = butter(filt_n, Wn, btype='band')
    
    StartTimePoint = Fs      
    EndTimePoint = 4 * Fs    
    
    raw_trials = []
    for i in range(n_trials):
        trial = MyEpoch[i, :, :]
        data_filter = filtfilt(filter_b, filter_a, trial, axis=0)
        segment = data_filter[StartTimePoint:EndTimePoint, :]
        X = segment.T  # shape: (channels, samples)
        raw_trials.append(X)
    
    if apply_ea:
        trials_R = apply_ea_alignment(raw_trials)
    else:
        trials_R = [np.dot(X, X.T) for X in raw_trials]
    
    trials_Type = labels.tolist()
    return trials_R, trials_Type

def load_subject_data(subject_folder, Fs=200, data_channel=31, apply_ea=False):
    mat_files = glob.glob(os.path.join(subject_folder, "*.mat"))
    aggregated_trials_R = []
    aggregated_trials_Type = []
    for mat_file in tqdm(mat_files, desc=f"Loading files from {os.path.basename(subject_folder)}"):
        trials_R, trials_Type = load_segmented_data(mat_file, Fs, data_channel, apply_ea)
        aggregated_trials_R.extend(trials_R)
        aggregated_trials_Type.extend(trials_Type)
    return aggregated_trials_R, aggregated_trials_Type

def load_all_subjects_data(root_folder, Fs=200, data_channel=31, apply_ea=False):
    subjects = [d for d in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, d))]
    all_trials_R = []
    all_trials_Type = []
    all_subject_ids = []
    for subj in tqdm(subjects, desc="Loading subjects"):
        subject_folder = os.path.join(root_folder, subj)
        trials_R, trials_Type = load_subject_data(subject_folder, Fs, data_channel, apply_ea)
        all_trials_R.extend(trials_R)
        all_trials_Type.extend(trials_Type)
        all_subject_ids.extend([subj] * len(trials_R))
    return all_trials_R, all_trials_Type, all_subject_ids

###############################################################################
# Incremental Subject LOSO Evaluation
###############################################################################
def incremental_loso_evaluation(trials_R, trials_Type, subject_ids, data_channel, N, classifier_type, 
                               min_subjects=2, log_func=print):
    """
    Evaluate the effect of dataset size on LOSO performance.
    Start with min_subjects for training, then incrementally add more subjects.
    For each training set size, test on all remaining subjects.
    
    Args:
        trials_R: List of covariance matrices
        trials_Type: List of corresponding labels
        subject_ids: List of subject IDs for each trial
        data_channel: Number of channels
        N: Number of CSP pairs
        classifier_type: Type of classifier to use
        min_subjects: Minimum number of subjects for training
        log_func: Function to use for logging
        
    Returns:
        results: Dictionary with results for each training set size
    """
    unique_subjects = np.unique(subject_ids)
    num_subjects = len(unique_subjects)
    
    # Convert to numpy arrays if not already
    trials_R = np.array(trials_R, dtype=object)
    trials_Type = np.array(trials_Type)
    subject_ids = np.array(subject_ids)
    
    results = {}
    
    # Loop through different training set sizes
    for train_size in range(min_subjects, num_subjects):
        log_func(f"\n==== Training with {train_size} subjects, Testing on {num_subjects - train_size} subjects ====")
        
        # For each size, try multiple random combinations of subjects
        num_combinations = min(10, len(list(itertools.combinations(unique_subjects, train_size))))
        size_accuracies = []
        
        # Randomly select combinations of subjects for training
        subject_combinations = list(itertools.combinations(unique_subjects, train_size))
        np.random.shuffle(subject_combinations)
        subject_combinations = subject_combinations[:num_combinations]
        
        for i, train_subjects in enumerate(subject_combinations):
            log_func(f"\nCombination {i+1}/{num_combinations}: Training on subjects {', '.join(train_subjects)}")
            train_subjects = set(train_subjects)
            test_subjects = [s for s in unique_subjects if s not in train_subjects]
            
            # Create masks for training and testing
            train_mask = np.array([subj in train_subjects for subj in subject_ids])
            
            # Extract training data
            trainR = trials_R[train_mask].tolist()
            trainType = trials_Type[train_mask].tolist()
            
            # Train model
            csp_clf = CSPClassifier(data_channel, N, classifier_type=classifier_type)
            csp_clf.fit(trainR, trainType)
            
            # Test on each left-out subject
            subject_accs = {}
            for test_subj in test_subjects:
                test_mask = (subject_ids == test_subj)
                testR = trials_R[test_mask].tolist()
                testType = trials_Type[test_mask].tolist()
                
                num_test_trials = len(testR)
                if num_test_trials == 0:
                    continue
                    
                acc, _ = csp_clf.score(testR, testType)
                subject_accs[test_subj] = acc
                log_func(f"  Subject {test_subj}: Accuracy = {acc * 100:.2f}% ({num_test_trials} trials)")
            
            # Get average accuracy across all test subjects
            if subject_accs:
                avg_acc = np.mean(list(subject_accs.values()))
                size_accuracies.append(avg_acc)
                log_func(f"  Average accuracy for this combination: {avg_acc * 100:.2f}%")
        
        # Average across all combinations for this training set size
        if size_accuracies:
            mean_acc = np.mean(size_accuracies)
            std_acc = np.std(size_accuracies)
            results[train_size] = {
                'mean_accuracy': mean_acc,
                'std_accuracy': std_acc,
                'all_accuracies': size_accuracies
            }
            log_func(f"\nTraining set size {train_size}: Mean accuracy = {mean_acc * 100:.2f}% ± {std_acc * 100:.2f}%")
    
    return results


def loso_with_proper_ea(trials_raw, trials_Type, subject_ids, data_channel, N, classifier_type='LDA'):
    unique_subjects = np.unique(subject_ids)
    loso_accuracies = {}
    
    for test_subj in unique_subjects:
        test_mask = (subject_ids == test_subj)
        train_mask = ~test_mask
        
        # Get raw trials for both sets
        train_raw = np.array(trials_raw)[train_mask].tolist()
        test_raw = np.array(trials_raw)[test_mask].tolist()
        
        # Calculate EA transformation matrix from training data
        transform_matrix = calculate_ea_transform(train_raw)
        # Apply the same transformation to train and test data
        train_aligned = apply_ea_transform(train_raw, transform_matrix)
        test_aligned = apply_ea_transform(test_raw, transform_matrix)
        
        # Continue with CSP and classification as before
        csp_clf = CSPClassifier(data_channel, N, classifier_type=classifier_type)
        csp_clf.fit(train_aligned, np.array(trials_Type)[train_mask].tolist())
        acc, _ = csp_clf.score(test_aligned, np.array(trials_Type)[test_mask].tolist())
        loso_accuracies[test_subj] = acc
        print(f"Subject {test_subj}: Accuracy = {acc * 100:.2f}%")

    return loso_accuracies


###############################################################################
# Main Evaluation Routine with Logging and Summary Saving
###############################################################################
def main():
    # Data and algorithm parameters
    root_folder = r"E:\Exoskeleton_DL\XK_work\Data_Epoch"  
    Fs = 200
    data_channel = 31
    N = 6
    apply_ea = True  # Set to True to apply EA
    classifier_type = 'LDA'  # Options: ['LDA', 'SVM', 'Logistic', 'RegLDA']
    
    # Evaluation flags - set to False to skip specific evaluations
    run_inter_subject = False    # Run inter-subject k-fold cross-validation
    run_loso = True             # Run leave-one-subject-out evaluation
    run_intra_subject = False    # Run intra-subject cross-validation
    run_intra_session = False    # Run intra-session evaluation
    run_incremental_loso = False  # Run incremental subject LOSO evaluation
    
    # Initialize result variables
    inter_acc = None
    loso_accuracies = {}
    overall_loso = None
    intra_subject_accuracies = {}
    overall_intra = None
    intra_session_accuracies = {}
    overall_intra_session = None
    incremental_loso_results = {}
    
    log_lines = []
    def log(msg):
        print(msg)
        log_lines.append(msg)
    
    log("Loading data from all subjects...")
    trials_R, trials_Type, subject_ids = load_all_subjects_data(root_folder, Fs, data_channel, apply_ea)
    totalTrials = len(trials_R)
    log(f"Total trials aggregated: {totalTrials}")
    
    trials_R = np.array(trials_R, dtype=object)
    trials_Type = np.array(trials_Type)
    subject_ids = np.array(subject_ids)
    
    unique_subjects = np.unique(subject_ids)
    subj_details = {}
    log("\nSubjects details:")
    for subj in unique_subjects:
        subject_folder = os.path.join(root_folder, subj)
        session_files = glob.glob(os.path.join(subject_folder, "*.mat"))
        session_count = len(session_files)
        trial_count = np.sum(subject_ids == subj)
        subj_details[subj] = {'sessions': session_count, 'trials': int(trial_count)}
        log(f"Subject {subj}: Sessions = {session_count}, Trials = {trial_count}")
    
    # Dictionary to store summary results
    results_summary = {}
    
    ############################################################################
    # Inter-Subject k-Fold Cross Validation
    ############################################################################
    if run_inter_subject:
        log("\n==== Running Inter-Subject k-Fold Cross Validation ====")
        k = 10
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        inter_right = 0
        inter_total = 0
        for train_index, test_index in tqdm(kf.split(range(totalTrials)), total=k, desc="Inter-subject k-fold"):
            trainR = trials_R[train_index].tolist()
            trainType = trials_Type[train_index].tolist()
            testR = trials_R[test_index].tolist()
            testType = trials_Type[test_index].tolist()
            csp_clf = CSPClassifier(data_channel, N, classifier_type=classifier_type) 
            csp_clf.fit(trainR, trainType)
            _, preds = csp_clf.score(testR, testType)
            inter_right += np.sum(preds == testType)
            inter_total += len(testType)
        inter_acc = inter_right / inter_total
        log(f"Inter-subject k-fold CV Accuracy: {inter_acc * 100:.2f}%")
        results_summary['inter_subject_accuracy'] = inter_acc
    else:
        log("\n==== Skipping Inter-Subject k-Fold Cross Validation ====")
        results_summary['inter_subject_accuracy'] = None

    ############################################################################
    # LOSO Evaluation
    ############################################################################
    if run_loso:
        log("\n==== Running Leave-One-Subject-Out Evaluation ====")
        loso_accuracies = {}
        for subj in tqdm(unique_subjects, desc="LOSO evaluation"):
            test_mask = (subject_ids == subj)
            train_mask = ~test_mask
            trainR = trials_R[train_mask].tolist()
            trainType = trials_Type[train_mask].tolist()
            testR = trials_R[test_mask].tolist()
            testType = trials_Type[test_mask].tolist()
            
            # Count the number of subjects and trials in training
            training_subjects = np.unique(subject_ids[train_mask])
            num_training_subjects = len(training_subjects)
            num_training_trials = len(trainR)
            
            # Count the number of trials for testing subject
            num_testing_trials = len(testR)
            
            log(f"\nLOSO - Testing on Subject {subj} ({num_testing_trials} trials)")
            log(f"  Training on {num_training_subjects} subjects ({num_training_trials} trials): {', '.join(training_subjects)}")
            
            csp_clf = CSPClassifier(data_channel, N, classifier_type=classifier_type)
            csp_clf.fit(trainR, trainType)
            acc, _ = csp_clf.score(testR, testType)
            loso_accuracies[subj] = acc
            log(f"  Subject {subj}: LOSO Accuracy = {acc * 100:.2f}%")
            
        overall_loso = np.mean(list(loso_accuracies.values()))
        log(f"\nOverall LOSO Accuracy: {overall_loso * 100:.2f}%")
        results_summary['loso_accuracies'] = loso_accuracies
        results_summary['overall_loso_accuracy'] = overall_loso
    else:
        log("\n==== Skipping Leave-One-Subject-Out Evaluation ====")
        results_summary['loso_accuracies'] = {}
        results_summary['overall_loso_accuracy'] = None

    ############################################################################
    # Intra-Subject Cross Validation (aggregated across sessions)
    ############################################################################
    if run_intra_subject:
        log("\n==== Running Intra-Subject Cross Validation ====")
        intra_subject_accuracies = {}
        for subj in tqdm(unique_subjects, desc="Intra-subject evaluation"):
            subj_mask = (subject_ids == subj)
            subj_trials_R = trials_R[subj_mask]
            subj_trials_Type = trials_Type[subj_mask]
            n_trials_subj = len(subj_trials_R)
            if n_trials_subj < 2:
                log(f"Subject {subj} has too few trials for intra-subject evaluation.")
                continue
            k_intra = min(5, n_trials_subj)
            kf_intra = KFold(n_splits=k_intra, shuffle=True, random_state=42)
            subj_acc = []
            for train_index, test_index in kf_intra.split(range(n_trials_subj)):
                trainR_intra = np.array(subj_trials_R, dtype=object)[train_index].tolist()
                trainType_intra = np.array(subj_trials_Type)[train_index].tolist()
                testR_intra = np.array(subj_trials_R, dtype=object)[test_index].tolist()
                testType_intra = np.array(subj_trials_Type)[test_index].tolist()
                csp_clf = CSPClassifier(data_channel, N, classifier_type=classifier_type)
                csp_clf.fit(trainR_intra, trainType_intra)
                acc, _ = csp_clf.score(testR_intra, testType_intra)
                subj_acc.append(acc)
            intra_accuracy = np.mean(subj_acc)
            intra_subject_accuracies[subj] = intra_accuracy
            log(f"Subject {subj}: Intra-subject CV Accuracy = {intra_accuracy * 100:.2f}%")
        overall_intra = np.mean(list(intra_subject_accuracies.values()))
        log(f"Overall Intra-subject CV Accuracy: {overall_intra * 100:.2f}%")
        results_summary['intra_subject_accuracies'] = intra_subject_accuracies
        results_summary['overall_intra_subject_accuracy'] = overall_intra
    else:
        log("\n==== Skipping Intra-Subject Cross Validation ====")
        results_summary['intra_subject_accuracies'] = {}
        results_summary['overall_intra_subject_accuracy'] = None

    ############################################################################
    # Intra-Session Evaluation: Evaluate each session separately for each subject
    ############################################################################
    if run_intra_session:
        log("\n==== Running Intra-Session Evaluation ====")
        intra_session_accuracies = {}
        for subj in tqdm(unique_subjects, desc="Intra-session evaluation"):
            subject_folder = os.path.join(root_folder, subj)
            session_files = glob.glob(os.path.join(subject_folder, "*.mat"))
            session_accs = []
            for sess_file in session_files:
                trials_R_sess, trials_Type_sess = load_segmented_data(sess_file, Fs, data_channel, apply_ea)
                n_trials_sess = len(trials_R_sess)
                if n_trials_sess < 2:
                    log(f"Session {os.path.basename(sess_file)} of subject {subj} has too few trials.")
                    continue
                k_intra_sess = min(5, n_trials_sess)
                kf_intra_sess = KFold(n_splits=k_intra_sess, shuffle=True, random_state=42)
                sess_fold_accs = []
                for train_index, test_index in kf_intra_sess.split(range(n_trials_sess)):
                    trainR_sess = np.array(trials_R_sess, dtype=object)[train_index].tolist()
                    trainType_sess = np.array(trials_Type_sess)[train_index].tolist()
                    testR_sess = np.array(trials_R_sess, dtype=object)[test_index].tolist()
                    testType_sess = np.array(trials_Type_sess)[test_index].tolist()
                    csp_clf = CSPClassifier(data_channel, N, classifier_type=classifier_type)
                    csp_clf.fit(trainR_sess, trainType_sess)
                    acc, _ = csp_clf.score(testR_sess, testType_sess)
                    sess_fold_accs.append(acc)
                if sess_fold_accs:
                    sess_acc = np.mean(sess_fold_accs)
                    session_accs.append(sess_acc)
                    log(f"Subject {subj}, Session {os.path.basename(sess_file)}: Intra-session CV Accuracy = {sess_acc*100:.2f}%")
            if session_accs:
                subj_intra_sess_acc = np.mean(session_accs)
                intra_session_accuracies[subj] = subj_intra_sess_acc
                log(f"Subject {subj}: Overall Intra-session CV Accuracy = {subj_intra_sess_acc*100:.2f}%")
        overall_intra_session = np.mean(list(intra_session_accuracies.values()))
        log(f"Overall Intra-session CV Accuracy (all subjects): {overall_intra_session*100:.2f}%")
        results_summary['intra_session_accuracies'] = intra_session_accuracies
        results_summary['overall_intra_session_accuracy'] = overall_intra_session
    else:
        log("\n==== Skipping Intra-Session Evaluation ====")
        results_summary['intra_session_accuracies'] = {}
        results_summary['overall_intra_session_accuracy'] = None

    ############################################################################
    # Incremental Subject LOSO Evaluation
    ############################################################################
    if run_incremental_loso:
        log("\n==== Running Incremental Subject LOSO Evaluation ====")
        min_train_subjects = 2  # Minimum number of subjects for training
        incremental_loso_results = incremental_loso_evaluation(
            trials_R, trials_Type, subject_ids, 
            data_channel, N, classifier_type, 
            min_subjects=min_train_subjects,
            log_func=log
        )
        results_summary['incremental_loso_results'] = incremental_loso_results
    else:
        log("\n==== Skipping Incremental Subject LOSO Evaluation ====")
        results_summary['incremental_loso_results'] = {}

    ############################################################################
    # Save logs and summary results in a folder with the current date, classifier type, and EA status e.g EA_CSP_LDA
    ############################################################################
    #create one folder called classification_results and if it exists, skip creating it and save the logs in it but not inside root_folder
    # Create a folder for classification results
    classification_results_folder = "classification_results"
    os.makedirs(classification_results_folder, exist_ok=True)
    # Create a folder with the current date and classifier type
    # and EA status e.g. logs_20231001_123456_EA_CSP_LDA
    from datetime import datetime
    # Create a folder with the current date
    classifier_name = classifier_type
    ea_status = "EA" if apply_ea else "No_EA"
    log_folder = os.path.join(classification_results_folder, f"logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{ea_status}_{classifier_name}")
    os.makedirs(log_folder, exist_ok=True)
    log_file = os.path.join(log_folder, "evaluation_log.txt")
    with open(log_file, "w") as f:
        for line in log_lines:
            f.write(line + "\n")
    print(f"Log saved to {log_file}")

    # Save summary results to a txt file in the same folder 
    summary_file = os.path.join(log_folder, "summary_results.txt")
    with open(summary_file, "w") as f:
        f.write(f"inter_subject={results_summary.get('inter_subject_accuracy', 'Not evaluated')}\n")
        f.write(f"loso={results_summary.get('loso_accuracies', {})}\n")
        f.write(f"overall_loso={results_summary.get('overall_loso_accuracy', 'Not evaluated')}\n")
        f.write(f"intra_subject={results_summary.get('intra_subject_accuracies', {})}\n")
        f.write(f"overall_intra_subject={results_summary.get('overall_intra_subject_accuracy', 'Not evaluated')}\n")
        f.write(f"intra_session={results_summary.get('intra_session_accuracies', {})}\n")
        f.write(f"overall_intra_session={results_summary.get('overall_intra_session_accuracy', 'Not evaluated')}\n")
        f.write(f"subject_details={subj_details}\n")
        f.write(f"incremental_loso={results_summary.get('incremental_loso_results', {})}\n")
    
    # Additionally, save a CSV file with the incremental LOSO results for easier plotting
    if run_incremental_loso and incremental_loso_results:
        incremental_csv = os.path.join(log_folder, "incremental_loso_results.csv")
        with open(incremental_csv, "w") as f:
            f.write("num_train_subjects,mean_accuracy,std_accuracy\n")
            for size, result in incremental_loso_results.items():
                f.write(f"{size},{result['mean_accuracy']},{result['std_accuracy']}\n")
        log(f"Incremental LOSO results saved to {incremental_csv}")
        
        # Also save individual accuracies for potential box plots
        incremental_detailed_csv = os.path.join(log_folder, "incremental_loso_detailed.csv")
        with open(incremental_detailed_csv, "w") as f:
            f.write("num_train_subjects,combination_idx,accuracy\n")
            for size, result in incremental_loso_results.items():
                for i, acc in enumerate(result['all_accuracies']):
                    f.write(f"{size},{i},{acc}\n")
        log(f"Detailed incremental LOSO results saved to {incremental_detailed_csv}")
    
    print("Summary results saved to summary_results.txt")

if __name__ == '__main__':
    main()

