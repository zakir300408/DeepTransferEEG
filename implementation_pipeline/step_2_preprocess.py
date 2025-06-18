import numpy as np
from scipy.signal import butter, iirnotch, filtfilt, spectrogram

# make these internal
def _init_filters(sample_rate: int):
    """
    Returns coefficients for notch (50Hz) and bandpass (8–32Hz).
    """
    nyq = sample_rate / 2
    bn, an = iirnotch(50.0/nyq, 30.0)
    b, a   = butter(5, [8/nyq, 32/nyq], btype='band')
    return (bn, an), (b, a)

def _init_tf_params(nperseg: int = 128, noverlap: int = 64):
    """
    Returns spectrogram parameters.
    """
    return nperseg, noverlap

def preprocess_trial(
    trial: np.ndarray,
    sample_rate: int
) -> np.ndarray:
    """
    Apply per-trial preprocessing:
      1. downsample 2× (200 Hz → 100 Hz)
      2. notch @50Hz, bandpass 8–32Hz
      3. time‐freq features (spectrogram flatten)
      4. per‐channel z‐score
    Args:
      trial: (n_ch, n_time_original)
      sample_rate: target sampling rate in Hz (after downsampling)
    Returns:
      proc_trial: (n_ch, n_time_downsampled + tf_feats)
    """
    
    tf_params = _init_tf_params()

    # --- notch + bandpass ---
    notch_filt, band_filt = _init_filters(sample_rate)
    bn, an = notch_filt
    b, a   = band_filt
    trial = filtfilt(bn, an, trial, axis=1)
    trial = filtfilt(b, a,   trial, axis=1)

    # --- time-frequency features ---
    nperseg, noverlap = tf_params
    ch, _ = trial.shape
    tf_list = []
    for c in range(ch):
        _, _, Sxx = spectrogram(trial[c], fs=sample_rate,
                                nperseg=nperseg, noverlap=noverlap)
        tf_list.append(Sxx.flatten())
    tf_feats = np.stack(tf_list)                # shape (ch, freq_bins*time_bins)
    trial = np.concatenate([trial, tf_feats], axis=1)

    # --- per-channel normalization ---
    mean = trial.mean(axis=1, keepdims=True)
    std  = trial.std(axis=1,  keepdims=True) + 1e-8
    trial = (trial - mean) / std

    return trial
