#!/usr/bin/env python3
"""
What: This script connects once to an LSL EEG stream, selects only the specified
     channels, then in a loop (or on demand) segments fixed‐length trials,
     applies a 50 Hz notch and 8–32 Hz bandpass (zero-phase), downsamples from
     500 Hz to 100 Hz, computes time–frequency spectrogram features,
     concatenates them with the time-series, and returns each trial as an array
     of shape (channels, timepoints + tf_feature_bins).
"""

import logging
import numpy as np
from pylsl import StreamInlet, resolve_streams, resolve_byprop
from scipy.signal import butter, iirnotch, sosfiltfilt, filtfilt, spectrogram
from joblib import Parallel, delayed

# ---------- constants (no magic numbers below) ----------
STREAM_NAME         = "iReW32_73"
ORIGINAL_RATE       = 500.0        # Hz of incoming stream
TARGET_RATE         = 100.0        # Hz after downsampling
TRIAL_DURATION      = 4.0          # seconds per trial
FILTER_ORDER        = 4            # order for Butterworth filters
BANDPASS_FREQS      = (8.0, 32.0)  # Hz bandpass range
NOTCH_FREQ          = 50.0         # Hz line-noise notch
NOTCH_Q             = 30.0         # quality factor for notch
NPERSEG             = 128          # spectrogram segment length
NOVERLAP            = 64           # spectrogram overlap
DESIRED_CHANNELS    = {
    "FP1","FZ","F3","F7","FC5","FC1","C3","T7",
    "CP5","CP1","PZ","P3","P7","O1","O2","P4",
    "P8","CP6","CP2","CZ","C4","T8","FC6","FC2",
    "F4","F8","FP2"
}
EXPECTED_CHANNELS   = len(DESIRED_CHANNELS)

# set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class EEGStreamFilter:
    def __init__(self, stream_name, debug=False):
        # optionally list all available LSL streams
        if debug:
            logger.info("Available LSL streams:")
            for info in resolve_streams():
                logger.info(f"  name={info.name()}, type={info.type()}, channels={info.channel_count()}")

        # resolve the named EEG stream once
        streams = resolve_byprop('name', stream_name)
        if not streams:
            raise RuntimeError(f"No EEG stream found with name {stream_name}")
        inlet = StreamInlet(streams[0])

        # read channel labels in original stream order
        info      = inlet.info()
        n_ch      = info.channel_count()
        chan_desc = info.desc().child('channels').child('channel')
        all_labels = []
        for _ in range(n_ch):
            all_labels.append(chan_desc.child_value('label'))
            chan_desc = chan_desc.next_sibling()

        # select only the desired channels by label
        self.keep_idx        = [
            i for i, lbl in enumerate(all_labels)
            if lbl.upper() in DESIRED_CHANNELS
        ]
        self.filtered_labels = [all_labels[i] for i in self.keep_idx]

        # verify no channels are missing or extra
        missing = DESIRED_CHANNELS - {lbl.upper() for lbl in all_labels}
        if missing:
            raise RuntimeError(f"Missing channels in stream: {missing}")
        if len(self.keep_idx) != EXPECTED_CHANNELS:
            raise RuntimeError(
                f"Channel count mismatch: expected {EXPECTED_CHANNELS}, got {len(self.keep_idx)}"
            )

        self.inlet = inlet

    def read_sample(self):
        # pull one sample and return only the filtered channels
        raw, ts = self.inlet.pull_sample()
        filtered = np.array([raw[i] for i in self.keep_idx])
        return filtered, ts

    def read_block(self, n_samples):
        # pull n_samples in a block, returns shape (n_samples, n_channels)
        block = np.zeros((n_samples, EXPECTED_CHANNELS))
        for i in range(n_samples):
            sample, _ = self.read_sample()
            block[i, :] = sample
        return block

def design_filters():
    # design 50 Hz notch at original rate
    b_notch, a_notch = iirnotch(NOTCH_FREQ, NOTCH_Q, ORIGINAL_RATE)
    # design 8–32 Hz bandpass at original rate
    sos_bp = butter(
        FILTER_ORDER, BANDPASS_FREQS,
        btype='bandpass', fs=ORIGINAL_RATE,
        output='sos'
    )
    return b_notch, a_notch, sos_bp

def process_block(raw_block, b_notch, a_notch, sos_bp):
    """
    1) zero-phase notch + bandpass on raw (500 Hz)
    2) downsample to TARGET_RATE
    3) compute TF spectrogram per channel
    4) flatten and concatenate TF features with time-series
    """
    # 1) filter
    data_notch = filtfilt(b_notch, a_notch, raw_block, axis=0)
    data_bp    = sosfiltfilt(sos_bp,   data_notch, axis=0)

    # 2) downsample by slicing
    decim = int(round(ORIGINAL_RATE / TARGET_RATE))
    data_ds = data_bp[::decim, :]  # shape (n_timepoints, n_channels)

    # shape into (1, channels, timepoints) for uniform TF code
    X = data_ds.T[np.newaxis, :, :]  # (1, C, T)
    sample_rate = TARGET_RATE

    # 3) compute spectrogram dims
    _, _, S0 = spectrogram(
        X[0, 0], fs=sample_rate,
        nperseg=NPERSEG, noverlap=NOVERLAP
    )
    freq_bins, time_bins = S0.shape

    # flatten trials × channels → rows
    flat_X = X.reshape(-1, X.shape[2])  # (1*C, T)

    def _compute_sxx(x):
        return spectrogram(
            x, fs=sample_rate,
            nperseg=NPERSEG, noverlap=NOVERLAP
        )[2]  # Sxx matrix

    # parallel TF computation
    sxx_list = Parallel(n_jobs=-1)(
        delayed(_compute_sxx)(flat_X[k])
        for k in range(flat_X.shape[0])
    )

    tf_feats = np.stack(sxx_list) \
                 .reshape(X.shape[0], X.shape[1], freq_bins, time_bins)
    tf_flat  = tf_feats.reshape(X.shape[0], X.shape[1], -1)

    # 4) concatenate along time axis
    X_aug = np.concatenate([X, tf_flat], axis=2)  # (1,C, T+F*T')

    return X_aug[0]  # (C, T+F*T')

if __name__ == "__main__":
    # initialize once
    logger.setLevel(logging.DEBUG)
    stream    = EEGStreamFilter(STREAM_NAME, debug=True)
    b_notch, a_notch, sos_bp = design_filters()
    n_samples = int(np.ceil(TRIAL_DURATION * ORIGINAL_RATE))

    # example: collect and process 3 trials in a loop
    for trial_idx in range(3):
        logger.info(f"Starting trial {trial_idx+1}")
        raw       = stream.read_block(n_samples)
        trial_data = process_block(raw, b_notch, a_notch, sos_bp)
        logger.info(f"Trial {trial_idx+1} shape: {trial_data.shape}")
        # trial_data is (27, int(TRIAL_DURATION*TARGET_RATE) + freq_bins*time_bins)
