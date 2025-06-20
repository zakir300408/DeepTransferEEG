#!/usr/bin/env python3
"""
What: This module provides EEGTrialStreamer, which connects once to an LSL EEG stream,
     selects only the specified channels, then on demand segments a fixed-length trial
     starting from the moment you call get_trial() (no old data), applies a 50 Hz
     notch and 8-32 Hz bandpass (zero-phase), downsamples from 500 Hz to 100 Hz,
     computes time-frequency spectrogram features, concatenates them with the
     time-series, and returns each trial as an array of shape
     (channels, timepoints + tf_feature_bins).
"""

import logging
import numpy as np
from pylsl import StreamInlet, resolve_streams, resolve_byprop
from scipy.signal import butter, iirnotch, sosfiltfilt, filtfilt, spectrogram
from joblib import Parallel, delayed

# ---------- constants (no magic numbers below) ----------
STREAM_NAME       = "iReW32_73"
ORIGINAL_RATE     = 500.0         # Hz of incoming stream
TARGET_RATE       = 100.0         # Hz after downsampling
TRIAL_DURATION    = 4.0           # default seconds per trial
FILTER_ORDER      = 4             # order for Butterworth filters
BANDPASS_FREQS    = (8.0, 32.0)   # Hz bandpass range
NOTCH_FREQ        = 50.0          # Hz line-noise notch
NOTCH_Q           = 30.0          # quality factor for notch
NPERSEG           = 128           # spectrogram segment length
NOVERLAP          = 64            # spectrogram overlap
DESIRED_CHANNELS  = {
    "FP1","FZ","F3","F7","FC5","FC1","C3","T7",
    "CP5","CP1","PZ","P3","P7","O1","O2","P4",
    "P8","CP6","CP2","CZ","C4","T8","FC6","FC2",
    "F4","F8","FP2"
}
EXPECTED_CHANNELS = len(DESIRED_CHANNELS)

# set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def design_filters():
    """Design the notch and bandpass filters at the original rate."""
    b_notch, a_notch = iirnotch(NOTCH_FREQ, NOTCH_Q, ORIGINAL_RATE)
    sos_bp = butter(
        FILTER_ORDER, BANDPASS_FREQS,
        btype="bandpass", fs=ORIGINAL_RATE,
        output="sos"
    )
    return b_notch, a_notch, sos_bp


def process_block(raw_block, b_notch, a_notch, sos_bp):
    """
    1) zero-phase notch + bandpass on raw (500 Hz)
    2) downsample to TARGET_RATE
    3) compute TF spectrogram per channel
    4) flatten and concatenate TF features with time-series
    Returns: array of shape (channels, timepoints + freq_bins * time_bins)
    """
    # 1) filter
    data_notch = filtfilt(b_notch, a_notch, raw_block, axis=0)
    data_bp    = sosfiltfilt(sos_bp, data_notch, axis=0)

    # 2) downsample by integer decimation
    decim = int(round(ORIGINAL_RATE / TARGET_RATE))
    data_ds = data_bp[::decim, :]  # (n_timepoints, n_channels)

    # prepare for spectrogram
    X = data_ds.T[np.newaxis, :, :]  # shape (1, C, T)
    sample_rate = TARGET_RATE

    # get freq/time-bin counts from one channel
    _, _, S0 = spectrogram(
        X[0, 0], fs=sample_rate,
        nperseg=NPERSEG, noverlap=NOVERLAP
    )
    freq_bins, time_bins = S0.shape

    # flatten channels for parallel TF
    flat_X = X.reshape(-1, X.shape[2])  # shape (C, T)

    def _compute_sxx(x):
        return spectrogram(
            x, fs=sample_rate,
            nperseg=NPERSEG, noverlap=NOVERLAP
        )[2]

    sxx_list = Parallel(n_jobs=-1)(
        delayed(_compute_sxx)(flat_X[k])
        for k in range(flat_X.shape[0])
    )

    tf_feats = (
        np.stack(sxx_list)
         .reshape(X.shape[0], X.shape[1], freq_bins, time_bins)
    )
    tf_flat = tf_feats.reshape(X.shape[0], X.shape[1], -1)

    # 4) concatenate along time axis
    X_aug = np.concatenate([X, tf_flat], axis=2)  # (1, C, T + F*T')
    return X_aug[0]  # (C, T + F*T')


class EEGTrialStreamer:
    def __init__(self, stream_name=STREAM_NAME, debug=False):
        # optionally list all streams
        if debug:
            logger.info("Available LSL streams:")
            for info in resolve_streams():
                logger.info(
                    f"  name={info.name()}, "
                    f"type={info.type()}, "
                    f"channels={info.channel_count()}"
                )

        # resolve named EEG stream once
        streams = resolve_byprop("name", stream_name)
        if not streams:
            raise RuntimeError(f"No EEG stream found with name '{stream_name}'")
        self.inlet = StreamInlet(streams[0])

        # read channel labels
        info      = self.inlet.info()
        n_ch      = info.channel_count()
        chan_desc = info.desc().child("channels").child("channel")
        all_labels = []
        for _ in range(n_ch):
            all_labels.append(chan_desc.child_value("label"))
            chan_desc = chan_desc.next_sibling()

        # select desired channels by label
        self.keep_idx = []
        self.kept_labels = []
        for i, lbl in enumerate(all_labels):
            if lbl.upper() in DESIRED_CHANNELS:
                self.keep_idx.append(i)
                self.kept_labels.append(lbl)

        labels_upper = {lbl.upper() for lbl in all_labels}
        missing = DESIRED_CHANNELS - labels_upper
        if missing:
            raise RuntimeError(f"Missing channels in stream: {missing}")
        if len(self.keep_idx) != EXPECTED_CHANNELS:
            raise RuntimeError(
                f"Channel count mismatch: expected {EXPECTED_CHANNELS}, got {len(self.keep_idx)}"
            )

        # design filters once
        self.b_notch, self.a_notch, self.sos_bp = design_filters()

    def _collect_raw(self, t):
        """
        Flush old samples and collect exactly t seconds of raw data,
        returning array of shape (n_samples, kept_channels).
        """
        n_samples = int(np.ceil(t * ORIGINAL_RATE))
        # flush buffered samples
        while True:
            chunk, _ = self.inlet.pull_chunk(timeout=0.0)
            if not chunk:
                break
        # collect until we have enough
        buffer = []
        while len(buffer) < n_samples:
            chunk, _ = self.inlet.pull_chunk(
                timeout=t + 0.1,
                max_samples=n_samples - len(buffer)
            )
            if chunk:
                buffer.extend(chunk)
        raw = np.array(buffer[:n_samples])                # (n_samples, all_channels)
        return raw[:, self.keep_idx]                      # select desired channels

    def get_trial(self, t=TRIAL_DURATION):
        """
        Flush older samples, then batch-read exactly t*ORIGINAL_RATE samples
        and return the processed trial.
        """
        # collect raw, then process
        raw_block = self._collect_raw(t)
        # process and return both processed and raw
        processed = process_block(
            raw_block,
            self.b_notch, self.a_notch, self.sos_bp
        )
        return processed, raw_block


if __name__ == "__main__":
    # example: get three back-to-back 4-second trials, each starting fresh
    logger.setLevel(logging.DEBUG)
    streamer = EEGTrialStreamer(debug=True)
    for idx in range(3):
        logger.info(f"Starting trial {idx+1}")
        trial = streamer.get_trial(t=4.0)
        logger.info(f"Trial {idx+1} shape: {trial.shape}")
        trial = streamer.get_trial(t=4.0)
        logger.info(f"Trial {idx+1} shape: {trial.shape}")
