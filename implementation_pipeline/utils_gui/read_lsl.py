#!/usr/bin/env python3
"""
What: This module provides EEGTrialStreamer, which connects once to an LSL EEG stream,
     selects only the specified channels, then on demand segments a fixed-length trial
     starting from the moment you call get_trial() (no old data), applies a 50 Hz
     notch and 8–32 Hz bandpass (zero-phase), downsamples from 500 Hz to 100 Hz,
     applies a multi‐band spectral‐fusion filterbank (θ, α, β, full), computes time‐frequency
     spectrogram features on the fused signal, concatenates them with the fused time‐series,
     and returns each trial as an array of shape (channels, timepoints + tf_feature_bins).
"""

import logging
import numpy as np
from pylsl import StreamInlet, resolve_streams, resolve_byprop
from scipy.signal import butter, iirnotch, sosfiltfilt, filtfilt, spectrogram
from joblib import Parallel, delayed

# name of the LabStreamingLayer stream
# STREAM_NAME    = "iReW32_73"  #for wifi based
STREAM_NAME = "iReUSB32_32"   #for usb based

# sampling rates (Hz)
ORIGINAL_RATE  = 500.0   # incoming
TARGET_RATE    = 100.0   # after downsampling

# trial timing (s)
TRIAL_DURATION = 4.0

# Butterworth filter settings
FILTER_ORDER   = 5
BANDPASS_FREQS = (8.0, 32.0)
NOTCH_FREQ     = 50.0
NOTCH_Q        = 30.0

# filterbank bands (Hz)
FILTERBANK_BANDS = [
(4.0, 7.0),    # theta
(7.0, 13.0),   # alpha
(13.0, 32.0),  # beta
(1.0, 40.0)    # full
]

# spectrogram parameters
NPERSEG   = 128
NOVERLAP  = 64

# which EEG channels to keep
DESIRED_CHANNELS = {
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
    """
    Design the 50 Hz notch and the 8–32 Hz band‑pass Butterworth filters
    at the original (500 Hz) sampling rate.

    Returns
    -------
    b_notch, a_notch : ndarray
        Coefficients of the IIR notch filter for `scipy.signal.filtfilt`.
    sos_bp : ndarray
        Second‑order‑sections representation of the band‑pass filter for
        `scipy.signal.sosfiltfilt`.
    """
    # 1. 50 Hz notch (IIR)
    b_notch, a_notch = iirnotch(NOTCH_FREQ, NOTCH_Q, fs=ORIGINAL_RATE)

    # 2. 8–32 Hz band‑pass (5th‑order Butterworth, SOS form)
    sos_bp = butter(
        FILTER_ORDER,
        BANDPASS_FREQS,
        btype='bandpass',
        fs=ORIGINAL_RATE,
        output='sos'
    )

    return b_notch, a_notch, sos_bp



def process_block(raw_block: np.ndarray,
                  b_notch: np.ndarray,
                  a_notch: np.ndarray,
                  sos_bp: np.ndarray) -> np.ndarray:
    """
    Convert one 4‑s raw EEG trial (500 Hz) into the augmented feature
    representation used by the CustomEpoch reference pipeline.

    Pipeline
    --------
    1) 50 Hz notch  ➜  8–32 Hz band‑pass  (both zero‑phase) @500 Hz
    2) Down‑sample to 100 Hz (integer decimation × 5)
    3) Multi‑band spectral‑fusion filterbank (θ, α, β, full)
    4) Channel‑wise spectrogram (magnitude) on the fused signal
    5) Concatenate fused time‑series with TF features and z‑score
       per channel.

    Returns
    -------
    X_aug : ndarray, shape (channels,
                            n_time_points + freq_bins × time_bins)
        Augmented feature matrix for the trial.
    """
    # ---------------------------------------------------------------
    # 1. 50 Hz notch  ➜  8–32 Hz band‑pass  (both zero‑phase) @500 Hz
    # ---------------------------------------------------------------
    data_notch = filtfilt(b_notch, a_notch, raw_block, axis=0)
    data_bp    = sosfiltfilt(sos_bp, data_notch, axis=0)

    # ---------------------------------------------------------------
    # 2. Down‑sample 500 → 100 Hz (decimate by 5)
    # ---------------------------------------------------------------
    decim   = int(round(ORIGINAL_RATE / TARGET_RATE))   # = 5
    data_ds = data_bp[::decim, :]                       # (n_ds, ch)

    # ---------------------------------------------------------------
    # 3. Multi‑band spectral‑fusion filterbank
    # ---------------------------------------------------------------
    nyq_fb  = TARGET_RATE / 2.0                         # 50 Hz
    X_fused = np.zeros_like(data_ds)

    for low, high in FILTERBANK_BANDS:
        b_fb, a_fb = butter(
            FILTER_ORDER,
            [low / nyq_fb, high / nyq_fb],
            btype='band'
        )
        X_fused += filtfilt(b_fb, a_fb, data_ds, axis=0)

    X_fused /= float(len(FILTERBANK_BANDS))             # average bands

    # ---------------------------------------------------------------
    # 4. Channel‑wise spectrogram (magnitude)
    # ---------------------------------------------------------------
    _, _, S0   = spectrogram(
        X_fused[:, 0],
        fs=TARGET_RATE,
        nperseg=NPERSEG,
        noverlap=NOVERLAP
    )
    freq_bins, time_bins = S0.shape                     # reference dims

    def _spectro(x):
        return spectrogram(
            x,
            fs=TARGET_RATE,
            nperseg=NPERSEG,
            noverlap=NOVERLAP
        )[2]

    sxx_list = Parallel(n_jobs=-1)(
        delayed(_spectro)(X_fused[:, ch]) for ch in range(X_fused.shape[1])
    )
    tf_feats = np.stack(sxx_list)                       # (ch, f, t)
    tf_flat  = tf_feats.reshape(X_fused.shape[1], -1)   # (ch, f*t)

    # ---------------------------------------------------------------
    # 5. Concatenate time‑series + TF; z‑score per channel
    # ---------------------------------------------------------------
    X_time = X_fused.T                                  # (ch, time)
    X_aug  = np.concatenate([X_time, tf_flat], axis=1)  # (ch, feats)

    median = np.median(X_aug, axis=1, keepdims=True)
    stdev  = np.std   (X_aug, axis=1, keepdims=True) + 1e-8
    X_aug  = (X_aug - median) / stdev

    return X_aug


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
        self.keep_idx    = []
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
        n_samples = int(round(t * ORIGINAL_RATE))
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
        raw = np.array(buffer[:n_samples])      # (n_samples, all_channels)
        return raw[:, self.keep_idx]            # select desired channels

    def get_trial(self, t=TRIAL_DURATION):
        """
        Flush older samples, then batch-read exactly t*ORIGINAL_RATE samples
        and return the processed trial plus the raw block.
        """
        raw_block = self._collect_raw(t)
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
        proc, raw = streamer.get_trial(t=4.0)
        logger.info(f"Trial {idx+1} processed shape: {proc.shape}, raw shape: {raw.shape}")
