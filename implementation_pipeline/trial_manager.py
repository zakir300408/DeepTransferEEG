import sys, os, threading
import logging
# allow LSL import from uncle directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from datetime import datetime
from utils_gui.read_lsl import EEGTrialStreamer, process_block, ORIGINAL_RATE
from utils_gui.trial_window_ui import TrialWindow
from utils_gui.constants import (
    show_rest_duration, show_fixation_duration,
    show_stimulus_duration, show_rest2_duration, TRIAL_DURATION
)
from PySide6.QtCore import QObject, Signal

# configure root logger once
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class TrialManager(QObject):
    # emits (trial_index, full_trial_array, fixation_segment_array)
    trial_data_ready = Signal(int, object, object)

    def __init__(self, ui, runner):              # added runner param
        super().__init__()
        self.ui = ui
        self.runner = runner                     # store runner
        self.streamer = EEGTrialStreamer()
        self.counter = {}
        self.trials_data = []      # collect full trial arrays
        self.fixations_data = []   # collect 4s fixation segments

    def launch(self):
        num = int(self.ui.ui.NumTrialsBox.text().strip() or "1")
        self.counter["idx"] = 0
        self.total = num
        # create and wire up a single TrialWindow
        self.trial = TrialWindow("left")
        self.trial.rest1_started.connect(self._on_rest1)
        self.trial.fixation_started.connect(self._log_fixation)
        # wait until the save/log thread finishes before starting the next trial
        self.trial_data_ready.connect(self._on_data_ready)
        # start first trial
        self.trial.set_trial_counter(1, num)
        self.trial.start()

    def _on_rest1(self):
        self._log_trial()
        self._read_and_save(self.counter["idx"] + 1)

    def _on_data_ready(self, idx, full_arr, fix_arr):
        # background work for trial 'idx' is done—start next or finish up
        self.counter["idx"] += 1
        if self.counter["idx"] < self.total:
            nxt = self.counter["idx"] + 1
            self.trial.set_trial_counter(nxt, self.total)
            self.trial.start()
        else:
            self.trial.close()
            self.trial.deleteLater()

    def _read_and_save(self, idx):
        # offload get_trial + saving so we don't block the UI thread
        def _collect():
            # compute total trial time in seconds
            t = (
                show_rest_duration +
                show_fixation_duration +
                show_stimulus_duration +
                show_rest2_duration
            ) / 1000.0
            # now returns (processed, raw_block)
            data, raw_block = self.streamer.get_trial(t=t)

            # save full trial (processed)
            full_fname = os.path.join(self.ui.out_dir, f"trial_{idx}.npy")
            np.save(full_fname, data)
            self.trials_data.append(data)

            # extract 4s fixation from raw_block at ORIGINAL_RATE
            fix_raw = segment_fixation_window(raw_block.T, fs=ORIGINAL_RATE)
            fix_data = process_block(
                fix_raw.T,
                self.streamer.b_notch, self.streamer.a_notch, self.streamer.sos_bp
            )

            # save processed fixation
            fix_fname = os.path.join(self.ui.out_dir, f"trial_{idx}_fixation.npy")
            # log shape of fixation data instead of print
            logger.info(f"Fixation data shape: {fix_data.shape}")
            np.save(fix_fname, fix_data)
            self.fixations_data.append(fix_data)
            self.trial_data_ready.emit(idx, data, fix_data)

            # dispatch prediction on fixation in separate thread
            threading.Thread(
                target=self._predict_on_fixation,
                args=(idx, fix_data),
                daemon=True
            ).start()

        threading.Thread(target=_collect, daemon=True).start()

    # new helper to run ensemble prediction
    def _predict_on_fixation(self, idx, fix_data):
        # fix_data is shape (channels, timepoints); runner.predict expects 2D (C, T)
        trial = fix_data
        pre_lbl, tta_lbl, avg_pre, avg_tta, p_pre, p_tta = self.runner.predict(trial)
        logger.info(
            f"Trial {idx} prediction → Pre-TTA:{pre_lbl}, TTA:{tta_lbl}, "
            f"avg_pre={avg_pre}, avg_tta={avg_tta}"
        )

    def _log_trial(self):
        # include trial ID (1-based) in the log
        idx = self.counter.get("idx", 0) + 1
        # log timestamped message via logger
        logger.info(f"Trial {idx} started")

    def _log_fixation(self):
        # include trial ID (1-based) in the log
        idx = self.counter.get("idx", 0) + 1
        logger.info(f"Trial {idx} – Fixation started")

def segment_trial(data, start_s, end_s, fs=100.0):
    """
    Truncate trial data between start_s and end_s (in seconds).

    data:    np.ndarray of shape (channels, timepoints)
    start_s: float, start time in seconds
    end_s:   float, end time in seconds
    fs:      sampling rate in Hz (default 100)
    """
    start_idx = int(start_s * fs)
    end_idx   = int(end_s   * fs)
    return data[:, start_idx:end_idx]

def segment_fixation_window(data, fs=100.0):
    """
    Return 4 seconds of data starting at fixation onset.
    Assumes show_rest_duration (ms) marks the end of rest1.
    """
    start_s = show_rest_duration / 1000.0
    end_s   = start_s + TRIAL_DURATION
    return segment_trial(data, start_s, end_s, fs)
