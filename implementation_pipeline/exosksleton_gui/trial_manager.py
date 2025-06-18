import sys, os, threading
# allow LSL import from uncle directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from implementation_main import EnsembleRunner

import numpy as np
from datetime import datetime
from read_w32.read_lsl import EEGTrialStreamer
from trial_window_ui import TrialWindow
from constants import (
    show_rest_duration, show_fixation_duration,
    show_stimulus_duration, show_rest2_duration
)
from PySide6.QtCore import QObject, Signal

class TrialManager(QObject):
    # emits (trial_index, full_trial_array, fixation_segment_array)
    trial_data_ready = Signal(int, object, object)

    def __init__(self, ui):
        super().__init__()
        self.ui = ui
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
        self.trial.trial_finished.connect(self._on_finished)
        # start first trial
        self.trial.set_trial_counter(1, num)
        self.trial.start()

    def _on_rest1(self):
        self._log_trial()
        self._read_and_save(self.counter["idx"] + 1)

    def _on_finished(self):
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
            data = self.streamer.get_trial(t=t)
            # save full trial
            full_fname = os.path.join(self.ui.out_dir, f"trial_{idx}.npy")
            np.save(full_fname, data)
            self.trials_data.append(data)
            # extract & save 4s fixation segment
            fix_data = segment_fixation_window(data)
            fix_fname = os.path.join(self.ui.out_dir, f"trial_{idx}_fixation.npy")
            np.save(fix_fname, fix_data)
            self.fixations_data.append(fix_data)

            # notify listeners that trial data is ready
            self.trial_data_ready.emit(idx, data, fix_data)

        threading.Thread(target=_collect, daemon=True).start()

    def _log_trial(self):
        print(f"Trial started at {datetime.now().isoformat()}")

    def _log_fixation(self):
        print(f"Fixation started at {datetime.now().isoformat()}")

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
    end_s   = start_s + 4.0
    return segment_trial(data, start_s, end_s, fs)
