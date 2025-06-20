import sys, os, threading
import logging
import time
# allow LSL import from uncle directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from datetime import datetime
from utils_gui.read_lsl import EEGTrialStreamer, process_block
from utils_gui.trial_window_ui import TrialWindow
from utils_gui.constants import (
    show_rest_duration, show_fixation_duration,
    show_stimulus_duration, TRIAL_DURATION
)
from PySide6.QtCore import QObject, Signal, QTimer, Slot  # added Slot
from control_exoskeleton import ControlExoskeleton, UP, DOWN

# configure root logger once
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class TrialManager(QObject):
    # emits (trial_index, fixation_segment_array)
    fixation_data_ready = Signal(int, object)
    prediction_started = Signal()

    def __init__(self, ui, runner):              # added runner param
        super().__init__()
        self.ui = ui
        self.runner = runner                     # store runner
        self.streamer = EEGTrialStreamer()
        self.counter = {}
        self.trials_data = []      # collect full trial arrays
        self.fixations_data = []   # collect 4s fixation segments

        # Exoskeleton setup
        self.exo = None
        ch340_port = ControlExoskeleton.find_ch340_port()
        if ch340_port:
            self.exo = ControlExoskeleton(ch340_port)
        else:
            logger.warning("CH340 device for exoskeleton not found.")

        # ensure scheduling runs in Qt thread
        self.fixation_data_ready.connect(self._schedule_prediction)
        self.prediction_started.connect(self._on_prediction_started)

    def launch(self):
        num = int(self.ui.ui.NumTrialsBox.text().strip() or "1")
        self.counter["idx"] = 0
        self.total = num
        # create and wire up a single TrialWindow
        self.trial = TrialWindow("left")
        self.trial.rest1_started.connect(self._on_rest1)
        self.trial.fixation_started.connect(self._log_fixation)
        self.trial.stimulus_started.connect(self._log_stimulus)     # <— new
        # when this single‐trial sequence finishes, start the next
        self.trial.trial_finished.connect(self._on_trial_finished)
        # start first trial and immediately kick off full‐trial save
        self.trial.set_trial_counter(1, num)
        self.trial.start()
        self._read_full_trial(1)

    def _on_trial_finished(self):
        """After trial ends, move to next trial or close."""
        self.counter["idx"] += 1
        if self.counter["idx"] < self.total:
            nxt = self.counter["idx"] + 1
            # wait 10 seconds (10000 ms) for action to complete, then start next trial
            QTimer.singleShot(10000, lambda nxt=nxt: (
                self._read_full_trial(nxt),
                self.trial.set_trial_counter(nxt, self.total),
                self.trial.start()
            ))
        else:
            self.trial.close()
            self.trial.deleteLater()

    def _read_full_trial(self, idx):
        """
        Collect the full trial (rest1+fix+stim) raw data
        in a background thread and save it as trial_{idx}_raw.npy.
        """
        def _collect():
            # sum durations (ms) → seconds
            t_full = (
                show_rest_duration +
                show_fixation_duration +
                show_stimulus_duration
            ) / 1000.0
            # get processed+raw, but we only save raw_block
            _, raw_block = self.streamer.get_trial(t=t_full)
            raw_fname = os.path.join(self.ui.out_dir, f"trial_{idx}_raw.npy")
            np.save(raw_fname, raw_block)
            self.trials_data.append(raw_block)
        threading.Thread(target=_collect, daemon=True).start()

    def _on_rest1(self):
        self._log_trial()
        # immediately read & process the 4s fixation window
        self._read_fixation(self.counter["idx"] + 1)

    def _read_fixation(self, idx):
        """Collect only the 4 s fixation window raw and process it right away."""
        def _collect():
            # TRIAL_DURATION is in ms; convert to seconds
            t_s = TRIAL_DURATION / 1000.0
            raw_block = self.streamer._collect_raw(t_s)  # (samples, channels)
            # process (expects shape (timepoints, channels)), no .T here
            fix_data = process_block(
                raw_block,
                self.streamer.b_notch, self.streamer.a_notch, self.streamer.sos_bp
            )
            # save processed fixation
            fix_fname = os.path.join(self.ui.out_dir, f"trial_{idx}_fixation.npy")
            logger.info(f"Fixation data shape: {fix_data.shape}")
            np.save(fix_fname, fix_data)
            self.fixations_data.append(fix_data)
            # emit to Qt thread
            self.fixation_data_ready.emit(idx, fix_data)

        threading.Thread(target=_collect, daemon=True).start()

    @Slot(int, object)
    def _schedule_prediction(self, idx, fix_data):
        """Run prediction after stimulus duration from the Qt event loop."""
        QTimer.singleShot(
            show_stimulus_duration,
            lambda: threading.Thread(
                target=self._predict_on_fixation,
                args=(idx, fix_data),
                daemon=True
            ).start()
        )

    @Slot()
    def _on_prediction_started(self):
        """Show prediction message in the trial window."""
        self.trial.show_message("Prediction")

    # new helper to run ensemble prediction
    def _predict_on_fixation(self, idx, fix_data):
        self.prediction_started.emit()
        # fix_data is shape (channels, timepoints); runner.predict expects 2D (C, T)
        trial = fix_data
        pre_lbl, tta_lbl, avg_pre, avg_tta, p_pre, p_tta = self.runner.predict(trial)

        log_parts = []
        if self.runner.mode in ["pre_tta", "both"]:
            log_parts.append(f"Pre-TTA:{pre_lbl}, avg_pre={avg_pre}")
        if self.runner.mode in ["tta", "both"]:
            log_parts.append(f"TTA:{tta_lbl}, avg_tta={avg_tta}")

        logger.info(f"Trial {idx} prediction → " + ", ".join(log_parts))

        # Control exoskeleton based on prediction
        label_to_use = None
        if self.runner.mode in ["tta", "both"]:
            label_to_use = tta_lbl
        elif self.runner.mode == "pre_tta":
            label_to_use = pre_lbl

        if self.exo and label_to_use == 1:
            logger.info(f"Label is 1, moving exoskeleton for trial {idx}.")
            self.exo.send_hex(UP)
            time.sleep(4.5)
            self.exo.send_hex(DOWN)
            time.sleep(4.5)
        elif self.exo and label_to_use == 0:
            logger.info(f"Label is 0, not moving exoskeleton for trial {idx}.")
        elif not self.exo:
            logger.warning("Exoskeleton not connected, skipping movement.")

    def _log_trial(self):
        # include trial ID (1-based) in the log
        idx = self.counter.get("idx", 0) + 1
        # log timestamped message via logger
        logger.info(f"Trial {idx} started")

    def _log_fixation(self):
        # include trial ID (1-based) in the log
        idx = self.counter.get("idx", 0) + 1
        logger.info(f"Trial {idx} – Fixation started")

    def _log_stimulus(self):     # <— new
        idx = self.counter.get("idx", 0) + 1
        logger.info(f"Trial {idx} – Stimulus started")

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
    end_s:   float, end time in seconds
    fs:      sampling rate in Hz (default 100)
    """
    start_s = show_rest_duration / 1000.0
    end_s   = start_s + TRIAL_DURATION
    return segment_trial(data, start_s, end_s, fs)
    end_idx   = int(end_s   * fs)
    return data[:, start_idx:end_idx]

def segment_fixation_window(data, fs=100.0):
    """
    Return 4 seconds of data starting at fixation onset.
    Assumes show_rest_duration (ms) marks the end of rest1.
    end_s:   float, end time in seconds
    fs:      sampling rate in Hz (default 100)
    """
    start_s = show_rest_duration / 1000.0
    end_s   = start_s + TRIAL_DURATION
    return segment_trial(data, start_s, end_s, fs)
