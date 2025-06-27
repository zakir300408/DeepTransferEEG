import sys, os, threading
import logging
import time
import random
import json
import pyedflib
# allow LSL import from uncle directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from datetime import datetime
from utils_gui.read_lsl import EEGTrialStreamer, process_block, ORIGINAL_RATE
from ui.trial_window_ui import TrialWindow
from utils_gui.constants import (
    show_rest_duration, show_fixation_duration,
    show_stimulus_duration, TRIAL_DURATION, ArmMovementDuration
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
        self.stimuli_sequence = []
        self.movement_events = []
        self.trial_results = []

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

        # Create and shuffle stimuli sequence - binary: stimulus or no stimulus
        num_stimulus = num // 2  # Half with stimulus
        num_none = num - num_stimulus  # Half without stimulus
        self.stimuli_sequence = ['stimulus'] * num_stimulus + ['none'] * num_none
        random.seed(42)  # for reproducibility
        random.shuffle(self.stimuli_sequence)
        logger.info(f"Stimuli sequence for {num} trials: {self.stimuli_sequence}")

        # Pre-populate trial results with stimulus and ground truth
        self.trial_results = [
            {
                "trial_index": i + 1,
                "stimulus": self.stimuli_sequence[i],
                "ground_truth": 1 if self.stimuli_sequence[i] == 'stimulus' else 0,
                "predicted_label": None,
            }
            for i in range(num)
        ]

        self.movement_events = [threading.Event() for _ in range(num)]

        # create and wire up a single TrialWindow
        self.trial = TrialWindow()
        self.trial.rest1_started.connect(self._on_rest1)
        self.trial.fixation_started.connect(self._log_fixation)
        self.trial.stimulus_started.connect(self._log_stimulus)     # <— new
        # when this single‐trial sequence finishes, start the next
        self.trial.trial_finished.connect(self._on_trial_finished)
        # start first trial
        self._prepare_and_start_trial(self.counter['idx'])

    def _prepare_and_start_trial(self, trial_idx):
        """Configure symbol then kick off trial #{trial_idx+1}."""
        # update counter in UI
        self.trial.set_trial_counter(trial_idx + 1, self.total)

        # pick box vs. cross based on your ground‐truth sequence
        from utils_gui.constants import Cross_Symbol, Stimulus_Symbol
        want = self.stimuli_sequence[trial_idx]
        sym = Stimulus_Symbol if want == 'stimulus' else Cross_Symbol
        self.trial.set_stimulus_symbol(sym)

        # now run the trial
        self.trial.start()
        self._read_full_trial(trial_idx + 1)

    def _on_trial_finished(self):
        """After trial UI ends, wait for exoskeleton movement to finish, then proceed."""
        current_idx = self.counter['idx']
        self.wait_timer = QTimer()
        self.wait_timer.timeout.connect(lambda: self._check_if_ready_for_next_trial(current_idx))
        self.wait_timer.start(100)  # Check every 100ms

    def _check_if_ready_for_next_trial(self, finished_trial_idx):
        """Slot for the wait_timer. When exo is done, advance to the next trial."""
        if self.movement_events[finished_trial_idx].is_set():
            self.wait_timer.stop()
            self.counter["idx"] += 1
            if self.counter["idx"] < self.total:
                self._prepare_and_start_trial(self.counter['idx'])
            else:
                self._finalize_experiment()
                self.trial.close()
                self.trial.deleteLater()

    def _finalize_experiment(self):
        """Calculate accuracy and save final results at the end of the experiment."""
        predicted_labels = [r["predicted_label"] for r in self.trial_results if r["predicted_label"] is not None]
        if not predicted_labels:
            logger.warning("No predictions were made, cannot calculate accuracy.")
            return

        ground_truths = [r["ground_truth"] for r in self.trial_results if r["predicted_label"] is not None]
        correct_predictions = sum(1 for gt, pred in zip(ground_truths, predicted_labels) if gt == pred)
        accuracy = (correct_predictions / len(predicted_labels)) * 100
        logger.info(f"Final Accuracy: {accuracy:.2f}% ({correct_predictions}/{len(predicted_labels)})")

        # Add final accuracy to the results and save one last time
        final_data = {
            "trials": self.trial_results,
            "final_accuracy_percent": accuracy
        }
        results_path = os.path.join(self.ui.out_dir, "trial_results.json")
        try:
            with open(results_path, "w", encoding="utf-8") as f:
                json.dump(final_data, f, indent=4)
            logger.info(f"Final results with accuracy saved to {results_path}")
        except Exception as e:
            logger.error(f"Could not write final trial results JSON: {e}")

    def _save_trial_results(self):
        """Saves the current trial results to a JSON file."""
        results_path = os.path.join(self.ui.out_dir, "trial_results.json")
        try:
            with open(results_path, "w", encoding="utf-8") as f:
                json.dump(self.trial_results, f, indent=4)
            logger.info(f"Updated trial results saved to {results_path}")
        except Exception as e:
            logger.error(f"Could not write trial results JSON: {e}")

    def _read_full_trial(self, idx):
        """
        Collect the full trial (rest1+fix+stim) raw data
        in a background thread and save it as trial_{idx}_raw.edf.
        """
        def _collect():
            # Total trial duration in seconds
            t_full = (
                show_rest_duration +
                show_fixation_duration +
                show_stimulus_duration
            ) / 1000.0

            # Acquire raw data block
            _, raw_block = self.streamer.get_trial(t=t_full)

            # Prepare EDF filename and data array (n_channels x n_samples)
            edf_fname = os.path.join(self.ui.out_dir, f"trial_{idx}_raw.edf")
            data = raw_block.T
            n_channels = data.shape[0]

            # Get 16-bit integer limits programmatically
            info = np.iinfo(np.int16)
            digital_min, digital_max = int(info.min), int(info.max)

            # Build headers with dynamic rounding so each value fits ≤8 chars
            signal_headers = []
            for ch_idx, label in enumerate(self.streamer.kept_labels):
                chan = data[ch_idx]
                min_val = float(chan.min())
                max_val = float(chan.max())

                # If min and max are equal, adjust slightly to avoid EDF error
                if min_val == max_val:
                    epsilon = 1e-6 if min_val == 0 else abs(min_val) * 1e-6
                    phys_min = round(min_val - epsilon, 6)
                    phys_max = round(max_val + epsilon, 6)
                else:
                    # figure out how many decimals we can keep
                    int_min = str(int(min_val))
                    dec_min = max(0, 8 - len(int_min) - 1)
                    phys_min = round(min_val, dec_min)

                    int_max = str(int(max_val))
                    dec_max = max(0, 8 - len(int_max) - 1)
                    phys_max = round(max_val, dec_max)

                signal_headers.append({
                    'label':            label,
                    'dimension':        'uV',
                    'sample_frequency': ORIGINAL_RATE,
                    'physical_min':     phys_min,
                    'physical_max':     phys_max,
                    'digital_min':      digital_min,
                    'digital_max':      digital_max,
                    'transducer':       '',
                    'prefilter':        ''
                })

            # Write the EDF file
            f = None
            try:
                f = pyedflib.EdfWriter(edf_fname, n_channels, file_type=pyedflib.FILETYPE_EDFPLUS)
                f.setSignalHeaders(signal_headers)
                f.writeSamples(data)
                logger.info(f"EDF write succeeded: {edf_fname}")
            except Exception as e:
                logger.error(f"Could not write EDF file {edf_fname}: {e}")
            finally:
                if f is not None:
                    f.close()

            # Store raw data for later use
            self.trials_data.append(raw_block)

        # Run collection in background to keep UI responsive
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
        self.movement_events[idx - 1].clear()  # Prepare for waiting on this trial's movement
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

    def control_exoskeleton(self, idx, label):
        """Control the exoskeleton based on the predicted label."""
        if self.exo:
            if label == 1:
                logger.info(f"Label is 1, moving exoskeleton for trial {idx}.")
                self.exo.send_hex(UP)
                time.sleep(ArmMovementDuration)  # wait for arm to move up
                self.exo.send_hex(DOWN)
                time.sleep(ArmMovementDuration)  # wait for arm to move down
            elif label == 0:
                logger.info(f"Label is 0, not moving exoskeleton for trial {idx}.")
        else:
            logger.warning("Exoskeleton not connected, skipping movement.")


    # new helper to run ensemble prediction
    def _predict_on_fixation(self, idx, fix_data):
        try:
            self.prediction_started.emit()
            # fix_data is shape (channels, timepoints); runner.predict expects 2D (C, T)
            trial = fix_data
            pre_lbl, tta_lbl, avg_pre, avg_tta, p_pre, p_tta = self.runner.predict(trial)

            log_parts = []
            if self.runner.mode in ["pre_tta", "both"]:
                log_parts.append(f"Pre-TTA:{pre_lbl} (from P(class1)={avg_pre[1]:.4f} > 0.5), avg_probs={avg_pre}")
            if self.runner.mode in ["tta", "both"]:
                log_parts.append(f"TTA:{tta_lbl} (from P(class1)={avg_tta[1]:.4f} > 0.5), avg_probs={avg_tta}")

            logger.info(f"Trial {idx} prediction → " + ", ".join(log_parts))

            # Control exoskeleton based on prediction
            label_to_use = None
            if self.runner.mode in ["tta", "both"]:
                label_to_use = tta_lbl
            elif self.runner.mode == "pre_tta":
                label_to_use = pre_lbl

            # Update results for the current trial
            if label_to_use is not None:
                self.trial_results[idx - 1]["predicted_label"] = label_to_use

            # Save results incrementally after each prediction
            self._save_trial_results()

            # Control the exoskeleton based on the predicted label
            self.control_exoskeleton(idx, label_to_use)
        finally:
            self.movement_events[idx - 1].set()  # Signal that movement task is complete

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
    fs:      sampling rate in Hz (default 100)
    """
    start_s = show_rest_duration / 1000.0
    end_s   = start_s + TRIAL_DURATION
    return segment_trial(data, start_s, end_s, fs)
