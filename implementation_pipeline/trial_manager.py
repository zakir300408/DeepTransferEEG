import sys
import os
import threading
import logging
import time
import random
import json
import pyedflib

# allow LSL import from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from datetime import datetime, timedelta
from utils_gui.read_lsl import EEGTrialStreamer, process_block, ORIGINAL_RATE
from ui.trial_window_ui import TrialWindow
from utils_gui.constants import (
    show_rest_duration,     # ms
    show_fixation_duration, # ms
    show_stimulus_duration, # ms
    TRIAL_DURATION,         # ms (window length)
    ArmMovementDuration,     # s
    DELAY_POST_STIMULUS      # ms
)
from PySide6.QtCore import QObject, Signal, QTimer, Slot
from control_exoskeleton import ControlExoskeleton, UP, DOWN

# configure root logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TrialManager(QObject):
    fixation_data_ready = Signal(int, object)
    prediction_started  = Signal()

    def __init__(self, ui, runner):
        super().__init__()
        self.ui               = ui
        self.runner           = runner
        self.streamer         = EEGTrialStreamer()
        self.counter          = {}
        self.trials_data      = []
        self.fixations_data   = []
        self.stimuli_sequence = []
        self.movement_events  = []
        self.trial_results    = []

        # Exoskeleton setup
        self.exo = None
        port = ControlExoskeleton.find_ch340_port()
        if port:
            self.exo = ControlExoskeleton(port)
        else:
            logger.warning("CH340 device for exoskeleton not found.")

        # connect prediction scheduler
        self.fixation_data_ready.connect(self._schedule_prediction)
        self.prediction_started.connect(self._on_prediction_started)

    def launch(self):
        num = int(self.ui.ui.NumTrialsBox.text().strip() or "1")
        self.counter["idx"] = 0
        self.total          = num

        # prepare stimuli sequence
        half = num // 2
        self.stimuli_sequence = ['stimulus'] * half + ['none'] * (num - half)
        random.seed(42)
        random.shuffle(self.stimuli_sequence)
        logger.info(f"Stimuli sequence ({num} trials): {self.stimuli_sequence}")

        # prepopulate results
        self.trial_results = [
            {
                "trial_index":     i + 1,
                "stimulus":        self.stimuli_sequence[i],
                "ground_truth":    1 if self.stimuli_sequence[i] == 'stimulus' else 0,
                "predicted_label": None,
            }
            for i in range(num)
        ]

        self.movement_events = [threading.Event() for _ in range(num)]

        # set up UI signals
        self.trial = TrialWindow()
        self.trial.rest1_started.connect(self._on_rest1)
        self.trial.fixation_started.connect(self._log_fixation)
        self.trial.stimulus_started.connect(self._on_stimulus)   # trigger delayed segmentation
        self.trial.trial_finished.connect(self._on_trial_finished)

        # start first trial
        self._prepare_and_start_trial(0)

    def _log_trial(self):
        idx = self.counter.get("idx", 0) + 1
        logger.info(f"[Trial {idx}] Rest1 started at {datetime.now().time()}")

    def _log_fixation(self):
        idx = self.counter.get("idx", 0) + 1
        logger.info(f"[Trial {idx}] Fixation started at {datetime.now().time()}")

    def _on_rest1(self):
        # called at t=0 when rest starts
        self._log_trial()

    def _on_stimulus(self):
        # called at t = rest+fixation
        idx = self.counter.get("idx", 0) + 1
        now = datetime.now().time()
        logger.info(f"[Trial {idx}] Stimulus started at {now}")
        # wait 500 ms into the stimulus period before grabbing data
        delay_ms = DELAY_POST_STIMULUS
        logger.info(f"[Trial {idx}] → Scheduling segmentation in {delay_ms}ms (+0.5s)")
        QTimer.singleShot(
            delay_ms,
            lambda: (
                logger.info(f"[Trial {idx}] → Beginning segmentation at {datetime.now().time()}"),
                self._read_fixation(idx)
            )
        )

    def _prepare_and_start_trial(self, trial_idx):
        self.trial.set_trial_counter(trial_idx + 1, self.total)
        from utils_gui.constants import Cross_Symbol, Stimulus_Symbol
        want = self.stimuli_sequence[trial_idx]
        sym  = Stimulus_Symbol if want == 'stimulus' else Cross_Symbol
        self.trial.set_stimulus_symbol(sym)

        logger.info(f"[Trial {trial_idx+1}] UI start at {datetime.now().time()}, symbol='{sym}'")
        self.trial.start()
        self._read_full_trial(trial_idx + 1)

    def _on_trial_finished(self):
        idx = self.counter["idx"]
        logger.info(f"[Trial {idx+1}] UI finished at {datetime.now().time()}, waiting for exo")
        self.wait_timer = QTimer()
        self.wait_timer.timeout.connect(lambda: self._check_if_ready_for_next_trial(idx))
        self.wait_timer.start(100)

    def _check_if_ready_for_next_trial(self, finished_idx):
        if self.movement_events[finished_idx].is_set():
            logger.info(f"[Trial {finished_idx+1}] Exo movement done at {datetime.now().time()}")
            self.wait_timer.stop()
            self.counter["idx"] += 1
            if self.counter["idx"] < self.total:
                self._prepare_and_start_trial(self.counter["idx"])
            else:
                self._finalize_experiment()
                self.trial.close()
                self.trial.deleteLater()

    def _finalize_experiment(self):
        preds = [r["predicted_label"] for r in self.trial_results if r["predicted_label"] is not None]
        if not preds:
            logger.warning("No predictions; skipping accuracy.")
            return

        gts     = [r["ground_truth"] for r in self.trial_results if r["predicted_label"] is not None]
        correct = sum(1 for gt, p in zip(gts, preds) if gt == p)
        acc     = (correct / len(preds)) * 100
        logger.info(f"Final accuracy: {acc:.2f}% ({correct}/{len(preds)})")

        out  = {"trials": self.trial_results, "final_accuracy_percent": acc}
        path = os.path.join(self.ui.out_dir, "trial_results.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=4)
        logger.info(f"Results saved to {path}")

    def _save_trial_results(self):
        path = os.path.join(self.ui.out_dir, "trial_results.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.trial_results, f, indent=4)
        logger.info(f"Intermediate results saved to {path}")

    def _read_full_trial(self, idx):
        def _collect():
            t_full = (
                show_rest_duration +
                show_fixation_duration +
                show_stimulus_duration
            ) / 1000.0
            logger.info(f"[Trial {idx}] _read_full_trial: collecting {t_full:.3f}s raw at {datetime.now().time()}")
            _, raw = self.streamer.get_trial(t=t_full)
            logger.info(f"[Trial {idx}] raw.shape={raw.shape}, expected={int(round(t_full*ORIGINAL_RATE))}")

            # write EDF...
            edf_fname = os.path.join(self.ui.out_dir, f"trial_{idx}_raw.edf")
            data       = raw.T
            n_ch       = data.shape[0]
            info       = np.iinfo(np.int16)
            dmin, dmax = int(info.min), int(info.max)
            gmin, gmax = float(data.min()), float(data.max())

            if gmin == gmax:
                eps    = 1e-6 if gmin == 0 else abs(gmin) * 1e-6
                pmin   = round(gmin - eps, 6)
                pmax   = round(gmax + eps, 6)
            else:
                im     = str(int(gmin))
                pmin   = round(gmin, max(0, 8-len(im)-1))
                gm     = str(int(gmax))
                pmax   = round(gmax, max(0, 8-len(gm)-1))

            headers = []
            for label in self.streamer.kept_labels:
                headers.append({
                    'label':            label,
                    'dimension':        'uV',
                    'sample_frequency': ORIGINAL_RATE,
                    'physical_min':     pmin,
                    'physical_max':     pmax,
                    'digital_min':      dmin,
                    'digital_max':      dmax,
                    'transducer':       '',
                    'prefilter':        ''
                })

            f = None
            try:
                f = pyedflib.EdfWriter(edf_fname, n_ch, file_type=pyedflib.FILETYPE_EDFPLUS)
                f.setDatarecordDuration(t_full)
                f.setSignalHeaders(headers)
                f.writeSamples(data)
                logger.info(f"[Trial {idx}] EDF written: {edf_fname}")
            except Exception as e:
                logger.error(f"[Trial {idx}] EDF write error {edf_fname}: {e}")
            finally:
                if f is not None:
                    f.close()

            self.trials_data.append(raw)

        threading.Thread(target=_collect, daemon=True).start()

    def _read_fixation(self, idx):
        """Collect exactly TRIAL_DURATION ms of data starting 0.5 s after stimulus onset."""
        def _collect():
            t_s   = TRIAL_DURATION / 1000.0
            start = datetime.now().time()
            logger.info(f"[Trial {idx}] _read_fixation: start raw collect at {start} for {t_s:.3f}s")
            raw_block = self.streamer._collect_raw(t_s)
            logger.info(f"[Trial {idx}] raw_block.shape={raw_block.shape}")
            fix_data  = process_block(raw_block, self.streamer.b_notch, self.streamer.a_notch, self.streamer.sos_bp)
            logger.info(f"[Trial {idx}] fix_data.shape after processing={fix_data.shape}")
            fname = os.path.join(self.ui.out_dir, f"trial_{idx}_fixation.npy")
            np.save(fname, fix_data)
            logger.info(f"[Trial {idx}] Saved fixation data to {fname}")
            self.fixations_data.append(fix_data)
            self.fixation_data_ready.emit(idx, fix_data)

        threading.Thread(target=_collect, daemon=True).start()

    @Slot(int, object)
    def _schedule_prediction(self, idx, fix_data):
        fire_time = datetime.now() + timedelta(milliseconds=show_stimulus_duration)
        logger.info(f"[Trial {idx}] _schedule_prediction: will fire at {fire_time.time()}")
        self.movement_events[idx - 1].clear()
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
        logger.info(f"[{datetime.now().time()}] Prediction message displayed")
        self.trial.show_message("Prediction")

    def control_exoskeleton(self, idx, label):
        logger.info(f"[Trial {idx}] control_exoskeleton: predicted label = {label}")
        if label == 1 and self.exo:
            logger.info(f"[Trial {idx}] Exo UP then DOWN")
            self.exo.send_hex(UP); time.sleep(ArmMovementDuration)
            self.exo.send_hex(DOWN); time.sleep(ArmMovementDuration)
        else:
            logger.info(f"[Trial {idx}] No movement, waiting {ArmMovementDuration*2}s")
            time.sleep(ArmMovementDuration * 2)

    def _predict_on_fixation(self, idx, fix_data):
        start = datetime.now().time()
        logger.info(f"[Trial {idx}] _predict_on_fixation started at {start}, input_shape={fix_data.shape}")
        self.prediction_started.emit()

        pre_lbl, tta_lbl, avg_pre, avg_tta, p_pre, p_tta = self.runner.predict(fix_data)
        logger.info(f"[Trial {idx}] prediction returned at {datetime.now().time()}")

        label_to_use = tta_lbl if self.runner.mode in ["tta", "both"] else pre_lbl
        self.trial_results[idx - 1]["predicted_label"] = label_to_use

        self._save_trial_results()
        self.control_exoskeleton(idx, label_to_use)

        self.movement_events[idx - 1].set()


# Optional segmentation helpers:
def segment_trial(data, start_s, end_s, fs=100.0):
    start_idx = int(start_s * fs)
    end_idx   = int(end_s   * fs)
    seg       = data[:, start_idx:end_idx]
    logger.info(f"[segment_trial] {start_s:.3f}s→{end_s:.3f}s => shape {seg.shape}")
    return seg

def segment_fixation_window(data, fs=100.0):
    start_s = (show_rest_duration + show_fixation_duration) / 1000.0
    end_s   = start_s + (TRIAL_DURATION / 1000.0)
    return segment_trial(data, start_s, end_s, fs)

def segment_poststim_window(data, fs=100.0, offset_ms=500, window_ms=TRIAL_DURATION):
    stim_onset_ms = show_rest_duration + show_fixation_duration
    start_s       = (stim_onset_ms + offset_ms) / 1000.0
    end_s         = start_s + (window_ms / 1000.0)
    return segment_trial(data, start_s, end_s, fs)
