"""
Run the GUI‑driven experiment but **read LSL only once per trial**.
After recording the full trial (≈ 11.5 s at 500 Hz) we slice out the
4‑second fixation window directly from that array, eliminating the
jitter caused by a second, separate LSL call.

The fixation segment is:
    start =  rest(4 s) + fixation(1.5 s) + DELAY_POST_STIMULUS(0.5 s)
    dur   =  TRIAL_DURATION (4 s)

Saved files per trial:
    trial_<idx>_raw.npy           – full ~11.5 s recording
    trial_<idx>_raw_meta.json
    trial_<idx>_fixation_raw.npy  – 4 s segment (extracted, *not* re‑read)
    trial_<idx>_fixation_raw_meta.json
    trial_<idx>_fixation.npy      – processed (band‑pass + notch) segment
"""

import os, sys, json, random, threading, logging, time
from datetime import datetime
from PySide6.QtCore import QObject, Signal, QTimer, Slot
import numpy as np

# project‑local imports ---------------------------------------------------------
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils_gui.read_lsl import EEGTrialStreamer, process_block, ORIGINAL_RATE
from ui.trial_window_ui import TrialWindow
from utils_gui.constants import (
    show_rest_duration,
    show_fixation_duration,
    show_stimulus_duration,
    TRIAL_DURATION,
    ArmMovementDuration,
    DELAY_POST_STIMULUS,
    Prediction_Text
)
from control_exoskeleton import ControlExoskeleton, UP, DOWN

# logging ----------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ───────────────────────────────────────────────────────────────────────────────
class TrialManager(QObject):
    fixation_data_ready = Signal(int, object)  # idx, processed_segment
    prediction_started  = Signal()

    # ------------------------------------------------------------------
    def __init__(self, ui, runner):
        super().__init__()
        self.ui       = ui
        self.runner   = runner
        self.streamer = EEGTrialStreamer()
        self.arm_side = "left"  # default arm side

        self.counter          = {}   # current trial index (0‑based)
        self.trials_data      = []   # full recordings
        self.stimuli_sequence = []
        self.trial_results    = []
        self.movement_events  = []

        # exoskeleton ---------------------------------------------------
        self.exo = None
        port = ControlExoskeleton.find_ch340_port()
        if port:
            self.exo = ControlExoskeleton(port)
        else:
            logger.warning("CH340 device for exoskeleton not found.")

        # callbacks -----------------------------------------------------
        self.fixation_data_ready.connect(self._schedule_prediction)
        self.prediction_started.connect(self._on_prediction_started)

    # ------------------------------------------------------------------
    #                              PUBLIC
    # ------------------------------------------------------------------
    def launch(self):
        """Entry‑point called by the main GUI."""
        # Get arm side selection from UI
        self.arm_side = self.ui.get_arm_side()
        logger.info(f"Arm side selected: {self.arm_side}")
        
        num_trials = int(self.ui.ui.NumTrialsBox.text().strip() or "1")
        self.total = num_trials
        self.counter["idx"] = 0

        # balanced stimuli sequence ------------------------------------
        half = num_trials // 2
        self.stimuli_sequence = ['stimulus'] * half + ['none'] * (num_trials - half)
        random.seed(42); random.shuffle(self.stimuli_sequence)
        logger.info(f"Stimuli sequence ({num_trials} trials): {self.stimuli_sequence}")

        # prepare results container ------------------------------------
        self.trial_results = [
            {
                "trial_index":  i + 1,
                "stimulus":     self.stimuli_sequence[i],
                "ground_truth": 1 if self.stimuli_sequence[i] == 'stimulus' else 0,
                "predicted_label": None,
            }
            for i in range(num_trials)
        ]
        self.movement_events = [threading.Event() for _ in range(num_trials)]

        # UI window -----------------------------------------------------
        self.trial = TrialWindow()
        self.trial.rest1_started.connect(self._on_rest1)
        self.trial.fixation_started.connect(self._log_fixation)
        self.trial.stimulus_started.connect(self._log_stimulus)
        self.trial.trial_finished.connect(self._on_trial_finished)

        self._prepare_and_start_trial(0)

    # ------------------------------------------------------------------
    #                     TRIAL‑LEVEL ACTIONS & LOGS
    # ------------------------------------------------------------------
    def _prepare_and_start_trial(self, trial_idx: int):
        """Configure UI and kick off the Qt‑driven timeline."""
        self.trial.set_trial_counter(trial_idx + 1, self.total)
        from utils_gui.constants import Stimulus_Symbol, No_Stimulus_Symbol
        sym = Stimulus_Symbol if self.stimuli_sequence[trial_idx] == 'stimulus' else No_Stimulus_Symbol
        self.trial.set_stimulus_symbol(sym)
        logger.info(f"[Trial {trial_idx+1}] UI start at {datetime.now().time()}, symbol='{sym}'")

        self.trial.start()
        self._read_full_trial(trial_idx + 1)   # non‑blocking thread

    def _on_rest1(self):
        idx = self.counter["idx"] + 1
        logger.info(f"[Trial {idx}] Rest1 started at {datetime.now().time()}")

    def _log_fixation(self):
        idx = self.counter["idx"] + 1
        logger.info(f"[Trial {idx}] Fixation started at {datetime.now().time()}")

    def _log_stimulus(self):
        idx = self.counter["idx"] + 1
        logger.info(f"[Trial {idx}] Stimulus started at {datetime.now().time()}")

    # ------------------------------------------------------------------
    #                       LSL → single read per trial
    # ------------------------------------------------------------------
    def _read_full_trial(self, idx: int):
        """
        Collect *once* from LSL, then slice out the fixation segment
        instead of launching a second reader thread.
        """
        def _collect():
            t_full = (
                show_rest_duration +
                show_fixation_duration +
                show_stimulus_duration
            ) / 1000.0   # seconds
            logger.info(f"[Trial {idx}] Collecting {t_full:.3f}s raw ...")
            _, raw = self.streamer.get_trial(t=t_full)
            logger.info(f"[Trial {idx}] raw.shape = {raw.shape}")

            # ─────────────── persist full trial ────────────────
            self._save_full_trial(idx, raw)

            # ─────────────── slice fixation window ─────────────
            fs = ORIGINAL_RATE              # 500 Hz
            start_ms = (
                show_rest_duration +
                show_fixation_duration +
                DELAY_POST_STIMULUS
            )
            start_samp = int(round(start_ms / 1000 * fs))
            n_samp     = int(round(TRIAL_DURATION / 1000 * fs))
            fix_raw    = raw[start_samp : start_samp + n_samp, :]
            logger.info(f"[Trial {idx}] fixation slice: start_ms={start_ms}, "
                        f"start_samp={start_samp}, n_samp={n_samp}")

            self._save_fixation(idx, fix_raw)

            # push for prediction on a background thread
            threading.Thread(
                target=self._run_prediction_pipeline,
                args=(idx, fix_raw),
                daemon=True
            ).start()

        threading.Thread(target=_collect, daemon=True).start()

    # ------------------------------------------------------------------
    #                           SAVE HELPERS
    # ------------------------------------------------------------------
    def _save_full_trial(self, idx: int, raw: np.ndarray):
        base = os.path.join(self.ui.out_dir, f"trial_{idx}_raw")
        np.save(base + ".npy", raw)
        meta = {
            "sampling_rate_hz": ORIGINAL_RATE,
            "channel_labels":   list(self.streamer.kept_labels),
            "duration_s_requested": raw.shape[0] / ORIGINAL_RATE,
            "n_samples":  int(raw.shape[0]),
            "n_channels": int(raw.shape[1]),
            "saved_at":   datetime.now().isoformat(timespec="seconds"),
        }
        with open(base + "_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        logger.info(f"[Trial {idx}] Full trial saved to {base}.[npy/json]")

    def _save_fixation(self, idx: int, fix_raw: np.ndarray):
        base = os.path.join(self.ui.out_dir, f"trial_{idx}_fixation_raw")
        np.save(base + ".npy", fix_raw)
        meta = {
            "sampling_rate_hz": ORIGINAL_RATE,
            "channel_labels":   list(self.streamer.kept_labels),
            "window_start_offset_ms": DELAY_POST_STIMULUS,
            "window_duration_ms":     TRIAL_DURATION,
            "n_samples":  int(fix_raw.shape[0]),
            "n_channels": int(fix_raw.shape[1]),
            "saved_at":   datetime.now().isoformat(timespec="seconds"),
        }
        with open(base + "_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        logger.info(f"[Trial {idx}] Fixation raw saved to {base}.[npy/json]")

    # ------------------------------------------------------------------
    #                    PREDICTION  (runs in sub‑thread)
    # ------------------------------------------------------------------
    def _run_prediction_pipeline(self, idx: int, fix_raw: np.ndarray):
        self.movement_events[idx - 1].clear()          # block UI advance
        self.fixation_data_ready.emit(idx, None)       # update UI text

        # preprocess ----------------------------------------------------
        fix_proc = process_block(
            fix_raw,
            self.streamer.b_notch,
            self.streamer.a_notch,
            self.streamer.sos_bp
        )
        np.save(os.path.join(self.ui.out_dir, f"trial_{idx}_fixation.npy"), fix_proc)

        # inference -----------------------------------------------------
        t0 = time.perf_counter()
        if self.runner.mode == "both":
            pre_lbl, tta_lbl, *_ = self.runner.predict(fix_proc)
            label = tta_lbl
        elif self.runner.mode == "tta":
            tta_lbl, *_ = self.runner.predict(fix_proc)
            label = tta_lbl
        elif self.runner.mode == "pre_tta":
            pre_lbl, *_ = self.runner.predict(fix_proc)
            label = pre_lbl
        else:
            raise ValueError(f"Unsupported runner mode: {self.runner.mode}")
        logger.info(f"[Trial {idx}] Prediction took {time.perf_counter()-t0:.3f}s → {label}")

        self.trial_results[idx - 1]["predicted_label"] = label
        self._save_trial_results()
        self._control_exoskeleton(idx, label)

        self.movement_events[idx - 1].set()            # allow next trial

    # ------------------------------------------------------------------
    def _schedule_prediction(self, idx, _unused):
        """UI helper – executed in main Qt thread."""
        self.trial.show_message(Prediction_Text)

    @Slot()
    def _on_prediction_started(self):
        """Legacy slot – no longer used but kept for compatibility."""
        pass

    # ------------------------------------------------------------------
    def _control_exoskeleton(self, idx: int, label: int):
        logger.info(f"[Trial {idx}] control_exoskeleton: label={label}, arm_side={self.arm_side}")
        if label == 1 and self.exo:
            if self.arm_side == "left":
                # Left arm: DOWN then UP
                self.exo.send_hex(DOWN); time.sleep(ArmMovementDuration)
                self.exo.send_hex(UP);   time.sleep(ArmMovementDuration)
            else:  # right arm
                # Right arm: UP then DOWN
                self.exo.send_hex(UP);   time.sleep(ArmMovementDuration)
                self.exo.send_hex(DOWN); time.sleep(ArmMovementDuration)
        else:
            time.sleep(1)  # shorter wait when no movement

    # ------------------------------------------------------------------
    def _on_trial_finished(self):
        finished_idx = self.counter["idx"]
        self.wait_timer = QTimer()
        self.wait_timer.timeout.connect(
            lambda: self._advance_if_ready(finished_idx))
        self.wait_timer.start(100)

    def _advance_if_ready(self, finished_idx):
        if self.movement_events[finished_idx].is_set():
            self.wait_timer.stop()
            self.counter["idx"] += 1
            if self.counter["idx"] < self.total:
                self._prepare_and_start_trial(self.counter["idx"])
            else:
                self._finalize_experiment()
                self.trial.close()
                self.trial.deleteLater()

    # ------------------------------------------------------------------
    def _finalize_experiment(self):
        preds = [r["predicted_label"] for r in self.trial_results if r["predicted_label"] is not None]
        if not preds:
            logger.warning("No predictions; skipping accuracy.")
            return
        gts     = [r["ground_truth"] for r in self.trial_results]
        accuracy = sum(int(p == g) for p, g in zip(preds, gts)) / len(preds) * 100
        logger.info(f"Final accuracy: {accuracy:.2f}%")
        out = {"trials": self.trial_results, "final_accuracy_percent": accuracy}
        with open(os.path.join(self.ui.out_dir, "trial_results.json"), "w", encoding="utf-8") as f:
            json.dump(out, f, indent=4)

    def _save_trial_results(self):
        with open(os.path.join(self.ui.out_dir, "trial_results.json"), "w", encoding="utf-8") as f:
            json.dump(self.trial_results, f, indent=2)
# ───────────────────────────────────────────────────────────────────────────────
