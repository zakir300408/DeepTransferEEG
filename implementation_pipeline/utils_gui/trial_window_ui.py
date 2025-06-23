"""
trial_window.py
---------------
Fullscreen stimulus window for a single trial with arrow cue.

Timeline (total 11 s)
0–4 s   : black “rest”
4–5 s   : white fixation “+”
5–11 s  : arrow cue (← or →)

Signals
-------
rest1_started()
fixation_started()
stimulus_started()
trial_finished()
"""

from PySide6.QtCore import Qt, QTimer, Signal, Slot
from PySide6.QtGui import QFont, QPalette, QColor
from PySide6.QtWidgets import QWidget, QLabel, QVBoxLayout
from .constants import (
    show_rest_duration, show_fixation_duration, show_stimulus_duration,
    Cross_Symbol, Stimulus_Symbol, Cross_Size, Stimulus_Symbol_Size
)
import logging


logger = logging.getLogger(__name__)


class TrialWindow(QWidget):
    # hooks for external triggers
    rest1_started    = Signal()
    fixation_started = Signal()
    stimulus_started = Signal()
    trial_finished   = Signal()

    def __init__(self, parent=None):
        """
        """
        super().__init__(parent)

        # full-screen black window
        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint)
        self.showFullScreen()
        pal = self.palette()
        pal.setColor(QPalette.Window, QColor("black"))
        self.setPalette(pal)
        self.setAutoFillBackground(True)

        # layout so label always fills and centers
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)

        # central label for fixation or arrow
        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.label)
        self.label.hide()

        # trial counter in the top‐left corner
        self.counter_label = QLabel(self)
        self.counter_label.setStyleSheet("color: white; font-size: 18pt;")
        self.counter_label.move(10, 10)
        self.counter_label.hide()

        # font for stimuli (+ and box)
        self.stimulus_font = QFont()
        self.stimulus_font.setPointSize(Stimulus_Symbol_Size)
        self.stimulus_font.setBold(True)

        # font for messages like "Prediction"
        self.message_font = QFont()
        self.message_font.setPointSize(128)
        self.message_font.setBold(True)

        # sequence: (duration_ms, handler)
        self.sequence = [
            (show_rest_duration, self._show_rest1),
            (show_fixation_duration, self._show_fixation),
            (show_stimulus_duration, self._show_arrow)
        ]
        self.current_step = -1
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self._next_step)

    def start(self):
        """Begin (or restart) the trial sequence."""
        self.current_step = -1                      # reset sequence
        self.counter_label.raise_()
        self.showFullScreen()                       # ensure it's visible
        self._next_step()

    def set_trial_counter(self, current: int, total: int):
        """Display 'current/total' in corner."""
        self.counter_label.setText(f"{current}/{total}")
        self.counter_label.show()

    def show_message(self, message: str):
        """Display a custom message with a different font."""
        logger.info(f"Displaying message: {message}")
        self.label.setFont(self.message_font)
        self.label.setText(message)
        self.label.show()

    def _show_rest1(self):
        logger.info("Rest1 started")
        self.label.hide()
        self.rest1_started.emit()

    def _show_fixation(self):
        logger.info("Fixation started")
        self.label.setFont(self.stimulus_font)
        self.label.setText(Cross_Symbol)
        self.label.show()
        self.fixation_started.emit()

    def _show_arrow(self):
        logger.info("Stimulus started")
        self.label.setFont(self.stimulus_font)
        self.label.setText(Stimulus_Symbol)
        self.label.show()
        self.stimulus_started.emit()

    @Slot()
    def _next_step(self):
        self.current_step += 1
        if self.current_step >= len(self.sequence):
            self.trial_finished.emit()
            # do not close so we can restart on demand
            return

        duration, handler = self.sequence[self.current_step]
        handler()
        self.timer.start(duration)


if __name__ == "__main__":
    from PySide6.QtWidgets import QApplication
    import sys

    app = QApplication(sys.argv)
    # example: show stimulus cue
    win = TrialWindow()
    win.start()
    sys.exit(app.exec())