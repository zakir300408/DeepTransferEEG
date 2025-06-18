"""
trial_window.py
---------------
Fullscreen stimulus window for a single trial with arrow cue.

Timeline (total 14 s)
0–4 s   : black “rest”
4–5 s   : white fixation “+”
5–10 s  : arrow cue (← or →)
10–14 s : black “rest”

Signals
-------
rest1_started()
fixation_started()
stimulus_started()
rest2_started()
trial_finished()
"""

from PySide6.QtCore import Qt, QTimer, Signal, Slot
from PySide6.QtGui import QFont, QPalette, QColor
from PySide6.QtWidgets import QWidget, QLabel, QVBoxLayout
from constants import (
    show_rest_duration, show_fixation_duration, show_stimulus_duration, show_rest2_duration,
    Cross_Symbol, Arrow_Left_Symbol, Arrow_Right_Symbol, Cross_Size, Arrow_Size
)


class TrialWindow(QWidget):
    # hooks for external triggers
    rest1_started    = Signal()
    fixation_started = Signal()
    stimulus_started = Signal()
    rest2_started    = Signal()
    trial_finished   = Signal()

    def __init__(self, direction: str, parent=None):
        """
        direction: 'left' or 'right'
        """
        super().__init__(parent)
        self.direction = direction.lower()

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

        # huge font for both + and arrow
        big_font = QFont()
        big_font.setPointSize(Cross_Size if self.direction == "left" else Arrow_Size)
        big_font.setBold(True)
        self.label.setFont(big_font)

        # sequence: (duration_ms, handler)
        self.sequence = [
            (show_rest_duration, self._show_rest1),
            (show_fixation_duration, self._show_fixation),
            (show_stimulus_duration, self._show_arrow),
            (show_rest2_duration, self._show_rest2),
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

    def _show_rest1(self):
        self.label.hide()
        self.rest1_started.emit()

    def _show_fixation(self):
        self.label.setText(Cross_Symbol)
        self.label.show()
        self.fixation_started.emit()

    def _show_arrow(self):
        arrow = Arrow_Left_Symbol if self.direction == "left" else Arrow_Right_Symbol
        self.label.setText(arrow)
        self.label.show()
        self.stimulus_started.emit()

    def _show_rest2(self):
        self.label.hide()
        self.rest2_started.emit()

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
    # example: show right-arrow cue
    win = TrialWindow("right")
    win.start()
    sys.exit(app.exec())
