# trial_window.py
# ---------------
# Fullscreen stimulus window for a single trial with arrow cue.
#
# Timeline (total 11 s)
# 0–4 s   : black “rest”
# 4–5 s   : white fixation “+”
# 5–11 s  : arrow cue (← or →) or GIF
#
# Signals
# -------
# rest1_started()
# fixation_started()
# stimulus_started()
# trial_finished()

from PySide6.QtCore import Qt, QTimer, Signal, Slot
from PySide6.QtGui import QFont, QPalette, QColor, QMovie
from PySide6.QtWidgets import QWidget, QLabel, QVBoxLayout
from implementation_pipeline.utils_gui.constants import (
    show_rest_duration, show_fixation_duration, show_stimulus_duration,
    Cross_Symbol, Stimulus_Symbol, No_Stimulus_Symbol, Text_Symbol_Size, STIMULUS_GIF_PATH
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

        # central label for fixation, arrow, or GIF
        self.label = QLabel(self)
        self.label.setStyleSheet("color: white")  # ensure all label text is white
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setScaledContents(True)  # <-- allow scaling to fill
        self.layout.addWidget(self.label)
        self.label.hide()

        # trial counter in the top‐left corner
        self.counter_label = QLabel(self)
        self.counter_label.setStyleSheet("color: white; font-size: 18pt;")
        self.counter_label.move(10, 10)
        self.counter_label.hide()

        # font for stimuli (+ and box)
        self.stimulus_font = QFont()
        self.stimulus_font.setPointSize(Text_Symbol_Size)
        self.stimulus_font.setBold(True)

        # font for messages like "Prediction"
        self.message_font = QFont()
        self.message_font.setPointSize(Text_Symbol_Size)
        self.message_font.setBold(True)

        # for playing the GIF
        self.movie = None

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
        self.stimulus_symbol = Stimulus_Symbol

    def start(self):
        """Begin (or restart) the trial sequence."""
        # ensure any running GIF is stopped
        if self.movie:
            self.movie.stop()
        self.timer.stop()
        self.current_step = -1
        self.counter_label.raise_()
        self.showFullScreen()
        self._next_step()

    def set_trial_counter(self, current: int, total: int):
        """Display 'current/total' in corner."""
        self.counter_label.setText(f"{current}/{total}")
        self.counter_label.show()

    def show_message(self, message: str):
        # stop GIF now that we're about to show static text
        if self.movie:
            self.movie.stop()
            self.movie = None

        pal = self.palette()
        pal.setColor(QPalette.Window, QColor("black"))
        self.setPalette(pal)

        logger.info(f"Displaying message: {message}")
        self.label.setFont(self.message_font)
        self.label.setText(message)
        self.label.show()

    def _show_rest1(self):
        pal = self.palette()
        pal.setColor(QPalette.Window, QColor("black"))
        self.setPalette(pal)

        logger.info("Rest1 started")
        self.label.setFont(self.message_font)
        self.label.setText("休息")
        self.label.show()
        self.rest1_started.emit()

    def _show_fixation(self):
        pal = self.palette()
        pal.setColor(QPalette.Window, QColor("black"))
        self.setPalette(pal)

        logger.info("Fixation started")
        self.label.setFont(self.stimulus_font)
        self.label.setText(Cross_Symbol)
        self.label.show()
        self.fixation_started.emit()

    def _show_arrow(self):
        """Called at t = rest+fixation to display the stimulus (arrow text or GIF)."""
        # set background color
        pal = self.palette()
        if self.stimulus_symbol == No_Stimulus_Symbol:
            pal.setColor(QPalette.Window, QColor(139, 0, 0))
        elif self.stimulus_symbol == Stimulus_Symbol:
            pal.setColor(QPalette.Window, QColor(0, 100, 0))
        else:
            pal.setColor(QPalette.Window, QColor("black"))
        self.setPalette(pal)

        logger.info("Stimulus started")

        if self.stimulus_symbol == Stimulus_Symbol:
            # stop any previous movie
            if self.movie:
                self.movie.stop()
            # load, scale, and start the GIF
            self.movie = QMovie(STIMULUS_GIF_PATH)
            # initial scale to label size
            self.movie.setScaledSize(self.label.size())
            self.label.setMovie(self.movie)
            self.movie.start()
        else:
            # stop GIF if it was playing
            if self.movie:
                self.movie.stop()
                self.movie = None
            # fallback to text stimulus
            self.label.setFont(self.stimulus_font)
            self.label.setText(self.stimulus_symbol)

        self.label.show()
        self.stimulus_started.emit()

    # ensure GIF always matches label size on resize
    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.movie:
            self.movie.setScaledSize(self.label.size())

    def set_stimulus_symbol(self, symbol: str):
        """Override the symbol shown; background set later in _show_arrow."""
        self.stimulus_symbol = symbol

    @Slot()
    def _next_step(self):
        self.current_step += 1
        if self.current_step >= len(self.sequence):
            # do not stop GIF here; let it keep playing until prediction
            self.trial_finished.emit()
            return

        duration, handler = self.sequence[self.current_step]
        handler()
        self.timer.start(duration)


if __name__ == "__main__":
    from PySide6.QtWidgets import QApplication
    import sys

    app = QApplication(sys.argv)
    win = TrialWindow()
    win.start()
    sys.exit(app.exec())
