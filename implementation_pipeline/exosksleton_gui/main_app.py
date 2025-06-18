import sys
from PySide6.QtWidgets import QApplication
from user_info import UserInfoApp
from trial_window_ui import TrialWindow
import threading
from datetime import datetime

def trial_start():
    """Print the start time for external triggers (non-blocking)."""
    def _log_time():
        print(f"Trial started at {datetime.now().isoformat()}")
    threading.Thread(target=_log_time, daemon=True).start()

def main():
    app = QApplication(sys.argv)
    ui = UserInfoApp()
    def launch_trials():
        num_trials = int(ui.ui.NumTrialsBox.text().strip() or "1")

        trial = TrialWindow("left")
        ui._trial = trial

        counter = {"idx": 0}
        def on_finished():
            counter["idx"] += 1
            if counter["idx"] < num_trials:
                trial.set_trial_counter(counter["idx"]+1, num_trials)
                trial_start()
                trial.start()
            else:
                # destroy window after last trial
                trial.close()
                trial.deleteLater()

        trial.trial_finished.connect(on_finished)

        trial.set_trial_counter(1, num_trials)
        trial_start()
        trial.start()

    ui.ui.StartButton.clicked.connect(launch_trials)
    ui.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
