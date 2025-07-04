# Monkey-patch typing.Self for older Pythons so torch._dynamo can import it
import typing
try:
    from typing import Self
except ImportError:
    from typing_extensions import Self
typing.Self = Self

import sys
import os
import logging
from PySide6.QtWidgets import QApplication
from ui.user_info import UserInfoApp
from implementation_main import EnsembleRunner
from trial_manager import TrialManager   
from utils_gui.constants import SEEDS

def main():
    app = QApplication(sys.argv)
    ui = UserInfoApp()
    # create ensemble runner first
    runner = EnsembleRunner(seeds=SEEDS, mode="tta")

    # Patch: connect to a signal after user data is saved to redirect logs
    def after_save():
        # Only redirect if out_dir is set and not already redirected
        if getattr(ui, "out_dir", None) and not hasattr(app, "_log_redirected"):
            log_path = os.path.join(ui.out_dir, "log.txt")
            log_file = open(log_path, "a", buffering=1)

            # Tee stdout/stderr to both console and file
            orig_stdout, orig_stderr = sys.stdout, sys.stderr
            class Tee:
                def __init__(self, *writers):
                    self.writers = writers
                def write(self, msg):
                    for w in self.writers:
                        w.write(msg)
                def flush(self):
                    for w in self.writers:
                        w.flush()
            sys.stdout = Tee(orig_stdout, log_file)
            sys.stderr = Tee(orig_stderr, log_file)

            import logging as _logging
            root = _logging.getLogger()
            # add a FileHandler alongside existing console handlers
            fh = _logging.FileHandler(log_path, mode="a", encoding="utf-8")
            fh.setLevel(_logging.INFO)
            fh.setFormatter(_logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
            root.addHandler(fh)

            app._log_redirected = True  # Prevent multiple redirections

    # Connect after_save to be called after data is saved
    orig_save_data = ui.save_data
    def wrapped_save_data(*args, **kwargs):
        result = orig_save_data(*args, **kwargs)
        after_save()
        return result
    ui.save_data = wrapped_save_data
    ui.ui.StartButton.clicked.disconnect()
    ui.ui.StartButton.clicked.connect(ui.save_data)

    # pass runner into manager
    manager = TrialManager(ui, runner)
    ui.ui.StartButton.clicked.connect(manager.launch)
    ui.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
