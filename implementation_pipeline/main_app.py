# Monkey-patch typing.Self for older Pythons so torch._dynamo can import it
import typing
try:
    from typing import Self
except ImportError:
    from typing_extensions import Self
typing.Self = Self

import sys
from PySide6.QtWidgets import QApplication
from utils_gui.user_info import UserInfoApp
from implementation_main import EnsembleRunner
from trial_manager import TrialManager   # reorder after importing runner

def main():
    app = QApplication(sys.argv)
    ui = UserInfoApp()
    # create ensemble runner first
    runner = EnsembleRunner(seeds=[2,3], sample_rate=100)
    # pass runner into manager
    manager = TrialManager(ui, runner)
    ui.ui.StartButton.clicked.connect(manager.launch)
    ui.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
    main()
