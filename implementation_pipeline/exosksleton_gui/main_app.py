import sys
from PySide6.QtWidgets import QApplication
from user_info import UserInfoApp
from trial_manager import TrialManager

def main():
    app = QApplication(sys.argv)
    ui = UserInfoApp()
    manager = TrialManager(ui)
    ui.ui.StartButton.clicked.connect(manager.launch)
    ui.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
