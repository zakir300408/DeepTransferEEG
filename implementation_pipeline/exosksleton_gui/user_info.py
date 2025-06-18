import sys
import os
import json
from datetime import datetime

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QMessageBox
)
from ui_main_window import Ui_MainWindow


class UserInfoApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # disable Start until all required inputs + folder are set
        self.ui.StartButton.setEnabled(False)

        # connect signals
        self.ui.BrowseButton.clicked.connect(self.browse_folder)
        self.ui.StartButton.clicked.connect(self.save_data)

        # keep Start button state up-to-date
        self.ui.lineEdit.textChanged.connect(self.update_start_button)
        self.ui.FirstNameBox.textChanged.connect(self.update_start_button)
        self.ui.LastNameBox.textChanged.connect(self.update_start_button)
        self.ui.AgeBox.textChanged.connect(self.update_start_button)
        self.ui.SessionIDText.textChanged.connect(self.update_start_button)
        self.ui.NumTrialsBox.textChanged.connect(self.update_start_button)
        self.ui.Female_radio.toggled.connect(self.update_start_button)
        self.ui.Male_radio.toggled.connect(self.update_start_button)

    def browse_folder(self):
        # only show directories
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Folder",
            os.getcwd(),
            QFileDialog.ShowDirsOnly
        )
        if directory:
            self.ui.lineEdit.setText(directory)

    def update_start_button(self, *args):
        folder       = self.ui.lineEdit.text().strip()
        first        = self.ui.FirstNameBox.toPlainText().strip()
        last         = self.ui.LastNameBox.toPlainText().strip()
        age          = self.ui.AgeBox.toPlainText().strip()
        session_id   = self.ui.SessionIDText.toPlainText().strip()
        num_trials   = self.ui.NumTrialsBox.text().strip()
        gender_valid = (
            self.ui.Female_radio.isChecked()
            or self.ui.Male_radio.isChecked()
        )

        ready = all([folder, first, last, age, session_id, num_trials, gender_valid])
        self.ui.StartButton.setEnabled(ready)

    def save_data(self):
        folder     = self.ui.lineEdit.text().strip()
        first      = self.ui.FirstNameBox.toPlainText().strip()
        last       = self.ui.LastNameBox.toPlainText().strip()
        age_str    = self.ui.AgeBox.toPlainText().strip()
        session_id = self.ui.SessionIDText.toPlainText().strip()
        comments   = self.ui.CommentsText.toPlainText().strip()
        num_trials_str = self.ui.NumTrialsBox.text().strip()
        gender     = (
            "Female"
            if self.ui.Female_radio.isChecked()
            else "Male"
        )

        # validate numeric age
        if not age_str.isdigit():
            QMessageBox.warning(
                self,
                "Invalid Age",
                "Age must be a positive integer."
            )
            return

        # validate number of trials
        if not num_trials_str.isdigit() or int(num_trials_str) <= 0:
            QMessageBox.warning(
                self,
                "Invalid Number of Trials",
                "Number of trials must be a positive integer."
            )
            return

        age = int(age_str)
        num_trials = int(num_trials_str)

        now = datetime.now()
        timestamp   = now.strftime("%Y%m%d_%H%M%S")
        folder_name = f"{first}_{session_id}_{timestamp}"
        out_dir     = os.path.join(folder, folder_name)

        # make the new subfolder
        try:
            os.makedirs(out_dir, exist_ok=False)
        except FileExistsError:
            QMessageBox.critical(
                self,
                "Folder Exists",
                f"A folder named\n{folder_name}\nalready exists."
            )
            return
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"Could not create folder:\n{e}"
            )
            return

        # build metadata dict
        metadata = {
            "first_name": first,
            "last_name": last,
            "gender": gender,
            "age": age,
            "session_id": session_id,
            "num_trials": num_trials,
            # include creation timestamp in ISO format
            "created_at": now.isoformat(timespec='seconds')
        }
        if comments:
            metadata["comments"] = comments

        # write JSON
        json_path = os.path.join(out_dir, "user_metadata.json")
        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=4)
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"Could not write JSON:\n{e}"
            )
            return

        QMessageBox.information(
            self,
            "Success",
            f"Data saved to:\n{json_path}"
        )


if __name__ == "__main__":
    app    = QApplication(sys.argv)
    window = UserInfoApp()
    window.show()
    sys.exit(app.exec())
