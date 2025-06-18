from PySide6.QtCore import QCoreApplication, Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QLabel, QLineEdit,
    QPlainTextEdit, QRadioButton, QPushButton,
    QVBoxLayout, QHBoxLayout, QFormLayout, QFrame,
    QSizePolicy
)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 800)

        # Central widget & main layout
        self.centralwidget = QWidget(MainWindow)
        main_layout = QVBoxLayout(self.centralwidget)
        main_layout.setContentsMargins(24, 24, 24, 24)
        main_layout.setSpacing(20)

        # Title
        self.MainTitle = QLabel("BCI Exoskeleton", self.centralwidget)
        title_font = QFont()
        title_font.setPointSize(22)
        title_font.setBold(True)
        self.MainTitle.setFont(title_font)
        self.MainTitle.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.MainTitle)

        # Separator
        separator = QFrame(self.centralwidget)
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        main_layout.addWidget(separator)

        # Form layout
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)
        form.setFormAlignment(Qt.AlignHCenter | Qt.AlignTop)
        form.setHorizontalSpacing(30)
        form.setVerticalSpacing(20)

        # Helper to standardize heights
        def make_edit():
            w = QPlainTextEdit(self.centralwidget)
            w.setFixedHeight(30)
            return w

        # First name
        self.FirstNameBox = make_edit()
        form.addRow("First Name:", self.FirstNameBox)

        # Last name
        self.LastNameBox = make_edit()
        form.addRow("Last Name:", self.LastNameBox)

        # Gender row
        gender_widget = QWidget(self.centralwidget)
        gender_layout = QHBoxLayout(gender_widget)
        gender_layout.setContentsMargins(0, 0, 0, 0)
        gender_layout.setSpacing(15)
        self.Female_radio = QRadioButton("Female", gender_widget)
        self.Male_radio = QRadioButton("Male", gender_widget)
        # match height to other fields
        for btn in (self.Female_radio, self.Male_radio):
            btn.setFont(QFont("", 14))
            btn.setFixedHeight(30)
        gender_layout.addWidget(self.Female_radio)
        gender_layout.addWidget(self.Male_radio)
        gender_layout.addStretch()
        form.addRow("Gender:", gender_widget)

        # Age
        self.AgeBox = make_edit()
        form.addRow("Age:", self.AgeBox)

        # Session ID
        self.SessionIDText = make_edit()
        form.addRow("Session ID:", self.SessionIDText)

        # Number of Trials (new)
        from PySide6.QtWidgets import QLineEdit
        self.NumTrialsBox = QLineEdit(self.centralwidget)
        self.NumTrialsBox.setPlaceholderText("e.g. 10")
        self.NumTrialsBox.setFixedHeight(30)
        self.NumTrialsBox.setFont(QFont("", 14))
        form.addRow("Number of Trials:", self.NumTrialsBox)

        # Comments (taller)
        self.CommentsText = QPlainTextEdit(self.centralwidget)
        self.CommentsText.setFixedHeight(60)
        form.addRow("Comments:", self.CommentsText)

        main_layout.addLayout(form)

        # File selector
        file_layout = QHBoxLayout()
        file_layout.setContentsMargins(0, 0, 0, 0)
        file_layout.setSpacing(15)
        self.lineEdit = QLineEdit(self.centralwidget)
        self.lineEdit.setPlaceholderText("File save location")
        self.lineEdit.setFont(QFont("", 14))
        self.lineEdit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.lineEdit.setFixedHeight(30)
        self.BrowseButton = QPushButton("Browse", self.centralwidget)
        browse_font = QFont()
        browse_font.setPointSize(16)
        browse_font.setBold(True)
        self.BrowseButton.setFont(browse_font)
        self.BrowseButton.setFixedHeight(36)
        file_layout.addWidget(self.lineEdit)
        file_layout.addWidget(self.BrowseButton)
        main_layout.addLayout(file_layout)

        # Push everything upwards
        main_layout.addStretch()

        # Start button
        self.StartButton = QPushButton("Start", self.centralwidget)
        start_font = QFont()
        start_font.setPointSize(30)
        self.StartButton.setFont(start_font)
        self.StartButton.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.StartButton.setFixedHeight(64)
        main_layout.addWidget(self.StartButton)

        MainWindow.setCentralWidget(self.centralwidget)
        self.retranslateUi(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(
            QCoreApplication.translate("MainWindow", "BCI Exoskeleton")
        )

if __name__ == "__main__":
    import sys
    from PySide6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    MainWindow = QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec())