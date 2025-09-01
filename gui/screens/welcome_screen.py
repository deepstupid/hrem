from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout

class WelcomeScreen(QWidget):
    """
    The first screen of the application. It provides a welcome message and a brief
    explanation of the application's philosophy.
    """
    start_requested = pyqtSignal()

    def __init__(self):
        super().__init__()

        # --- Layouts ---
        main_layout = QVBoxLayout(self)
        content_layout = QVBoxLayout()
        button_layout = QHBoxLayout()

        # --- Widgets ---
        title_label = QLabel("Welcome to the Scientific Discovery Engine")
        title_label.setObjectName("titleLabel")

        description_label = QLabel(
            "This application is a testbed for a core theory: that scientific progress "
            "can be viewed as a resource allocation problem. You 'invest' a computational "
            "budget, and the engine works to maximize the 'return' in scientific insights."
        )
        description_label.setObjectName("descriptionLabel")
        description_label.setWordWrap(True)

        header_label = QLabel("The Process")
        header_label.setObjectName("headerLabel")

        process_label = QLabel(
            "1.  <b>INVEST:</b> On the next screen, you will configure an experiment. You'll define the "
            "scientific problem, select algorithms to compare, and set your 'patience'—the "
            "computational budget you're willing to invest.<br><br>"
            "2.  <b>DISCOVER:</b> The engine will then run the experiment, automatically managing "
            "resources to find the most promising results within your budget. You can watch the "
            "progress in real-time as the engine generates insights."
        )
        process_label.setWordWrap(True)

        self.start_button = QPushButton("Begin Investigation →")
        self.start_button.clicked.connect(self.start_requested.emit)

        # --- Assemble Layout ---
        content_layout.addWidget(title_label)
        content_layout.addWidget(description_label)
        content_layout.addSpacing(20)
        content_layout.addWidget(header_label)
        content_layout.addWidget(process_label)
        content_layout.addStretch()

        button_layout.addStretch()
        button_layout.addWidget(self.start_button)
        button_layout.addStretch()

        main_layout.addLayout(content_layout)
        main_layout.addLayout(button_layout)
        main_layout.setContentsMargins(50, 50, 50, 50)
