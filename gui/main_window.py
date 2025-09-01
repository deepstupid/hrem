from PyQt6.QtWidgets import QMainWindow, QStackedWidget, QMessageBox
from .screens.welcome_screen import WelcomeScreen
from .screens.investment_screen import InvestmentScreen
from .screens.discovery_screen import DiscoveryScreen
from .styles import get_main_stylesheet

class MainWindow(QMainWindow):
    """
    The main window of the application, managing screen transitions and shared state.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Scientific Discovery Engine")
        self.setGeometry(100, 100, 1200, 800)
        self.setStyleSheet(get_main_stylesheet())

        # --- Screen Management ---
        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)

        self.welcome_screen = WelcomeScreen()
        self.investment_screen = InvestmentScreen()
        self.discovery_screen = DiscoveryScreen()

        self.stacked_widget.addWidget(self.welcome_screen)
        self.stacked_widget.addWidget(self.investment_screen)
        self.stacked_widget.addWidget(self.discovery_screen)

        # --- Signal Connections ---
        self.welcome_screen.start_requested.connect(self.show_investment_screen)
        self.investment_screen.experiment_started.connect(self.start_experiment)
        self.discovery_screen.experiment_concluded.connect(self.conclude_experiment)

        # --- Initial Screen ---
        self.stacked_widget.setCurrentWidget(self.welcome_screen)

    def show_investment_screen(self):
        """Switches to the investment screen."""
        self.investment_screen.reset_ui()
        self.stacked_widget.setCurrentWidget(self.investment_screen)

    def start_experiment(self, config: dict):
        """Switches to the discovery screen and starts the experiment."""
        try:
            self.discovery_screen.start_experiment_run(config)
            self.stacked_widget.setCurrentWidget(self.discovery_screen)
        except Exception as e:
            self._show_error(f"Failed to start experiment: {e}")
            self.conclude_experiment() # Go back to a safe state

    def conclude_experiment(self):
        """
        Switches back to the investment screen.
        """
        self.show_investment_screen()

    def _show_error(self, message: str):
        """Displays a critical error message in a modal dialog box."""
        QMessageBox.critical(
            self,
            "Error",
            f"⚠️ {message}",
            QMessageBox.StandardButton.Ok
        )

    def closeEvent(self, event):
        """Handle the user closing the window."""
        # Clean up the worker thread if it's running
        if self.discovery_screen and self.discovery_screen.worker:
            self.discovery_screen._cancel_experiment()
        event.accept()
