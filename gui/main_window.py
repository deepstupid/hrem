from PyQt6.QtWidgets import QMainWindow, QMessageBox
from .screens.discovery_screen import DiscoveryScreen
from .styles import get_main_stylesheet

class MainWindow(QMainWindow):
    """
    The main application window.
    It now directly hosts the main dashboard screen.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Scientific Discovery Engine - Dashboard")
        self.setGeometry(100, 100, 1400, 900) # Increased size for the new dashboard
        self.setStyleSheet(get_main_stylesheet())

        # The DiscoveryScreen is now our main dashboard.
        self.dashboard_screen = DiscoveryScreen()
        self.setCentralWidget(self.dashboard_screen)

        # The dashboard will manage its own experiment runs.
        # Signal connections will be handled within the dashboard itself
        # or connected to other components like optimization windows later.

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
        # Ensure the worker thread in the dashboard is cleaned up on exit.
        if self.dashboard_screen and self.dashboard_screen.worker:
            self.dashboard_screen._cancel_experiment()
        event.accept()
