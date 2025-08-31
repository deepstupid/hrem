from PyQt6.QtWidgets import QMainWindow, QStackedWidget
from .screens.investment_screen import InvestmentScreen
from .screens.discovery_screen import DiscoveryScreen

class MainWindow(QMainWindow):
    """The main window of the application."""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Scientific Discovery Engine")
        self.setGeometry(100, 100, 1200, 800)

        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)

        self.investment_screen = InvestmentScreen()
        self.discovery_screen = DiscoveryScreen()

        self.stacked_widget.addWidget(self.investment_screen)
        self.stacked_widget.addWidget(self.discovery_screen)

        # Connect signals between screens
        self.investment_screen.experiment_started.connect(self.start_experiment)
        self.discovery_screen.experiment_concluded.connect(self.conclude_experiment)

        self.stacked_widget.setCurrentWidget(self.investment_screen)

    def start_experiment(self, config: dict):
        """Switches to the discovery screen and starts the experiment."""
        self.discovery_screen.start_experiment_run(config)
        self.stacked_widget.setCurrentWidget(self.discovery_screen)

    def conclude_experiment(self):
        """
        Switches back to the investment screen and re-enables the start button.
        """
        self.stacked_widget.setCurrentWidget(self.investment_screen)
        self.investment_screen.enable_start_button()
