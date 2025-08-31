import threading
from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, TabbedContent, TabPane
from textual.binding import Binding

from sc_engine.core.model_runner import ScientificModelRunner
from .progress_handler import TUIProgressHandler
from .messages import EngineEvent
from .screens.interactive_discovery_launcher import InteractiveDiscoveryLauncherScreen
from .screens.interactive_discovery_dashboard import InteractiveDiscoveryDashboard

# Import other screens
from .screens.evaluation import EvaluationScreen
from .screens.optimization import OptimizationScreen
from .screens.testing import TestingScreen
from .screens.dataset_management import DatasetManagementScreen
from .screens.demo import DemoScreen


class HRMApp(App):
    """A Textual app to run HRM System experiments."""

    TITLE = "HRM System"
    SUB_TITLE = "Interactive Scientific Discovery"
    CSS_PATH = "styles.tcss"

    BINDINGS = [
        Binding("d", "toggle_dark", "Toggle dark mode"),
        Binding("q", "quit", "Quit"),
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_runner = ScientificModelRunner()
        self.progress_handler = TUIProgressHandler(self)

    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        yield Header()
        with TabbedContent(initial="interactive-discovery"):
            with TabPane("Launcher", id="launcher"):
                yield InteractiveDiscoveryLauncherScreen()
            with TabPane("Interactive Discovery", id="interactive-discovery"):
                yield InteractiveDiscoveryDashboard()
            with TabPane("Evaluation", id="evaluation"):
                yield EvaluationScreen()
            with TabPane("Optimization", id="optimization"):
                yield OptimizationScreen()
            with TabPane("Testing", id="testing"):
                yield TestingScreen()
            with TabPane("Dataset Management", id="dataset"):
                yield DatasetManagementScreen()
            with TabPane("Demo", id="demo"):
                yield DemoScreen()
        yield Footer()

if __name__ == "__main__":
    app = HRMApp()
    app.run()
