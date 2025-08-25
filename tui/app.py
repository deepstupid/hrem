from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, TabbedContent, TabPane

from .screens.evaluation import EvaluationScreen
from .screens.optimization import OptimizationScreen
from .screens.visualizer import DatasetVisualizer
from .screens.testing import TestingScreen
from .screens.dataset_management import DatasetManagementScreen
from .screens.demo import DemoScreen
from .screens.progress import ProgressScreen

class HRMApp(App):
    """A Textual app to run HRM System experiments."""

    TITLE = "HRM System"
    SUB_TITLE = "Evaluation, Optimization, and Testing"
    CSS_PATH = "styles.tcss"

    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        yield Header()
        with TabbedContent(initial="demo"):
            with TabPane("Demo", id="demo"):
                yield DemoScreen()
            with TabPane("Evaluation", id="evaluation"):
                yield EvaluationScreen()
            with TabPane("Optimization", id="optimization"):
                yield OptimizationScreen()
            with TabPane("Testing", id="testing"):
                yield TestingScreen()
            with TabPane("Dataset Management", id="dataset"):
                yield DatasetManagementScreen()
            with TabPane("Dataset Visualizer", id="visualizer"):
                yield DatasetVisualizer()
        yield Footer()

    def show_progress_screen(self):
        self.push_screen(ProgressScreen())

    def show_main_screen(self):
        self.pop_screen()

if __name__ == "__main__":
    app = HRMApp()
    app.run()
