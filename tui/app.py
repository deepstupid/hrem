from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, TabbedContent, TabPane

from .screens.evaluation import EvaluationScreen
from .screens.optimization import OptimizationScreen
from .screens.visualizer import DatasetVisualizer

class HRMApp(App):
    """A Textual app to run HRM System experiments."""

    TITLE = "HRM System"
    SUB_TITLE = "Evaluation and Optimization"

    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        yield Header()
        with TabbedContent(initial="evaluation"):
            with TabPane("Evaluation", id="evaluation"):
                yield EvaluationScreen()
            with TabPane("Optimization", id="optimization"):
                yield OptimizationScreen()
            with TabPane("Dataset Visualizer", id="visualizer"):
                yield DatasetVisualizer()
        yield Footer()

if __name__ == "__main__":
    app = HRMApp()
    app.run()
