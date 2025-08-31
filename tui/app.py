import threading
from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, TabbedContent, TabPane, Button
from textual.binding import Binding

from sc_engine.core.model_runner import ScientificModelRunner
from .progress_handler import TUIProgressHandler
from .messages import EngineEvent
from .screens.discovery_dashboard import DiscoveryDashboard
from .screens.interactive_discovery import InteractiveDiscoveryScreen

# Import other screens as before
from .screens.evaluation import EvaluationScreen
from .screens.optimization import OptimizationScreen
from .screens.visualizer import DatasetVisualizer
from .screens.testing import TestingScreen
from .screens.dataset_management import DatasetManagementScreen
from .screens.demo import DemoScreen
from .screens.run_comparison import RunComparisonScreen


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
        self.patience_counter = 0

    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        yield Header()
        with TabbedContent(initial="interactive-discovery"):
            with TabPane("Interactive Discovery", id="interactive-discovery"):
                yield InteractiveDiscoveryScreen()
            with TabPane("Discovery Dashboard", id="discovery-dashboard"):
                yield DiscoveryDashboard()
            with TabPane("Run Comparison", id="run-comparison"):
                yield RunComparisonScreen()
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
            with TabPane("Demo", id="demo"):
                yield DemoScreen()
        yield Footer()

    def on_mount(self) -> None:
        self.query_one(DiscoveryDashboard).set_challenge("N/A")

    def on_interactive_discovery_screen_start_run(self, message: InteractiveDiscoveryScreen.StartRun) -> None:
        """Handle the start run message from the interactive discovery screen."""
        self.run_discovery_session(message.challenge_id)
        self.query_one(TabbedContent).active = "discovery-dashboard"

    def run_discovery_session(self, challenge_id: str):
        """Run the scientific discovery engine in a background thread."""
        dashboard = self.query_one(DiscoveryDashboard)
        dashboard.clear_all()
        self.patience_counter = 0
        dashboard.set_challenge(challenge_id)

        def run_in_thread():
            # For now, we'll keep this as a demo run type with smoke_test
            # This could be expanded to take more config from the UI
            self.model_runner.run(
                run_type="demo",
                challenge_id=challenge_id,
                smoke_test=True,
                progress_handler=self.progress_handler
            )

        thread = threading.Thread(target=run_in_thread)
        thread.start()

    def on_engine_event(self, message: EngineEvent) -> None:
        """Handle events from the scientific engine."""
        dashboard = self.query_one(DiscoveryDashboard)
        event_type = message.event_type
        data = message.data

        if event_type == 'start_session':
            dashboard.set_challenge(data.get('challenge', 'Unknown Challenge'))

        elif event_type == 'insights_generated':
            for i, insight in enumerate(data.get('insights', [])):
                dashboard.add_insight({
                    "rank": i + 1,
                    "text": insight.implications[0] if insight.implications else "Generated insight",
                    "category": insight.type,
                    "confidence": insight.confidence,
                    "potential": insight.discovery_potential
                })

        elif event_type == 'end_trial':
            # This event has 'params' and 'loss'
            # We need to map loss to a performance score for visualization
            performance = 1 - data.get('loss', 1.0)
            dashboard.add_hparam_point({
                "params": data.get('params', {}),
                "performance": performance
            })

        elif event_type == 'trainer:train_batch':
            self.patience_counter += 1
            num_insights = len(dashboard.insights_data)
            dashboard.add_patience_point((self.patience_counter, num_insights))

if __name__ == "__main__":
    app = HRMApp()
    app.run()
