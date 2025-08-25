from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import Header, Footer, Log, ProgressBar, DataTable
from textual import work
import threading
from typing import Dict, Any

from hrm_system import (
    ExperimentConfig,
    run_evaluation,
    run_optimization,
)


class ProgressScreen(Screen):
    """A screen to display the progress of an experiment."""

    BINDINGS = [("escape", "app.pop_screen", "Back to Config")]

    def __init__(self, experiment_config: ExperimentConfig) -> None:
        super().__init__()
        self.experiment_config = experiment_config
        self.stop_event = threading.Event()

    def compose(self) -> ComposeResult:
        """Create child widgets for the screen."""
        yield Header()
        yield DataTable(id="results_table", show_cursor=False, show_header=True)
        yield ProgressBar(id="main_progress", total=100, show_eta=False)
        yield Log(id="demo_log", classes="log_view", auto_scroll=True)
        yield Footer()

    def on_mount(self) -> None:
        """Initialize the results table and start the experiment."""
        table = self.query_one("#results_table", DataTable)
        table.add_columns("Model", "Accuracy", "Loss", "Steps", "Parameters")
        self.run_experiment_worker()

    @work(thread=True)
    def run_experiment_worker(self) -> None:
        """Run the experiment in a worker thread."""
        self.run_experiment()

    def run_experiment(self) -> None:
        """Run the experiment based on the config."""
        try:
            self.log_message("[bold blue]🚀 Starting Experiment...[/bold blue]")
            self.update_progress(0)

            results = {}
            if self.experiment_config.mode == "evaluate":
                results = run_evaluation(self.experiment_config)
            elif self.experiment_config.mode == "optimize":
                results = run_optimization(self.experiment_config)

            self.update_progress(50)
            self.show_results("Experiment Results", results.get("results", {}))
            self.update_progress(100)

            self.log_message("[bold green]✅ Experiment Completed Successfully![/bold green]")

        except Exception as e:
            self.log_message(f"[bold red]❌ Experiment failed: {e}[/bold red]")
        finally:
            self.log_message("Press Esc to return to the configuration screen.")

    def show_results(self, title: str, results: Dict[str, Any]) -> None:
        """Display results in the table."""
        self.log_message(f"[bold]{title}:[/bold]")

        table = self.query_one("#results_table", DataTable)
        table.clear()
        table.add_columns("Model", "Accuracy", "Loss", "Steps", "Parameters")

        model_names = [name for name in ['HRM', 'HREM', 'HRM_best', 'HREM_best'] if name in results]

        for model_name in model_names:
            metrics = results.get(model_name, {})
            if not metrics:
                continue

            accuracy = f"{metrics.get('all/accuracy', 'N/A'):.4f}" if isinstance(metrics.get('all/accuracy'), float) else 'N/A'
            loss = f"{metrics.get('all/lm_loss', 'N/A'):.4f}" if isinstance(metrics.get('all/lm_loss'), float) else 'N/A'
            steps = f"{metrics.get('all/steps', 'N/A'):.0f}" if isinstance(metrics.get('all/steps'), (int, float)) else 'N/A'
            params = f"{metrics.get('num_params', 'N/A'):,}" if isinstance(metrics.get('num_params'), int) else 'N/A'

            table.add_row(model_name, accuracy, loss, steps, params)
            self.log_message(f"  {model_name} - Accuracy: {accuracy}, Loss: {loss}, Steps: {steps}, Params: {params}")

    def update_progress(self, progress: float) -> None:
        """Update a progress bar."""
        try:
            progress_bar = self.query_one("#main_progress", ProgressBar)
            progress_bar.update(progress=progress)
        except Exception:
            pass

    def log_message(self, message: str) -> None:
        """Add a message to the log."""
        try:
            log_widget = self.query_one("#demo_log", Log)
            log_widget.write(str(message))
        except Exception:
            pass

    def on_unmount(self) -> None:
        """Clean up when the screen is unmounted."""
        self.stop_event.set()
