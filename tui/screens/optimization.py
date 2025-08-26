from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static, Select, Input, Button, Checkbox, RichLog
from textual import work

from hrm_system import (
    ExperimentConfig,
    RunConfig,
    DataConfig,
    OptimizationConfig,
    run_optimization,
)

class OptimizationScreen(Static):
    """The screen for running hyperparameter optimization."""

    def compose(self) -> ComposeResult:
        yield Static("Configure and run hyperparameter optimization for the HREM model.", classes="header")

        with Vertical(classes="form"):
            yield Static("Dataset:")
            yield Select(
                options=[("Synthetic", "synthetic"), ("ARC", "arc"), ("Sudoku", "sudoku"), ("Maze", "maze")],
                value="synthetic",
                id="opt_dataset"
            )
            yield Static("Number of trials:")
            yield Input(value="10", id="opt_n_trials")
            yield Static("Number of parallel jobs:")
            yield Input(value="1", id="opt_n_jobs")
            yield Checkbox("Run as smoke test", value=True, id="opt_smoke_test")

        yield Button("Run Optimization", variant="primary", id="run_optimization_button")

        yield RichLog(id="optimization_log", classes="log_view", auto_scroll=True, wrap=True)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle the run optimization button press."""
        if event.button.id == "run_optimization_button":
            self.run_the_optimization()

    def run_the_optimization(self) -> None:
        """Get config from form and run the optimization in a worker thread."""
        log_widget = self.query_one("#optimization_log", Log)
        log_widget.clear()

        try:
            dataset = self.query_one("#opt_dataset", Select).value
            n_trials = int(self.query_one("#opt_n_trials", Input).value)
            n_jobs = int(self.query_one("#opt_n_jobs", Input).value)
            smoke_test = self.query_one("#opt_smoke_test", Checkbox).value
        except (ValueError, IndexError):
            log_widget.write("[bold red]Error: Invalid form values.[/bold red]")
            return

        study_name = f"tui_opt_{dataset}"
        if smoke_test:
            study_name += "_smoke"

        config = ExperimentConfig(
            mode="optimize",
            run_config=RunConfig(
                smoke_test=smoke_test,
                study_name=study_name,
                logger_callback=lambda msg: self.post_message(self.LogMessage(msg))
            ),
            data_config=DataConfig(dataset=dataset),
            optimization_config=OptimizationConfig(n_trials=n_trials, n_jobs=n_jobs)
        )

        log_widget.write(f"Starting optimization: {study_name}...")
        self.run_worker(self.execute_optimization, config, exclusive=True)

    @work(thread=True, exclusive=True)
    def execute_optimization(self, config: ExperimentConfig) -> None:
        """The worker thread that runs the optimization."""
        try:
            run_optimization(config)
            self.post_message(self.LogMessage("[bold green]Optimization finished successfully.[/bold green]"))
        except Exception as e:
            self.post_message(self.LogMessage(f"[bold red]Optimization failed: {e}[/bold red]"))

    class LogMessage(Static):
        """A message to be posted to the log."""
        def __init__(self, message: Any) -> None:
            super().__init__()
            self.message = message

    def on_optimization_screen_log_message(self, message: LogMessage) -> None:
        """Handle a log message from a worker."""
        log_widget = self.query_one("#optimization_log", RichLog)
        if isinstance(message.message, str):
            log_widget.write(message.message)
        else:
            # Clear the log and write the table for live updates
            log_widget.clear()
            log_widget.write(message.message)
