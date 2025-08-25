from textual.app import ComposeResult
from textual.containers import Vertical, Horizontal
from textual.widgets import Static, Select, Input, Button, Checkbox, Log
from textual import work

from hrm_system import (
    ExperimentConfig,
    RunConfig,
    DataConfig,
    EvaluationConfig,
    run_evaluation,
)

class EvaluationScreen(Static):
    """The screen for running model evaluations."""

    def compose(self) -> ComposeResult:
        yield Static("Configure and run a side-by-side evaluation of HRM vs. HREM.", classes="header")

        with Vertical(classes="form"):
            yield Static("Dataset:")
            yield Select(
                options=[("Synthetic", "synthetic"), ("ARC", "arc"), ("Sudoku", "sudoku"), ("Maze", "maze")],
                value="synthetic",
                id="eval_dataset"
            )
            yield Static("Number of runs (for statistical significance):")
            yield Input(value="1", id="eval_n_runs")
            yield Checkbox("Run as smoke test (uses tiny dataset)", value=True, id="eval_smoke_test")

        yield Button("Run Evaluation", variant="primary", id="run_evaluation_button")

        yield Log(id="evaluation_log", classes="log_view", auto_scroll=True)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle the run evaluation button press."""
        if event.button.id == "run_evaluation_button":
            self.run_the_evaluation()

    def run_the_evaluation(self) -> None:
        """Get config from form and run the evaluation in a worker thread."""
        log_widget = self.query_one("#evaluation_log", Log)
        log_widget.clear()

        try:
            dataset = self.query_one("#eval_dataset", Select).value
            n_runs = int(self.query_one("#eval_n_runs", Input).value)
            smoke_test = self.query_one("#eval_smoke_test", Checkbox).value
        except (ValueError, IndexError):
            log_widget.write("[bold red]Error: Invalid form values.[/bold red]")
            return

        study_name = f"tui_eval_{dataset}"
        if smoke_test:
            study_name += "_smoke"

        config = ExperimentConfig(
            mode="evaluate",
            run_config=RunConfig(
                smoke_test=smoke_test,
                study_name=study_name,
                logger_callback=lambda msg: self.post_message(self.LogMessage(msg))
            ),
            data_config=DataConfig(dataset=dataset),
            evaluation_config=EvaluationConfig(n_runs=n_runs)
        )

        log_widget.write(f"Starting evaluation: {study_name}...")
        self.run_worker(self.execute_evaluation, config, exclusive=True)

    @work(thread=True, exclusive=True)
    def execute_evaluation(self, config: ExperimentConfig) -> None:
        """The worker thread that runs the evaluation."""
        try:
            run_evaluation(config)
            self.post_message(self.LogMessage("[bold green]Evaluation finished successfully.[/bold green]"))
        except Exception as e:
            self.post_message(self.LogMessage(f"[bold red]Evaluation failed: {e}[/bold red]"))

    class LogMessage(Static):
        """A message to be posted to the log."""
        def __init__(self, message: str) -> None:
            super().__init__()
            self.message = message

    def on_evaluation_screen_log_message(self, message: LogMessage) -> None:
        """Handle a log message from a worker."""
        log_widget = self.query_one("#evaluation_log", Log)
        log_widget.write(message.message)
