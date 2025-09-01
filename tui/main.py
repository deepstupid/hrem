from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import (
    Header, Footer, Static, Button, Checkbox, RadioSet, TabbedContent, TabPane, Log, DataTable, Select
)
from textual import work
from typing import Dict, Any

from sc_engine.core.config_manager import ConfigManager
from sc_engine.core.challenge_registry import ChallengeRegistry

from .worker import (
    TUIExperimentWorker, ProgressUpdate, ExperimentFinished, WorkerFinished, ErrorOccurred
)


class DiscoveryTUI(App):
    """A Textual UI for the Scientific Discovery Engine."""

    # --- TUI Setup ---
    CSS_PATH = "styles.css"

    BINDINGS = [
        ("d", "toggle_dark", "Toggle dark mode"),
        ("q", "quit", "Quit")
    ]

    # --- App State ---
    def __init__(self):
        super().__init__()
        self.current_algorithm = ""
        self.algorithm_steps = {}
        self.experiment_worker: TUIExperimentWorker | None = None

    # --- UI Composition ---
    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        with Horizontal():
            # Left Pane: Controls
            with Vertical(id="controls-pane"):
                yield Static("1. Select Challenge", classes="header")
                yield Select([], id="challenge-select", prompt="Select a challenge...")

                yield Static("2. Select Models", classes="header")
                with VerticalScroll(id="model-checkboxes"):
                    pass # Populated by on_mount

                yield Static("3. Set Patience", classes="header")
                with RadioSet(id="patience-radioset"):
                    yield Button("Low", id="patience_low")
                    yield Button("Medium", id="patience_medium")
                    yield Button("High", id="patience_high")

                yield Button("🚀 Start Experiment", id="start-button", variant="primary")
                yield Button("⏸️ Pause", id="pause-button", disabled=True)

            # Right Pane: Results
            with Vertical(id="results-pane"):
                with TabbedContent(id="results-tabs"):
                    with TabPane("📜 Live Log", id="log-pane"):
                        yield Log(id="live-log", highlight=True, auto_scroll=True)
                    with TabPane("📊 Live Metrics", id="metrics-pane"):
                        yield DataTable(id="metrics-table")
                    with TabPane("📈 Performance Plot", id="plot-pane"):
                        yield Static("Plotting is not yet implemented in the TUI.", id="plot-placeholder")
        yield Footer()

    # --- Initial Setup ---
    def on_mount(self) -> None:
        """Called when the app is mounted."""
        self.query_one("#live-log").write_line("Welcome to the Scientific Discovery Engine TUI.")
        self.query_one(DataTable).add_columns("Metric", "Value")
        self._populate_controls()

    def _populate_controls(self):
        """Populates the control widgets with data from the config files."""
        config_manager = ConfigManager()
        challenge_registry = ChallengeRegistry(config_manager)

        challenges = challenge_registry.get_all_challenges()
        challenge_select = self.query_one(Select)
        challenge_select.set_options([(c.name, c.id) for c in challenges])
        if challenges:
            challenge_select.value = challenges[0].id

        model_configs = config_manager.load_model_configs()
        model_checkboxes_container = self.query_one("#model-checkboxes")
        for model_name in sorted(model_configs.keys()):
            checkbox = Checkbox(model_name, id=f"model_{model_name}")
            if model_name in ["HRM", "HREM"]:
                checkbox.value = True
            model_checkboxes_container.mount(checkbox)

        self.query_one("#patience_low").variant = "primary" # Default selection

    # --- Event Handlers & Workers ---
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press events for starting, stopping, and pausing."""
        button_id = event.button.id

        if button_id == "start-button":
            if self.experiment_worker:
                self.query_one(Log).write_line("[bold yellow]--- Stopping Experiment ---[/bold yellow]")
                self.experiment_worker.stop()
                event.button.disabled = True
            else:
                self._start_experiment()

        elif button_id == "pause-button":
            if self.experiment_worker:
                if self.experiment_worker.is_paused:
                    self.experiment_worker.resume()
                    self.query_one(Log).write_line("[bold cyan]--- Resuming Experiment ---[/bold cyan]")
                    event.button.label = "⏸️ Pause"
                else:
                    self.experiment_worker.pause()
                    self.query_one(Log).write_line("[bold cyan]--- Pausing Experiment ---[/bold cyan]")
                    event.button.label = "▶️ Resume"

    def _start_experiment(self):
        """Gather config and launch the background worker."""
        log = self.query_one(Log)

        challenge = self.query_one(Select).value
        if not challenge:
            log.write_line("[bold red]Please select a challenge.[/bold red]")
            return

        selected_models = [cb.label for cb in self.query(Checkbox) if cb.value]
        if not selected_models:
            log.write_line("[bold red]Please select at least one model.[/bold red]")
            return

        patience = "low"
        if self.query_one("#patience_medium").variant == "primary":
            patience = "medium"
        elif self.query_one("#patience_high").variant == "primary":
            patience = "high"

        config = {
            "run_type": "comparison",
            "challenge_id": challenge,
            "patience_level": patience,
            "models": selected_models,
            "smoke_test": False
        }

        # --- Reset UI and Run ---
        self.query_one(DataTable).clear()
        log.clear()
        log.write_line(f"[bold]--- Starting Experiment ---[/bold]")
        log.write_line(f"Config: {config}")

        self.query_one("#start-button").label = "🛑 Stop Experiment"
        self.query_one("#pause-button").disabled = False

        self.experiment_worker = TUIExperimentWorker(config, self)
        self.run_worker(self.experiment_worker.run, exclusive=True, thread=True)

    # --- Message Handlers ---
    def on_progress_update(self, message: ProgressUpdate) -> None:
        """Handle progress updates from the worker thread."""
        log = self.query_one(Log)
        event = message.event_type
        data = message.data

        if event == 'start_algorithm':
            self.current_algorithm = data.get('algorithm', '')
            self.algorithm_steps[self.current_algorithm] = 0
            log.write_line(f"[bold blue]--- Running Algorithm: {self.current_algorithm} ---[/bold blue]")
        elif event == 'trainer:train_batch':
            metrics = data.get('metrics', {})
            self._update_metrics_table(metrics)
            self.algorithm_steps[self.current_algorithm] += 1
        elif event == 'end_algorithm':
            log.write_line(f"[bold]--- Finished Algorithm: {self.current_algorithm} ---[/bold]")
            self._update_metrics_table(data.get('final_metrics', {}))
        else:
            log.write_line(f"[dim]{event}[/dim]")

    def on_experiment_finished(self, message: ExperimentFinished) -> None:
        """Handle the experiment finished message."""
        self.query_one(Log).write_line(f"\n[bold green]🎉 Experiment Finished! 🎉[/bold green]")

    def on_worker_finished(self, message: WorkerFinished) -> None:
        """Handle the worker finished message."""
        self.query_one(Log).write_line("[bold]--- Worker Finished ---[/bold]")
        self.experiment_worker = None
        start_button = self.query_one("#start-button")
        start_button.label = "🚀 Start Experiment"
        start_button.disabled = False
        pause_button = self.query_one("#pause-button")
        pause_button.label = "⏸️ Pause"
        pause_button.disabled = True

    def on_error_occurred(self, message: ErrorOccurred) -> None:
        """Handle an error from the worker."""
        if "cancelled by user" in message.error_message:
            self.query_one(Log).write_line(f"[bold yellow]⚠️ {message.error_message}[/bold yellow]")
        else:
            self.query_one(Log).write_line(f"[bold red]❌ ERROR: {message.error_message}[/bold red]")

    def _update_metrics_table(self, metrics: dict):
        """Updates the DataTable with new metrics."""
        if not metrics: return
        table = self.query_one(DataTable)
        current_metrics = {table.get_cell(row_key, "Metric"): row_key for row_key, _ in table.rows.items()}

        for key, value in metrics.items():
            display_value = f"{value:.4f}" if isinstance(value, float) else str(value)
            if key in current_metrics:
                table.update_cell(current_metrics[key], "Value", display_value)
            else:
                table.add_row(str(key), display_value, key=str(key))


if __name__ == "__main__":
    app = DiscoveryTUI()
    app.run()
