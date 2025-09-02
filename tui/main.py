from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import (
    Header, Footer, Static, Button, Checkbox, RadioSet, TabbedContent, TabPane, Log, DataTable, Select
)
from textual.worker import Worker, get_current_worker, WorkerState
from textual import work
import time
from enum import Enum, auto

# Import the core engine components
from sc_engine.core.config_manager import ConfigManager
from sc_engine.core.challenge_registry import ChallengeRegistry
from sc_engine.core.model_runner import ScientificModelRunner
from sc_engine.core.progress_handler import ProgressHandler
from typing import Dict, Any, Optional

from .control import ExperimentControl

# --- App State Management ---

class ExperimentState(Enum):
    """Represents the possible states of the experiment UI."""
    IDLE = auto()
    RUNNING = auto()
    PAUSED = auto()

# --- TUI-Specific Progress Handler ---

class TuiProgressHandler(ProgressHandler):
    """A ProgressHandler that bridges the engine's events to the TUI thread."""
    def __init__(self):
        self.worker = get_current_worker()

    def on_progress(self, event_type: str, data: Dict[str, Any]):
        """Receives progress and posts a message to the TUI, checking for cancellation."""
        if self.worker.is_cancelled:
            # This will be caught by the run_experiment worker method
            raise InterruptedError("TUI experiment cancelled by user.")

        # Send the event data back to the main TUI thread for processing
        self.worker.app.post_message(
            self.ProgressUpdate(event_type=event_type, data=data)
        )


class DiscoveryTUI(App):
    """A Textual UI for the Scientific Discovery Engine."""

    # --- TUI Setup ---
    CSS_PATH = "styles.css"

    BINDINGS = [
        ("d", "toggle_dark", "Toggle dark mode"),
        ("q", "quit", "Quit")
    ]

    # --- Custom TUI Messages ---
    from textual.message import Message

    class ProgressUpdate(Message):
        """A message to publish progress updates from the worker."""
        def __init__(self, event_type: str, data: Dict[str, Any]) -> None:
            self.event_type = event_type
            self.data = data
            super().__init__()

    # --- App State ---
    def __init__(self):
        super().__init__()
        self.model_runner = ScientificModelRunner()
        self.current_algorithm = ""
        self.algorithm_steps = {}
        self.state = ExperimentState.IDLE
        self.experiment_control: Optional[ExperimentControl] = None
        self.experiment_worker: Optional[Worker] = None

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

                yield Button("🚀 Start", id="start-button", variant="primary")
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
        self._set_state(ExperimentState.IDLE) # Set initial button state

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

        self.query_one("#patience_low").variant = "primary"

    # --- State Management & Workers ---
    def _set_state(self, state: ExperimentState):
        """Update the application state and UI elements."""
        self.state = state
        start_button = self.query_one("#start-button")
        pause_button = self.query_one("#pause-button")

        if state == ExperimentState.IDLE:
            start_button.label = "🚀 Start"
            start_button.variant = "primary"
            pause_button.label = "⏸️ Pause"
            pause_button.disabled = True
        elif state == ExperimentState.RUNNING:
            start_button.label = "🔁 Cancel"
            start_button.variant = "error"
            pause_button.label = "⏸️ Pause"
            pause_button.disabled = False
            pause_button.variant = "primary"
        elif state == ExperimentState.PAUSED:
            start_button.label = "🔁 Cancel"
            start_button.variant = "error"
            pause_button.label = "▶️ Resume"
            pause_button.variant = "success"
            pause_button.disabled = False


    async def action_quit(self):
        """Gracefully shuts down the experiment worker on quit."""
        if self.experiment_worker is not None and self.experiment_worker.state == WorkerState.RUNNING:
            self.query_one(Log).write_line("[bold yellow]Cancelling running experiment...[/bold yellow]")
            self.experiment_worker.cancel()
            await self.experiment_worker.wait()
        self.exit()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press events."""
        if event.button.id == "start-button":
            self._start_or_cancel_experiment()
        elif event.button.id == "pause-button":
            self._toggle_pause_experiment()

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        """Reset state when worker is done."""
        if event.worker.name == "run_experiment":
            if event.worker.state in (WorkerState.SUCCESS, WorkerState.CANCELLED, WorkerState.ERROR):
                self.experiment_worker = None
                self._set_state(ExperimentState.IDLE)


    @work(exclusive=True, thread=True, name="run_experiment")
    def run_experiment(self, config: dict, control: ExperimentControl) -> None:
        """Runs the experiment in a background thread."""
        try:
            progress_handler = TuiProgressHandler()
            self.model_runner.run(
                progress_handler=progress_handler,
                control=control,
                **config
            )
        except InterruptedError:
            self.post_message(self.ProgressUpdate(event_type="info", data={"message": "Experiment cancelled by user."}))
        except Exception as e:
            self.post_message(self.ProgressUpdate(event_type="error", data={"message": f"An unexpected error occurred: {e}"}))

    def _start_or_cancel_experiment(self):
        """Starts a new experiment or cancels an existing one."""
        if self.experiment_worker is not None and self.experiment_worker.state == WorkerState.RUNNING:
            self.query_one(Log).write_line("[bold yellow]Cancelling experiment...[/bold yellow]")
            self.experiment_worker.cancel()
            return

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
        if self.query_one("#patience_medium").variant == "primary": patience = "medium"
        elif self.query_one("#patience_high").variant == "primary": patience = "high"
        config = {"run_type": "comparison", "challenge_id": challenge, "patience_level": patience, "models": selected_models, "smoke_test": False}

        self.query_one(DataTable).clear()
        log.clear()
        log.write_line(f"[bold]--- Starting Experiment ---[/bold]")
        log.write_line(f"Config: {config}")

        self.experiment_control = ExperimentControl()
        self._set_state(ExperimentState.RUNNING)
        self.experiment_worker = self.run_experiment(config, self.experiment_control)

    def _toggle_pause_experiment(self):
        """Toggles the paused state of the running experiment."""
        if self.state == ExperimentState.RUNNING:
            self.experiment_control.pause()
            self._set_state(ExperimentState.PAUSED)
            self.query_one(Log).write_line("[bold yellow]⏸️ Experiment Paused.[/bold yellow]")
        elif self.state == ExperimentState.PAUSED:
            self.experiment_control.resume()
            self._set_state(ExperimentState.RUNNING)
            self.query_one(Log).write_line("[bold green]▶️ Experiment Resumed.[/bold green]")

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
            # Increment step count for plotting later
            if self.current_algorithm in self.algorithm_steps:
                self.algorithm_steps[self.current_algorithm] += 1

        elif event == 'end_algorithm':
            log.write_line(f"[bold]--- Finished Algorithm: {self.current_algorithm} ---[/bold]")
            self._update_metrics_table(data.get('final_metrics', {}))

        elif event == 'error':
            log.write_line(f"[bold red]❌ ERROR: {data.get('message', 'Unknown error')}[/bold red]")

        elif event == 'info':
            log.write_line(f"[dim]ℹ️ {data.get('message', 'Info')}[/dim]")

        elif event == 'experiment_finished':
             log.write_line(f"\n[bold green]🎉 Experiment Finished! 🎉[/bold green]")

        else:
            # For other events, just log them
            log.write_line(f"[dim]{event}[/dim]")

    def _update_metrics_table(self, metrics: dict):
        """Updates the DataTable with new metrics."""
        if not metrics: return
        table = self.query_one(DataTable)

        # Create a lookup of existing rows by metric name
        current_metrics = {table.get_cell(row_key, "Metric"): row_key for row_key, _ in table.rows.items()}

        for key, value in metrics.items():
            display_value = f"{value:.4f}" if isinstance(value, float) else str(value)

            if key in current_metrics:
                # Update existing row
                table.update_cell(current_metrics[key], "Value", display_value)
            else:
                # Add new row
                table.add_row(str(key), display_value, key=str(key))


if __name__ == "__main__":
    app = DiscoveryTUI()
    app.run()
