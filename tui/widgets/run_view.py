from textual.widgets import Static, Log, Button
from textual.containers import Vertical, Horizontal
from textual.app import ComposeResult
from textual.message import Message
from textual.reactive import reactive

from .metrics_panel import MetricsPanel
from tui.events import PauseExperiment, ResumeExperiment, CancelExperiment

class RunView(Static):
    """The view for monitoring a running experiment."""

    is_paused = reactive(False)

    def compose(self) -> ComposeResult:
        """Render the run view."""
        with Vertical(id="run-container"):
            with Horizontal(id="run-header"):
                yield Static("Phase: Not Started", id="phase-tracker")
                yield Static("Status: Idle", id="status-tracker")
            with Horizontal(id="run-content"):
                yield MetricsPanel(id="metrics-panel")
                yield Log(id="live-log", highlight=True, auto_scroll=True)
            with Horizontal(classes="control-buttons"):
                yield Button("Pause", variant="primary", id="pause-button")
                yield Button("Cancel", variant="error", id="cancel-button")

    def on_mount(self) -> None:
        """Clear the log and metrics on mount."""
        self.query_one("#live-log").clear()
        self.query_one(MetricsPanel).clear_metrics()
        self.is_paused = False

    def watch_is_paused(self, paused: bool) -> None:
        """Update the button label when the paused state changes."""
        pause_button = self.query_one("#pause-button")
        if paused:
            pause_button.label = "Resume"
            pause_button.variant = "success"
        else:
            pause_button.label = "Pause"
            pause_button.variant = "primary"

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle control button presses."""
        if event.button.id == "pause-button":
            if self.is_paused:
                self.post_message(ResumeExperiment())
            else:
                self.post_message(PauseExperiment())
            self.is_paused = not self.is_paused
        elif event.button.id == "cancel-button":
            self.post_message(CancelExperiment())
