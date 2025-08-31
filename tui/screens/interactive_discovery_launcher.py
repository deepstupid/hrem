from textual.app import ComposeResult
from textual.widgets import Static, Input, Button
from textual.containers import Vertical
from textual.message import Message

class InteractiveDiscoveryLauncherScreen(Static):
    """A screen to configure and launch interactive discovery runs."""

    class StartRun(Message):
        """Message to signal starting a run with a specific configuration."""
        def __init__(self, challenge_id: str):
            self.challenge_id = challenge_id
            super().__init__()

    def compose(self) -> ComposeResult:
        """Create child widgets for the screen."""
        yield Static("Configure Interactive Run", classes="header")
        with Vertical(id="run-config-form"):
            yield Input(placeholder="Enter Challenge ID (e.g., synthetic_sort)", id="challenge-id-input", value="quick_comparison")
            yield Button("Start Discovery", variant="primary", id="start-discovery-btn")

    def on_button_pressed(self, event: Button.Pressed):
        """Handle the button press to start a run."""
        if event.button.id == "start-discovery-btn":
            challenge_id_input = self.query_one("#challenge-id-input", Input)
            challenge_id = challenge_id_input.value
            if challenge_id:
                self.post_message(self.StartRun(challenge_id))
            else:
                challenge_id_input.border_title = "Challenge ID cannot be empty"
                challenge_id_input.styles.border = ("round", "red")
