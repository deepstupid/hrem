from textual.widgets import (
    Static, Button, Checkbox, RadioSet, Select
)
from textual.containers import VerticalScroll, Vertical, Horizontal
from textual.message import Message
from textual.app import ComposeResult
from typing import List, Dict, Any

from sc_engine.core.config_manager import ConfigManager
from sc_engine.core.challenge_registry import ChallengeRegistry

class SetupView(Static):
    """The view for configuring and launching a new experiment."""

    class StartExperiment(Message):
        """Message to signal that the user wants to start an experiment."""
        def __init__(self, config: Dict[str, Any]) -> None:
            self.config = config
            super().__init__()

    def compose(self) -> ComposeResult:
        """Render the setup view."""
        with Vertical(id="setup-container"):
            yield Static("1. Select Challenge", classes="header")
            yield Select([], id="challenge-select", prompt="Select a challenge...")

            yield Static("2. Select Models", classes="header")
            with VerticalScroll(id="model-checkboxes"):
                # This will be populated on_mount
                pass

            yield Static("3. Set Patience Level", classes="header")
            with RadioSet(id="patience-radioset"):
                yield Button("Low", id="patience_low", variant="primary")
                yield Button("Medium", id="patience_medium")
                yield Button("High", id="patience_high")

            yield Static() # Spacer
            yield Button("🚀 Launch Experiment", id="start-button", variant="success")

    def on_mount(self) -> None:
        """Populates the control widgets with data from the config files."""
        try:
            config_manager = ConfigManager()
            challenge_registry = ChallengeRegistry(config_manager)

            # Populate challenges
            challenges = challenge_registry.get_all_challenges()
            challenge_select = self.query_one(Select)
            challenge_select.set_options([(c.name, c.id) for c in challenges])
            if challenges:
                challenge_select.value = challenges[0].id

            # Populate models
            model_configs = config_manager.load_model_configs()
            model_checkboxes_container = self.query_one("#model-checkboxes")
            for model_name in sorted(model_configs.keys()):
                checkbox = Checkbox(model_name, id=f"model_{model_name}")
                # Pre-select the main models for convenience
                if model_name in ["HRM", "HREM"]:
                    checkbox.value = True
                model_checkboxes_container.mount(checkbox)
        except Exception as e:
            # If config files are missing/malformed, we can't proceed.
            self.query_one("#start-button").disabled = True
            self.mount(Static(f"[bold red]Error loading configuration: {e}[/bold red]"))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press events to configure and start the experiment."""
        if event.button.id.startswith("patience_"):
            # Deselect all patience buttons
            for btn in self.query(Button):
                if btn.id.startswith("patience_"):
                    btn.variant = "default"
            # Select the pressed one
            event.button.variant = "primary"

        elif event.button.id == "start-button":
            challenge = self.query_one(Select).value
            if not challenge:
                # Handle error: no challenge selected
                return

            selected_models = [cb.label for cb in self.query(Checkbox) if cb.value]
            if not selected_models:
                # Handle error: no models selected
                return

            patience = "low"
            if self.query_one("#patience_medium").variant == "primary":
                patience = "medium"
            elif self.query_one("#patience_high").variant == "primary":
                patience = "high"

            config = {
                "run_type": "comparison",
                "challenge_id": str(challenge),
                "patience_level": patience,
                "models": selected_models,
                "smoke_test": False, # We can add a checkbox for this later
            }
            self.post_message(self.StartExperiment(config))
