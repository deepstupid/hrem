"""The main demo screen for the HRM System TUI."""

from textual.app import ComposeResult
from textual.containers import VerticalScroll, Vertical, Horizontal
from textual.widgets import Static, Button, Select
from rich.text import Text

from demo_config_manager import EnhancedConfigManager
from unified_demo_runner import run_demo
from demo_config import DemoConfig, DemoMode

class DemoScreen(Static):
    """The main screen for running pre-configured demo challenges."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.challenges = EnhancedConfigManager.load_challenge_config()
        self.selected_challenge = self.challenges[0] if self.challenges else None

    def compose(self) -> ComposeResult:
        """Create child widgets for the demo screen."""
        challenge_options = [(c['name'], c['id']) for c in self.challenges]

        yield VerticalScroll(
            Static("🚀 HRM vs HREM Demonstration", classes="header"),
            Static("Select a pre-configured challenge to run the end-to-end demo.", classes="description"),

            Horizontal(
                Select(challenge_options, value=self.selected_challenge['id'] if self.selected_challenge else None, id="challenge_select"),
                classes="select_container"
            ),

            Static(id="challenge_details", classes="details_box"),

            Button("Run Challenge", variant="primary", id="run_challenge_button", classes="run_button"),

            Static(id="results_output")
        )

    def on_mount(self) -> None:
        """Called when the widget is mounted."""
        self.update_challenge_details()

    def on_select_changed(self, event: Select.Changed) -> None:
        """Handle select changes."""
        if event.select.id == "challenge_select":
            challenge_id = event.value
            self.selected_challenge = next((c for c in self.challenges if c['id'] == challenge_id), None)
            self.update_challenge_details()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "run_challenge_button" and self.selected_challenge:
            self.query_one("#results_output", Static).update("") # Clear previous results
            self.run_challenge()

    def update_challenge_details(self):
        """Update the challenge details box."""
        if not self.selected_challenge:
            self.query_one("#challenge_details").update("[dim]No challenge selected.[/dim]")
            return

        details = Text.assemble(
            ("Challenge: ", "bold"), f"{self.selected_challenge['name']}\n",
            ("Description: ", "bold"), f"{self.selected_challenge['description']}\n",
            ("Difficulty: ", "bold"), f"{self.selected_challenge['difficulty']}\n",
            ("Models: ", "bold"), f"{', '.join(self.selected_challenge['models'])}\n",
            ("Duration: ", "bold"), f"{self.selected_challenge['duration']}"
        )
        self.query_one("#challenge_details").update(details)

    def run_challenge(self):
        """Configure and run the selected challenge."""
        if not self.selected_challenge:
            return

        dataset_info = self.selected_challenge.get("dataset", {})
        dataset_name = dataset_info.get("dataset", "synthetic")
        task = ""
        if "synthetic" in dataset_name:
            parts = dataset_name.split("-", 1)
            if len(parts) > 1:
                task = parts[1]

        config = DemoConfig(
            demo_mode=DemoMode.COMPREHENSIVE,
            models=self.selected_challenge.get("models", ["HRM", "HREM"]),
            dataset=dataset_name.split("-")[0],
            task=task,
            patience_level=self.selected_challenge.get("patience_level", "low"),
            study_name=self.selected_challenge.get("id", "tui_demo"),
        )
        
        # This will run the full demo, printing to the console.
        # The TUI will be paused during this time.
        with self.app.suspend():
             run_demo(config)
