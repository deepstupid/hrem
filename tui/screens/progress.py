from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import Header, Footer, Static

# TODO: This entire screen needs to be rewritten to use the new sc_engine
# and ScientificModelRunner.

class ProgressScreen(Screen):
    """A screen to display the progress of an experiment."""

    BINDINGS = [("escape", "app.pop_screen", "Back")]

    def compose(self) -> ComposeResult:
        """Create child widgets for the screen."""
        yield Header()
        yield Static("Experiment in progress...", classes="header")
        yield Static("This screen is currently under construction.", classes="description")
        yield Footer()
