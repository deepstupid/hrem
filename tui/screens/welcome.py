from textual.screen import Screen
from textual.widgets import Header, Footer, Button, Static
from textual.containers import Center, Vertical
from textual.app import ComposeResult
from textual.message import Message

class WelcomeScreen(Screen):
    """The first screen the user sees. Greets them and provides a button to continue."""

    class Proceed(Message):
        """A message to signal that the user wants to proceed."""
        pass

    CSS = """
    #welcome-container {
        align: center middle;
        height: 100%;
    }
    #welcome-title {
        font-style: bold;
        padding-bottom: 2;
    }
    #welcome-subtitle {
        padding-bottom: 2;
    }
    """

    def compose(self) -> ComposeResult:
        """Render the welcome screen."""
        yield Header()
        with Center(id="welcome-container"):
            with Vertical():
                yield Static("Welcome to the Scientific Discovery Engine", id="welcome-title")
                yield Static("Press the button below to begin configuring your experiment.", id="welcome-subtitle")
                yield Button("Proceed to Setup", variant="primary", id="proceed")
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle the user pressing the proceed button."""
        if event.button.id == "proceed":
            self.post_message(self.Proceed())
