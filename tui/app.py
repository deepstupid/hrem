from textual.app import App

from .screens.welcome import WelcomeScreen
from .screens.discovery import DiscoveryScreen

class DiscoveryTUI(App):
    """The main application for the Scientific Discovery Engine TUI."""

    CSS_PATH = "styles.css"

    SCREENS = {
        "welcome": WelcomeScreen,
        "discovery": DiscoveryScreen,
    }

    BINDINGS = [
        ("q", "quit", "Quit"),
    ]

    def on_mount(self) -> None:
        """Called when the app is first mounted. Shows the welcome screen."""
        self.push_screen("welcome")

    def on_welcome_screen_proceed(self, message: WelcomeScreen.Proceed) -> None:
        """
        Handles the message from the welcome screen to proceed to the main interface.
        """
        self.switch_screen("discovery")
