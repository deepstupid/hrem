"""The main demo screen for the HRM System TUI."""

from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Static, Button, Select

# TODO: This entire screen needs to be rewritten to use the new sc_engine
# and ScientificModelRunner. The old demo system seems to be deprecated.

class DemoScreen(Static):
    """The main screen for running pre-configured demo challenges."""

    def compose(self) -> ComposeResult:
        """Create child widgets for the demo screen."""
        yield VerticalScroll(
            Static("🚀 HRM vs HREM Demonstration", classes="header"),
            Static("This screen is currently under construction.", classes="description"),
        )
