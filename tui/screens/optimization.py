from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static

# TODO: This entire screen needs to be rewritten to use the new sc_engine
# and ScientificModelRunner.

class OptimizationScreen(Static):
    """The screen for running hyperparameter optimization."""

    def compose(self) -> ComposeResult:
        yield Vertical(
            Static("Hyperparameter Optimization", classes="header"),
            Static("This screen is currently under construction.", classes="description"),
        )
