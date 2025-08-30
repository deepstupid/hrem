from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static

# TODO: This entire screen needs to be rewritten to use the new sc_engine
# and ScientificModelRunner.

class EvaluationScreen(Static):
    """The screen for running model evaluations."""

    def compose(self) -> ComposeResult:
        yield Vertical(
            Static("Model Evaluation", classes="header"),
            Static("This screen is currently under construction.", classes="description"),
        )
