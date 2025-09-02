from textual.screen import Screen
from textual.app import ComposeResult
from textual.widgets import Header, Footer, Button
from textual.containers import Container
from typing import List, Any, Optional

from tui.widgets.insights_panel import InsightsPanel

class ResultsScreen(Screen):
    """A screen to display the final results of an experiment."""

    BINDINGS = [
        ("n", "new_experiment", "New Experiment"),
    ]

    def __init__(
        self,
        insights: List[Any],
        plot_path: Optional[str],
        report_path: Optional[str],
        log_histories: Optional[Dict[str, List[Dict[str, Any]]]],
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        self.insights = insights
        self.plot_path = plot_path
        self.report_path = report_path
        self.log_histories = log_histories
        super().__init__(name=name, id=id, classes=classes)

    def compose(self) -> ComposeResult:
        """Render the results screen."""
        yield Header()
        with Container(id="results-container"):
            yield InsightsPanel(id="insights-panel")
            yield Button("New Experiment", variant="primary", id="new-experiment-button")
        yield Footer()

    def on_mount(self) -> None:
        """Populate the insights panel with the results."""
        panel = self.query_one(InsightsPanel)
        panel.show_results(self.insights, self.plot_path, self.report_path, self.log_histories)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle the new experiment button."""
        if event.button.id == "new-experiment-button":
            self.action_new_experiment()

    def action_new_experiment(self) -> None:
        """Dismisses the results screen to return to the discovery screen."""
        self.dismiss()
