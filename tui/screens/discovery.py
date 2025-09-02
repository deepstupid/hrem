from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import Header, Footer, Log, Static
from textual.containers import Container
from typing import Any

from tui.widgets.setup_view import SetupView
from tui.widgets.run_view import RunView
from tui.widgets.metrics_panel import MetricsPanel
from tui.screens.results import ResultsScreen
from tui.engine import ExperimentRunner
from tui.events import (
    LogUpdate, PhaseUpdate, AlgorithmUpdate, ResultsGenerated, ExperimentFinished, ExperimentFailed,
    PauseExperiment, ResumeExperiment
)

class DiscoveryScreen(Screen):
    """
    The main screen for setting up, running, and viewing experiment results.
    This screen manages view transitions and orchestrates the experiment lifecycle.
    """

    def compose(self) -> ComposeResult:
        """Render the discovery screen."""
        yield Header()
        with Container(id="discovery-container"):
            yield ExperimentRunner()
            yield SetupView()
            yield RunView(classes="hidden")
        yield Footer()

    def _reset_to_setup_view(self):
        """Resets the screen to its initial state, ready for a new experiment."""
        self.query_one(SetupView).remove_class("hidden")
        self.query_one(RunView).add_class("hidden")
        # Clear the metrics and log from the previous run
        self.query_one(MetricsPanel).clear_metrics()
        self.query_one(Log).clear()
        self.query_one("#status-tracker").update("Status: Idle")
        self.query_one("#phase-tracker").update("Phase: Not Started")

    # --- Event Handlers for Lifecycle ---

    def on_setup_view_start_experiment(self, message: SetupView.StartExperiment) -> None:
        """Starts the experiment when requested by the SetupView."""
        self.query_one(SetupView).add_class("hidden")
        self.query_one(RunView).remove_class("hidden")
        self.query_one(ExperimentRunner).start_experiment(message.config)
        self.query_one(Log).write("[bold green]🚀 Experiment starting...[/bold green]")
        self.query_one("#status-tracker").update("Status: Running")

    def on_run_view_cancel_experiment(self, message: RunView.CancelExperiment) -> None:
        """Cancels the experiment when requested by the RunView."""
        self.query_one(ExperimentRunner).cancel_experiment()
        self.query_one(Log).write("[bold yellow]🟡 Cancellation request sent.[/bold yellow]")
        self.query_one("#status-tracker").update("Status: Cancelling")

    def on_pause_experiment(self, message: PauseExperiment) -> None:
        """Handles a request to pause the experiment."""
        self.query_one(ExperimentRunner).pause_experiment()
        self.query_one(Log).write("[bold yellow]⏸️ Experiment Paused.[/bold yellow]")
        self.query_one("#status-tracker").update("Status: Paused")

    def on_resume_experiment(self, message: ResumeExperiment) -> None:
        """Handles a request to resume the experiment."""
        self.query_one(ExperimentRunner).resume_experiment()
        self.query_one(Log).write("[bold green]▶️ Experiment Resumed.[/bold green]")
        self.query_one("#status-tracker").update("Status: Running")

    def on_results_generated(self, message: ResultsGenerated) -> None:
        """Displays the results screen when the experiment successfully generates insights."""
        self.query_one("#status-tracker").update("Status: Finished")

        def _reset_on_dismiss(result: Any):
            """A callback to reset the discovery screen when the results screen is dismissed."""
            self._reset_to_setup_view()

        results_screen = ResultsScreen(
            insights=message.insights,
            plot_path=message.plot_path,
            report_path=message.report_path
        )
        self.app.push_screen(results_screen, _reset_on_dismiss)

    def on_experiment_finished(self, message: ExperimentFinished) -> None:
        """
        Handles a clean experiment finish, typically after cancellation or if no
        insights were generated.
        """
        # This event is now primarily for runs that don't generate a ResultsGenerated event.
        self.query_one(Log).write("\n[bold green]✅ Experiment Concluded.[/bold green]")
        self._reset_to_setup_view()

    def on_experiment_failed(self, message: ExperimentFailed) -> None:
        """Handles a crashed experiment by displaying a dedicated error screen."""
        self.query_one("#status-tracker").update("Status: Failed")

        def _reset_on_dismiss(result: Any):
            """A callback to reset the discovery screen when the error screen is dismissed."""
            self._reset_to_setup_view()

        self.app.push_screen(ErrorScreen(error=message.error), _reset_on_dismiss)

    # --- Event Handlers for Live Updates ---

    def on_log_update(self, message: LogUpdate) -> None:
        """Writes a message to the live log."""
        self.query_one(Log).write(f"[{message.style}]{message.message}[/]")

    def on_phase_update(self, message: PhaseUpdate) -> None:
        """Updates the phase tracker display."""
        self.query_one("#phase-tracker").update(f"Phase: {message.phase_name.replace('_', ' ').title()}")

    def on_algorithm_update(self, message: AlgorithmUpdate) -> None:
        """Updates the metrics panel with new data."""
        if message.metrics:
            self.query_one(MetricsPanel).update_metrics(message.algorithm_name, message.metrics)
