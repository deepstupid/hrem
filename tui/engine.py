from textual.worker import Worker
from textual.widget import Widget
from textual.app import get_current_app
from textual.worker import get_current_worker

from sc_engine.core.progress_handler import ProgressHandler
from sc_engine.core.model_runner import ScientificModelRunner
from tui.control import ExperimentControl # This needs to be created
from .events import (
    ExperimentStarted, ExperimentFinished, ExperimentFailed,
    LogUpdate, PhaseUpdate, AlgorithmUpdate, OptimizationUpdate, ResultsGenerated
)

from typing import Dict, Any

class TUIProgressHandler(ProgressHandler):
    """A progress handler that translates engine events into TUI messages."""

    def __init__(self):
        try:
            self.worker = get_current_worker()
            self.app = get_current_app()
        except RuntimeError:
            # This can happen if not initialized in a worker context.
            # We'll handle this gracefully, though it shouldn't occur in our design.
            self.worker = None
            self.app = None

    def on_progress(self, event_type: str, data: Dict[str, Any]):
        """Receives an event and posts a corresponding message to the TUI."""
        if not self.worker or self.worker.is_cancelled:
            raise InterruptedError("TUI experiment cancelled by user.")

        # --- Translate engine events into TUI events ---
        if event_type == 'start_phase':
            phase = data.get('phase', 'Unknown Phase')
            self.app.post_message(PhaseUpdate(phase_name=phase))
            self.app.post_message(LogUpdate(f"Entering phase: [bold cyan]{phase}[/bold cyan]", style="bold"))

        elif event_type == 'start_algorithm':
            alg = data.get('algorithm', 'Unknown Algorithm')
            self.app.post_message(AlgorithmUpdate(algorithm_name=alg, status='running'))
            self.app.post_message(LogUpdate(f"Running algorithm: [bold green]{alg}[/bold green]"))

        elif event_type == 'end_algorithm':
            alg = data.get('algorithm', 'Unknown Algorithm')
            metrics = data.get('metrics', {})
            self.app.post_message(AlgorithmUpdate(algorithm_name=alg, status='finished', metrics=metrics))
            self.app.post_message(LogUpdate(f"Finished algorithm: [bold green]{alg}[/bold green]"))

        elif event_type == 'trainer:train_batch':
            # This is a high-frequency event, good for live metrics
            pass # For now, we'll handle metrics at the end of the algorithm run

        elif event_type == 'start_trial':
            # Potentially useful for detailed optimization view
            pass

        elif event_type == 'end_trial':
            # For now, we'll just log this
            loss = data.get('loss', float('inf'))
            self.app.post_message(LogUpdate(f"  Trial finished. Loss: {loss:.4f}"))

        elif event_type == 'insights_generated':
            self.app.post_message(ResultsGenerated(
                insights=data.get('insights', []),
                plot_path=data.get('plot_path'),
                report_path=data.get('report_path')
            ))
            self.app.post_message(LogUpdate("✅ Scientific insights generated.", style="bold green"))

        elif event_type == 'no_insights':
            self.app.post_message(LogUpdate("No significant insights were generated from this run.", style="yellow"))

        else:
            # Generic log for other events for now
            self.app.post_message(LogUpdate(f"Event: [dim]{event_type}[/dim]"))


class ExperimentRunner(Widget):
    """A widget to manage running experiments in a background worker."""

    def __init__(self) -> None:
        super().__init__()
        self.experiment_worker: Worker | None = None
        self.experiment_control = ExperimentControl()

    def start_experiment(self, config: Dict[str, Any]) -> None:
        """Starts the experiment in a background worker."""
        if self.experiment_worker is not None and self.experiment_worker.state == "running":
            return

        self.post_message(ExperimentStarted(config=config))
        self.experiment_worker = self.run_experiment(config)

    def pause_experiment(self) -> None:
        """Pauses the running experiment."""
        self.experiment_control.pause()

    def resume_experiment(self) -> None:
        """Resumes a paused experiment."""
        self.experiment_control.resume()

    def cancel_experiment(self) -> None:
        """Cancels the currently running experiment."""
        if self.experiment_worker is not None and self.experiment_worker.state == "running":
            self.experiment_worker.cancel()

    @work(exclusive=True, thread=True)
    def run_experiment(self, config: Dict[str, Any]) -> None:
        """The background worker method that runs the actual experiment."""
        try:
            progress_handler = TUIProgressHandler()
            runner = ScientificModelRunner()
            runner.run(
                progress_handler=progress_handler,
                control=self.experiment_control,
                **config
            )
            self.app.post_message(ExperimentFinished())
        except InterruptedError:
            # This is a clean cancellation, not an error.
            self.app.post_message(LogUpdate("Experiment cancelled by user.", style="yellow"))
            self.app.post_message(ExperimentFinished()) # Still "finished", just early
        except Exception as e:
            # This is a real error.
            self.app.post_message(ExperimentFailed(error=e))
