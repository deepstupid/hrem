from textual.worker import Worker
from textual.widget import Widget
from textual.app import App
from textual.worker import get_current_worker

from sc_engine.core.progress_handler import ProgressHandler
from sc_engine.core.model_runner import ScientificModelRunner
from tui.control import ExperimentControl
from .events import (
    ExperimentStarted, ExperimentFinished, ExperimentFailed,
    LogUpdate, PhaseUpdate, AlgorithmUpdate, OptimizationUpdate, ResultsGenerated,
    LiveMetricUpdate
)

from typing import Dict, Any, Optional

class TUIProgressHandler(ProgressHandler):
    """A progress handler that translates engine events into TUI messages."""

    def __init__(self, app: App):
        self.app = app
        try:
            self.worker = get_current_worker()
        except RuntimeError:
            self.worker = None
        self.current_algorithm: Optional[str] = None

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
            self.current_algorithm = alg
            self.app.post_message(AlgorithmUpdate(algorithm_name=alg, status='running'))
            self.app.post_message(LogUpdate(f"Running algorithm: [bold green]{alg}[/bold green]"))

        elif event_type == 'end_algorithm':
            alg = data.get('algorithm', 'Unknown Algorithm')
            metrics = data.get('metrics', {})
            self.app.post_message(AlgorithmUpdate(algorithm_name=alg, status='finished', metrics=metrics))
            self.app.post_message(LogUpdate(f"Finished algorithm: [bold green]{alg}[/bold green]"))
            self.current_algorithm = None

        elif event_type == 'trainer:train_batch':
            if self.current_algorithm:
                self.app.post_message(LiveMetricUpdate(
                    metrics=data.get('metrics', {}),
                    step=data.get('step', 0),
                    total_steps=data.get('total_steps', 0),
                    algorithm_name=self.current_algorithm
                ))

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
                report_path=data.get('report_path'),
                log_histories=data.get('log_histories')
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
        self.experiment_worker = self.run_worker(lambda: self.run_experiment(config), exclusive=True, thread=True)

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

    def run_experiment(self, config: Dict[str, Any]) -> None:
        """The background worker method that runs the actual experiment."""
        try:
            progress_handler = TUIProgressHandler(app=self.app)
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
