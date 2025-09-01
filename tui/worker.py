import time
from typing import Dict, Any, Optional

from textual.app import App
from textual.message import Message

from sc_engine.core.model_runner import ScientificModelRunner
from sc_engine.core.progress_handler import ProgressHandler

# --- Custom Messages for TUI <-> Worker Communication ---

class ProgressUpdate(Message):
    """A message to publish progress updates from the worker."""
    def __init__(self, event_type: str, data: Dict[str, Any]) -> None:
        self.event_type = event_type
        self.data = data
        super().__init__()

class ExperimentFinished(Message):
    """A message to signal the successful completion of an experiment."""
    def __init__(self, results: Any) -> None:
        self.results = results
        super().__init__()

class WorkerFinished(Message):
    """A message to signal that the worker thread has finished."""
    pass

class ErrorOccurred(Message):
    """A message to signal that an error occurred in the worker."""
    def __init__(self, error_message: str) -> None:
        self.error_message = error_message
        super().__init__()

# --- Worker-Related Classes ---

class ExperimentCancelledError(Exception):
    """Custom exception for when an experiment is cancelled by the user."""
    pass

class TuiProgressHandler(ProgressHandler):
    """A ProgressHandler that bridges the engine's events to the TUI."""
    def __init__(self, worker: 'TUIExperimentWorker'):
        self.worker = worker

    def on_progress(self, event_type: str, data: Dict[str, Any]):
        """Receives progress and posts a message, checking for cancellation and pause."""
        if self.worker.is_cancelled:
            raise ExperimentCancelledError("Experiment cancelled by user.")

        while self.worker.is_paused:
            time.sleep(0.1)

        self.worker.app.post_message(ProgressUpdate(event_type, data))

class TUIExperimentWorker:
    """
    A worker object that runs the scientific experiment in a background thread,
    communicating with the TUI via messages.
    """
    def __init__(self, config: dict, app: App):
        self.config = config
        self.app = app
        self.model_runner = ScientificModelRunner()
        self.is_cancelled = False
        self.is_paused = False

    def stop(self):
        """Signals the worker to stop the experiment."""
        self.is_cancelled = True

    def pause(self):
        """Signals the worker to pause the experiment."""
        self.is_paused = True

    def resume(self):
        """Signals the worker to resume the experiment."""
        self.is_paused = False

    def run(self):
        """
        Executes the experiment. This method is meant to be run in a Textual Worker.
        """
        try:
            progress_handler = TuiProgressHandler(self)
            results = self.model_runner.run(
                progress_handler=progress_handler,
                **self.config
            )
            if not self.is_cancelled:
                self.app.post_message(ExperimentFinished(results))
        except ExperimentCancelledError as e:
            self.app.post_message(ErrorOccurred(str(e)))
        except Exception as e:
            error_message = f"{type(e).__name__}: {e}"
            if not self.is_cancelled:
                self.app.post_message(ErrorOccurred(error_message))
        finally:
            self.app.post_message(WorkerFinished())
