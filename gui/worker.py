import time
from PyQt6.QtCore import QObject, pyqtSignal
from sc_engine.core.model_runner import ScientificModelRunner
from sc_engine.core.progress_handler import ProgressHandler
from typing import Dict, Any

class ExperimentCancelledError(Exception):
    """Custom exception for when an experiment is cancelled by the user."""
    pass

class GuiProgressHandler(ProgressHandler):
    """A ProgressHandler that bridges the engine's events to the GUI."""
    def __init__(self, worker: 'ExperimentWorker'):
        self.worker = worker

    def on_progress(self, event_type: str, data: Dict[str, Any]):
        """Receives progress and emits a signal, checking for cancellation and pause."""
        if self.worker._is_cancelled:
            raise ExperimentCancelledError("Experiment cancelled by user.")

        # Pause the execution if requested
        while self.worker._is_paused:
            # Sleep in the worker's thread to avoid blocking the GUI
            self.worker.thread().msleep(100)

        payload = {'event': event_type, 'data': data}
        self.worker.progress_updated.emit(payload)

class ExperimentWorker(QObject):
    """
    A worker object that runs the scientific experiment in a separate thread.
    """
    progress_updated = pyqtSignal(dict)
    experiment_finished = pyqtSignal(object)
    error_occurred = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.model_runner = ScientificModelRunner()
        self._is_cancelled = False
        self._is_paused = False

    def stop(self):
        """Signals the worker to stop the experiment."""
        self._is_cancelled = True

    def pause(self):
        """Signals the worker to pause the experiment."""
        self._is_paused = True

    def resume(self):
        """Signals the worker to resume the experiment."""
        self._is_paused = False

    def is_paused(self):
        return self._is_paused

    def run(self):
        """
        Executes the experiment. This method is meant to be run in a QThread.
        """
        try:
            progress_handler = GuiProgressHandler(self)

            results = self.model_runner.run(
                progress_handler=progress_handler,
                **self.config
            )
            if not self._is_cancelled:
                self.experiment_finished.emit(results)
        except ExperimentCancelledError as e:
            # Don't propagate as a full error, just a clean exit
            self.error_occurred.emit(str(e))
        except Exception as e:
            # Emit the full error message, including the type of exception
            error_message = f"{type(e).__name__}: {e}"
            if not self._is_cancelled:
                self.error_occurred.emit(error_message)
        finally:
            self.finished.emit()
