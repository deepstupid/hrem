import time
from PyQt6.QtCore import QObject, pyqtSignal
from sc_engine.core.model_runner import ScientificModelRunner

class ExperimentCancelledError(Exception):
    """Custom exception for when an experiment is cancelled by the user."""
    pass

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

    def stop(self):
        """Signals the worker to stop the experiment."""
        self._is_cancelled = True

    def run(self):
        """
        Executes the experiment. This method is meant to be run in a QThread.
        """
        try:
            def progress_handler(payload: dict):
                if self._is_cancelled:
                    raise ExperimentCancelledError("Experiment cancelled by user.")
                self.progress_updated.emit(payload)

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
            self.error_occurred.emit(str(e))
        finally:
            self.finished.emit()
