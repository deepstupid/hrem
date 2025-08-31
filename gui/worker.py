from PyQt6.QtCore import QObject, pyqtSignal
from sc_engine.core.model_runner import ScientificModelRunner

class ExperimentWorker(QObject):
    """
    A worker object that runs the scientific experiment in a separate thread.
    """
    progress_updated = pyqtSignal(dict)
    experiment_finished = pyqtSignal(object)  # Using 'object' for DiscoveryResults
    error_occurred = pyqtSignal(str)

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.model_runner = ScientificModelRunner()

    def run(self):
        """
        Executes the experiment. This method is meant to be run in a QThread.
        """
        try:
            # The progress_handler will be a simple lambda that emits the signal
            def progress_handler(payload: dict):
                self.progress_updated.emit(payload)

            results = self.model_runner.run(
                progress_handler=progress_handler,
                **self.config
            )
            self.experiment_finished.emit(results)
        except Exception as e:
            self.error_occurred.emit(str(e))
