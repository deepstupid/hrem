from textual.message import Message
from typing import Dict, Any, List, Optional

class ExperimentEvent(Message):
    """Base class for all experiment-related events."""
    pass

class ExperimentStarted(ExperimentEvent):
    """Signals that the experiment worker has started."""
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        super().__init__()

class ExperimentFinished(ExperimentEvent):
    """Signals that the experiment has finished successfully."""
    pass

class ExperimentFailed(ExperimentEvent):
    """Signals that the experiment has failed with an exception."""
    def __init__(self, error: Exception) -> None:
        self.error = error
        super().__init__()

class LogUpdate(ExperimentEvent):
    """For sending general, formatted log messages to the UI."""
    def __init__(self, message: str, style: str = "dim") -> None:
        self.message = message
        self.style = style
        super().__init__()

class PhaseUpdate(ExperimentEvent):
    """Signals a change in the experiment phase (e.g., 'baseline', 'optimization')."""
    def __init__(self, phase_name: str) -> None:
        self.phase_name = phase_name
        super().__init__()

class AlgorithmUpdate(ExperimentEvent):
    """Provides an update on a specific algorithm's progress."""
    def __init__(self, algorithm_name: str, status: str, metrics: Optional[Dict[str, Any]] = None) -> None:
        self.algorithm_name = algorithm_name
        self.status = status
        self.metrics = metrics or {}
        super().__init__()

class OptimizationUpdate(ExperimentEvent):
    """Provides an update on a hyperparameter optimization trial."""
    def __init__(self, trial_number: int, total_trials: int, params: Dict[str, Any], loss: Optional[float]) -> None:
        self.trial_number = trial_number
        self.total_trials = total_trials
        self.params = params
        self.loss = loss
        super().__init__()

class ResultsGenerated(ExperimentEvent):
    """Sends the final results (insights, plot path) to the UI."""
    def __init__(self, insights: List[Any], plot_path: Optional[str], report_path: Optional[str]) -> None:
        self.insights = insights
        self.plot_path = plot_path
        self.report_path = report_path

class PauseExperiment(ExperimentEvent):
    """A message to signal the user wants to pause the experiment."""
    pass

class ResumeExperiment(ExperimentEvent):
    """A message to signal the user wants to resume the experiment."""
    pass
        super().__init__()
