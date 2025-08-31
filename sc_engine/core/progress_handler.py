from abc import ABC, abstractmethod
from typing import Dict, Any

class ProgressHandler(ABC):
    """
    Abstract base class for handling progress updates from the ScientificDiscoveryEngine.
    """

    @abstractmethod
    def on_progress(self, event_type: str, data: Dict[str, Any]):
        """
        Receives a progress update from the engine.

        Args:
            event_type: The type of event that occurred (e.g., 'start_phase', 'end_trial').
            data: A dictionary containing data associated with the event.
        """
        pass
