from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseUIRunner(ABC):
    """
    Abstract base class for a UI runner. It defines a standard interface
    for how UIs interact with the scientific discovery engine.
    """

    @abstractmethod
    def run_interactive(self):
        """
        Run the scientific discovery engine in interactive mode.
        """
        pass

    @abstractmethod
    def run_once(self, challenge_id: str, models: List[str], patience: str, smoke_test: bool):
        """
        Run the scientific discovery engine in non-interactive mode.
        """
        pass

    @abstractmethod
    def list_challenges(self):
        """
        List all available scientific challenges.
        """
        pass

    @abstractmethod
    def list_models(self):
        """
        List all available models.
        """
        pass

    @abstractmethod
    def list_datasets(self):
        """
        List all available datasets.
        """
        pass
