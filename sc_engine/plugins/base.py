from abc import ABC, abstractmethod
from typing import List, Dict, Any

class AlgorithmPlugin(ABC):
    """
    Base class for algorithm plugins.
    """
    @abstractmethod
    def get_name(self) -> str:
        pass

    @abstractmethod
    def get_model_config(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def get_search_space(self) -> Dict[str, Any]:
        pass

class ChallengePlugin(ABC):
    """
    Base class for challenge plugins.
    """
    @abstractmethod
    def get_name(self) -> str:
        pass

    @abstractmethod
    def get_challenge_config(self) -> Dict[str, Any]:
        pass

class InsightGeneratorPlugin(ABC):
    """
    Base class for insight generator plugins.
    """
    @abstractmethod
    def get_name(self) -> str:
        pass

    @abstractmethod
    def generate_insights(self, data: Any) -> List[str]:
        pass
