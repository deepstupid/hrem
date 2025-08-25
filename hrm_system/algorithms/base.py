import abc
from typing import Dict, Any, Callable

from hrm_system.config import ModelConfig, TrainingConfig

class Algorithm(abc.ABC):
    """
    Abstract base class for a trainable algorithm.
    This interface allows the main system to treat any model as a pluggable component.
    """

    @abc.abstractmethod
    def __init__(self, model_config: ModelConfig, training_config: TrainingConfig):
        """
        Initializes the algorithm, including model creation.

        Args:
            model_config: Configuration for the model architecture.
            training_config: Configuration for the training process.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def train(self, data_path: str, logger_callback: Callable[[str], None]) -> Dict[str, Any]:
        """
        Runs the training loop for the algorithm.

        Args:
            data_path: The path to the dataset.
            logger_callback: A callback function for logging output.

        Returns:
            A dictionary containing the final training metrics.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def evaluate(self, data_path: str) -> Dict[str, Any]:
        """
        Runs the evaluation for the algorithm.

        Args:
            data_path: The path to the dataset for evaluation.

        Returns:
            A dictionary containing the evaluation metrics.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def load_checkpoint(self, path: str):
        """
        Loads the model state from a checkpoint file.

        Args:
            path: The path to the checkpoint file.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def save_checkpoint(self, path: str):
        """
        Saves the model state to a checkpoint file.

        Args:
            path: The path to save the checkpoint file.
        """
        raise NotImplementedError
