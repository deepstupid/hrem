import time
from typing import Dict, Any, List, Tuple, Callable
import threading

from .config import AlgorithmConfig
from .patience_manager import ExplorationPhase
from .progress_handler import ProgressHandler
from .trainer import Trainer

class ModelEvaluator:
    """Evaluates a set of models for a given phase of the discovery process."""

    def __init__(self, default_training_config: Dict[str, Any], progress_handler: ProgressHandler, cancel_event: threading.Event, algorithms: List[AlgorithmConfig], challenge_id: str, challenge_dataset: Dict[str, Any]):
        self.default_training_config = default_training_config
        self.progress_handler = progress_handler
        self.cancel_event = cancel_event
        self.algorithms = algorithms
        self.challenge_id = challenge_id
        self.challenge_dataset = challenge_dataset

    def run_evaluation(self, title: str, phase: ExplorationPhase, run_suffix: str, model_config_fn: Callable, result_key_fn: Callable) -> Tuple[Dict[str, Any], Dict[str, List]]:
        """Runs a single evaluation phase."""
        self.send_progress('start_evaluation', {'title': title})

        trainers = self._initialize_trainers(run_suffix, model_config_fn)
        results, histories = self._run_sequential_evaluation(trainers, result_key_fn)

        self.send_progress('end_evaluation', {'title': title, 'results': results})
        return results, histories

    def _initialize_trainers(self, run_suffix: str, model_config_fn: Callable) -> List[Trainer]:
        trainers = []
        for alg in self.algorithms:
            run_config = {"study_name": f"{self.challenge_id}_{run_suffix}_{alg.name}", "output_dir": "experiments"}
            model_config = model_config_fn(alg)
            trainer = Trainer(self.default_training_config, model_config, self.challenge_dataset, run_config, progress_handler=self.progress_handler, cancel_event=self.cancel_event)
            trainer.initialize()
            trainers.append(trainer)
        return trainers

    def _run_sequential_evaluation(self, trainers: List[Trainer], result_key_fn: Callable) -> Tuple[Dict[str, Any], Dict[str, List]]:
        results, histories = {}, {}
        for i, trainer in enumerate(trainers):
            if self.cancel_event.is_set():
                self.send_progress('evaluation_cancelled', {'reason': 'Cancelled by user.'})
                break
            alg = self.algorithms[i]
            self.send_progress('start_algorithm', {'algorithm': alg.name, 'progress': (i + 1) / len(trainers)})
            metrics, history = trainer.run_sequential_training()
            results[result_key_fn(alg)] = metrics
            histories[result_key_fn(alg)] = history
            self.send_progress('end_algorithm', {'algorithm': alg.name, 'metrics': metrics})
        return results, histories

    def send_progress(self, event_type: str, data: Dict = None):
        if self.progress_handler:
            self.progress_handler.on_progress(event_type, data if data is not None else {})
