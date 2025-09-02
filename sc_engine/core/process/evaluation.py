import time
from typing import List, Dict, Any, Callable, Tuple

from ..config import AlgorithmConfig
from ..patience_manager import ExplorationPhase
from ..trainer import Trainer


class Evaluation:
    """Handles the evaluation of algorithms."""

    def __init__(self, engine):
        self.engine = engine

    def run_evaluation(self,
                        title: str,
                        phase: ExplorationPhase,
                        patience_allocation: float,
                        run_suffix: str,
                        model_config_fn: Callable[[AlgorithmConfig], Dict[str, Any]],
                        result_key_fn: Callable[[AlgorithmConfig], str],
                        timing_value: int,
                        interleaved: bool = False
                        ) -> Tuple[Dict[str, Any], Dict[str, List]]:
        self.engine.send_progress('start_evaluation', {'title': title})
        self.engine.patience_manager.allocate_for_phase(phase, patience_allocation)
        start_time = time.time()

        trainers = self._initialize_trainers(run_suffix, model_config_fn)

        if interleaved:
            results, histories = self._run_interleaved_evaluation(trainers, result_key_fn)
        else:
            results, histories = self._run_sequential_evaluation(trainers, result_key_fn)

        elapsed_time = time.time() - start_time
        self.engine.patience_manager.update_patience_consumption(elapsed_time, phase)
        self.engine.timing_manager.record_discovery_timing(f"{run_suffix}_evaluation", elapsed_time, timing_value)
        self.engine.send_progress('end_evaluation', {'title': title, 'results': results})
        return results, histories

    def _initialize_trainers(self, run_suffix: str, model_config_fn: Callable[[AlgorithmConfig], Dict[str, Any]]) -> List[Trainer]:
        """Initializes and returns a list of trainers for the current algorithms."""
        trainers = []
        smoke_test = self.engine.config.get("smoke_test", False)
        for alg in self.engine.algorithms:
            run_config = {
                "study_name": f"{self.engine.challenge.id}_{run_suffix}_{alg.name}",
                "output_dir": "experiments",
                "smoke_test": smoke_test
            }
            model_config = model_config_fn(alg)
            trainer = Trainer(
                self.engine.default_training_config,
                model_config,
                self.engine.challenge.dataset,
                run_config,
                progress_handler=self.engine.progress_handler,
                cancel_event=self.engine.cancel_event
            )
            trainer.initialize()
            trainers.append(trainer)
        return trainers

    def _run_interleaved_evaluation(self, trainers: List[Trainer], result_key_fn: Callable[[AlgorithmConfig], str]) -> Tuple[Dict[str, Any], Dict[str, List]]:
        """Runs an interleaved evaluation across all trainers."""
        results = {}
        histories = {}
        num_steps = trainers[0].train_state.total_steps if trainers else 0
        for step in range(num_steps):
            for i, trainer in enumerate(trainers):
                alg = self.engine.algorithms[i]
                self.engine.send_progress('start_algorithm_step', {'algorithm': alg.name, 'step': step})
                _, is_finished = trainer.train_batch()
                if is_finished:
                    break

        for i, trainer in enumerate(trainers):
            alg = self.engine.algorithms[i]
            metrics = trainer.evaluate(trainer.run_config['output_dir'])
            results[result_key_fn(alg)] = metrics
            histories[result_key_fn(alg)] = []  # History not collected per-step in this mode
            self.engine.send_progress('end_algorithm', {'algorithm': alg.name, 'metrics': metrics})

        return results, histories

    def _run_sequential_evaluation(self, trainers: List[Trainer], result_key_fn: Callable[[AlgorithmConfig], str]) -> Tuple[Dict[str, Any], Dict[str, List]]:
        """Runs a sequential evaluation for each trainer."""
        results = {}
        histories = {}
        for i, trainer in enumerate(trainers):
            if self.engine.cancel_event.is_set():
                self.engine.send_progress('evaluation_cancelled', {'reason': 'Cancelled by user.'})
                break
            alg = self.engine.algorithms[i]
            self.engine.send_progress('start_algorithm', {'algorithm': alg.name, 'progress': (i + 1) / len(self.engine.algorithms)})
            metrics, history = trainer.run_sequential_training()
            results[result_key_fn(alg)] = metrics
            histories[result_key_fn(alg)] = history
            self.engine.send_progress('end_algorithm', {'algorithm': alg.name, 'metrics': metrics})
        return results, histories
