import time
from typing import Dict, Any, List, Optional, Tuple, Callable
import json
import yaml
import threading
import torch
import numpy as np
from rich.console import Console
from dataclasses import dataclass

from .config import ChallengeConfig, AlgorithmConfig, PatienceBudget
from .config_manager import ConfigManager
from .patience_manager import AdaptivePatienceManager, ExplorationPhase, ScientificInsight
from sc_engine.utils.plotting import generate_performance_plot
from .challenge_registry import ChallengeRegistry
from .schemas import ChallengeSchema
from .progress_handler import ProgressHandler
from .trainer import Trainer
from .interactive_runner import InteractiveRunner
from dataset_manager import dataset_manager
from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig
from dataset.common import PuzzleDatasetMetadata
from models.hrm.hrem import HREM
from .scheduler import DiscoveryAwareScheduler
from .insight_generator import ScientificInsightGenerator
from .report_generator import ScientificReportGenerator
from .timing_manager import ScientificTimingManager
from .optimization import HyperparameterOptimizer, OptunaOptimizer


console = Console()

@dataclass
class DiscoveryResults:
    """Results from a scientific discovery session."""
    challenge: ChallengeConfig
    algorithm_results: Dict[str, Any]
    insights: List[Any]
    timing_data: Dict[str, Any]
    metadata: Dict[str, Any]
    log_histories: Dict[str, List[Dict[str, Any]]] = None

class ScientificModelRunner:
    """Runner for discovery-oriented algorithm comparison."""

    def __init__(self, config_dir: str = 'config'):
        self.config_manager = ConfigManager(config_dir)
        self.challenge_registry = ChallengeRegistry(self.config_manager)
        self.model_configs = self.config_manager.load_model_configs()
        self.search_spaces = self.config_manager.load_search_spaces()
        with open("config/training/default.yaml", 'r') as f:
            self.default_training_config = yaml.safe_load(f)

        # Engine-related attributes, to be initialized in run_discovery_session
        self.challenge: Optional[ChallengeConfig] = None
        self.algorithms: Optional[List[AlgorithmConfig]] = None
        self.patience_manager: Optional[AdaptivePatienceManager] = None
        self.timing_manager: Optional[ScientificTimingManager] = None
        self.progress_handler: Optional[ProgressHandler] = None
        self.cancel_event: Optional[threading.Event] = None
        self.optimizer: Optional[HyperparameterOptimizer] = None
        self.insight_generator: Optional[ScientificInsightGenerator] = None

    def run(self, run_type: str, progress_handler: Optional[ProgressHandler] = None, cancel_event: Optional[threading.Event] = None, **kwargs):
        """Main entry point for kicking off an experiment."""
        self.progress_handler = progress_handler
        self.cancel_event = cancel_event or threading.Event()

        challenge_id = kwargs.get("challenge_id")
        if not challenge_id:
            raise ValueError("A 'challenge_id' must be provided.")
        self._get_challenge_data(challenge_id)

        if run_type in ["comparison", "optimization"]:
            if not kwargs.get("patience_level"):
                raise ValueError("A 'patience_level' must be provided.")
            models = kwargs.get("models", [])
            if not models:
                raise ValueError("At least one model must be specified.")

            available_models = self.model_configs.keys()
            for model_name in models:
                if model_name not in available_models:
                    raise ValueError(f"Model '{model_name}' not found. Available: {', '.join(available_models)}")

            return self._run_discovery_session(run_type, **kwargs)

        elif run_type == "interactive":
            # Interactive mode has its own validation inside the method
            return self._run_interactive_session(**kwargs)
        else:
            raise ValueError(f"Invalid run type: {run_type}")

    # Discovery Session Methods
    def _run_discovery_session(self, run_type: str, **kwargs):
        """Configures and runs the scientific discovery engine."""
        challenge_schema = self._get_challenge_data(kwargs.get("challenge_id"))
        models_to_run = [kwargs.get("model_to_optimize")] if run_type == "optimization" else kwargs.get("models")

        self.challenge = self._create_challenge_config(challenge_schema, kwargs.get("smoke_test", False), kwargs.get("dataset"))
        self.algorithms = self._load_algorithms(models_to_run, arch_overrides=kwargs.get("arch_overrides"))
        patience_budget = PatienceBudget(level=kwargs.get("patience_level"))

        # Initialize engine components
        self.timing_manager = ScientificTimingManager()
        self.patience_manager = AdaptivePatienceManager(patience_budget, self.timing_manager)
        self.optimizer = OptunaOptimizer()
        self.insight_generator = ScientificInsightGenerator(self.challenge, self.algorithms, "config/insight_config.yaml")

        self.send_progress('start_session', {'challenge': self.challenge.name})

        baseline_results, baseline_histories = self._run_baseline_evaluation()
        optimization_results = self._run_hyperparameter_optimization(baseline_results)
        final_results, final_histories = self._run_final_evaluation(optimization_results)

        log_histories = {**baseline_histories, **final_histories}
        plot_path = generate_performance_plot(log_histories) if log_histories else None
        report_path = "scientific_report.md"

        insights = self._generate_insights(final_results, plot_path, report_path, log_histories)

        results = DiscoveryResults(
            challenge=self.challenge,
            algorithm_results=final_results,
            insights=insights,
            timing_data=self.timing_manager.export_metrics(),
            metadata={
                'session_start_time': self.patience_manager.start_time,
                'total_elapsed_time': self.patience_manager.get_elapsed_time(),
                'plot_path': plot_path
            },
            log_histories=log_histories
        )
        console.print("[green]✅ Scientific discovery session completed![/green]")
        self._display_results(results)
        return results

    def _run_baseline_evaluation(self) -> Tuple[Dict[str, Any], Dict[str, List]]:
        self.send_progress('start_phase', {'phase': 'baseline_evaluation'})
        results, histories = self._run_evaluation(
            title="⚡ Running Baseline Evaluation", phase=ExplorationPhase.BASELINE_EVALUATION,
            patience_allocation=0.5, run_suffix="baseline", model_config_fn=lambda alg: alg.config,
            result_key_fn=lambda alg: alg.name, timing_value=0
        )
        self.send_progress('end_phase', {'phase': 'baseline_evaluation', 'results': results})
        return results, histories

    def _run_final_evaluation(self, optimization_results: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, List]]:
        self.send_progress('start_phase', {'phase': 'final_evaluation'})
        def model_config_fn(alg):
            config = alg.config.copy()
            config['arch_overrides'] = optimization_results.get(alg.name, {}).get('best_params', {})
            return config
        results, histories = self._run_evaluation(
            title="🏆 Running Final Evaluation", phase=ExplorationPhase.FINAL_EVALUATION,
            patience_allocation=0.7, run_suffix="final", model_config_fn=model_config_fn,
            result_key_fn=lambda alg: f"{alg.name}_optimized", timing_value=1
        )
        self.send_progress('end_phase', {'phase': 'final_evaluation', 'results': results})
        return results, histories

    def _run_evaluation(self, title: str, phase: ExplorationPhase, patience_allocation: float, run_suffix: str, model_config_fn: Callable, result_key_fn: Callable, timing_value: int) -> Tuple[Dict[str, Any], Dict[str, List]]:
        self.send_progress('start_evaluation', {'title': title})
        self.patience_manager.allocate_for_phase(phase, patience_allocation)
        start_time = time.time()
        trainers = self._initialize_trainers(run_suffix, model_config_fn)
        results, histories = self._run_sequential_evaluation(trainers, result_key_fn)
        elapsed_time = time.time() - start_time
        self.patience_manager.update_patience_consumption(elapsed_time, phase)
        self.timing_manager.record_discovery_timing(f"{run_suffix}_evaluation", elapsed_time, timing_value)
        self.send_progress('end_evaluation', {'title': title, 'results': results})
        return results, histories

    def _initialize_trainers(self, run_suffix: str, model_config_fn: Callable) -> List[Trainer]:
        trainers = []
        for alg in self.algorithms:
            run_config = {"study_name": f"{self.challenge.id}_{run_suffix}_{alg.name}", "output_dir": "experiments"}
            model_config = model_config_fn(alg)
            trainer = Trainer(self.default_training_config, model_config, self.challenge.dataset, run_config, progress_handler=self.progress_handler, cancel_event=self.cancel_event)
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
            self.send_progress('start_algorithm', {'algorithm': alg.name, 'progress': (i + 1) / len(self.algorithms)})
            metrics, history = trainer.run_sequential_training()
            results[result_key_fn(alg)] = metrics
            histories[result_key_fn(alg)] = history
            self.send_progress('end_algorithm', {'algorithm': alg.name, 'metrics': metrics})
        return results, histories

    def _run_hyperparameter_optimization(self, baseline_results: Dict[str, Any]) -> Dict[str, Any]:
        self.send_progress('start_phase', {'phase': 'optimization'})
        optimization_results = {}
        for alg in self.algorithms:
            if self.cancel_event.is_set():
                self.send_progress('optimization_cancelled', {'reason': 'Cancelled by user.'})
                break
            if not alg.search_space:
                self.send_progress('skip_optimization', {'algorithm': alg.name, 'reason': 'No search space defined.'})
                optimization_results[alg.name] = baseline_results.get(alg.name, {})
                continue
            self.send_progress('start_optimization_alg', {'algorithm': alg.name})
            best_params = self.optimizer.optimize(objective=lambda hparams: self._optimization_objective(alg, hparams), search_space=alg.search_space, n_trials=10)
            self.send_progress('end_optimization_alg', {'algorithm': alg.name, 'best_params': best_params})
            optimization_results[alg.name] = {'best_params': best_params}
        self.send_progress('end_phase', {'phase': 'optimization', 'results': optimization_results})
        return optimization_results

    def _optimization_objective(self, alg: AlgorithmConfig, hparams: Dict[str, Any]) -> float:
        self.send_progress('start_trial', {'params': hparams})
        model_config = alg.config.copy()
        model_config['arch_overrides'] = hparams
        run_config = {"study_name": f"{self.challenge.id}_optimize_{alg.name}", "output_dir": "experiments"}
        trainer = Trainer(self.default_training_config, model_config, self.challenge.dataset, run_config, progress_handler=self.progress_handler)
        metrics, _ = trainer.train_and_evaluate()
        loss = metrics.get('all/lm_loss', float('inf'))
        self.send_progress('end_trial', {'params': hparams, 'loss': loss})
        return loss

    def _generate_insights(self, final_results: Dict[str, Any], plot_path: Optional[str], report_path: str, log_histories: Optional[Dict[str, List[Dict[str, Any]]]]) -> List[ScientificInsight]:
        self.send_progress('start_phase', {'phase': 'insight_generation'})
        self.patience_manager.allocate_for_phase(ExplorationPhase.INSIGHT_GENERATION, discovery_potential=0.9)
        start_time = time.time()
        insights = self.insight_generator.extract_insights(final_results)
        for insight in insights:
            insight.discovery_potential = self.insight_generator.classify_discovery_potential(insight.discovery_potential)
        elapsed_time = time.time() - start_time
        self.patience_manager.update_patience_consumption(elapsed_time, ExplorationPhase.INSIGHT_GENERATION)
        self.timing_manager.record_discovery_timing("insight_generation", elapsed_time, len(insights))
        if insights:
            self.send_progress('insights_generated', {'insights': insights, 'report_path': report_path})
            report_generator = ScientificReportGenerator(challenge_name=self.challenge.name, algorithm_names=[alg.name for alg in self.algorithms], final_results=final_results, plot_path=plot_path)
            report_content = report_generator.generate_report(insights)
            with open(report_path, "w") as f: f.write(report_content)
        else:
            self.send_progress('no_insights')
        self.send_progress('end_phase', {'phase': 'insight_generation'})
        return insights

    # Interactive Session Methods
    def _run_interactive_session(self, **kwargs) -> Dict[str, Any]:
        """Runs a single puzzle through models for interactive visualization."""
        if not all(k in kwargs for k in ["dataset_path", "puzzle_index", "challenge_id", "models"]):
             raise ValueError("Missing required arguments for interactive mode.")

        dataset_path, puzzle_index = kwargs["dataset_path"], kwargs["puzzle_index"]
        challenge_schema = self._get_challenge_data(kwargs["challenge_id"])
        challenge_config = self._create_challenge_config(challenge_schema, smoke_test=False)
        algorithm_configs = self._load_algorithms(kwargs["models"])

        console.print(f"Running interactive puzzle {puzzle_index} from {dataset_path}")
        puzzle_data, dummy_metadata = self._load_interactive_puzzle_data(dataset_path, puzzle_index)

        all_results = {}
        for algo_config in algorithm_configs:
            console.print(f"[bold cyan]Running model: {algo_config.name}[/bold cyan]")
            trainer = self._initialize_trainer_for_puzzle(algo_config, dummy_metadata, challenge_config.dataset)
            model = trainer.train_state.model
            is_hrem = isinstance(model.model, HREM)
            runner = InteractiveRunner(model, is_hrem=is_hrem)
            batch = {k: v.to('cpu') for k, v in puzzle_data.items()}
            runner.reset(batch)
            step_results = []
            for i in range(puzzle_data['inputs'].shape[1]):
                single_step_batch = {'inputs': batch['inputs'][:, i:i+1], 'labels': batch['labels'][:, i:i+1], 'puzzle_identifiers': batch['puzzle_identifiers']}
                result, metrics = runner.run_step(single_step_batch)
                result["metrics"] = metrics
                step_results.append(result)
            all_results[algo_config.name] = step_results
            console.print(f"Finished running {algo_config.name}. Collected {len(step_results)} steps.")
        return all_results

    def _load_interactive_puzzle_data(self, dataset_path: str, puzzle_index: int) -> Tuple[Dict[str, torch.Tensor], PuzzleDatasetMetadata]:
        """Loads a single puzzle and creates dummy metadata for it."""
        dataset_config = PuzzleDatasetConfig(seed=42, dataset_path=dataset_path, global_batch_size=1, test_set_mode=True, epochs_per_iter=1, rank=0, num_replicas=1)
        dataset = PuzzleDataset(config=dataset_config, split="test")
        dataset._lazy_load_dataset()
        set_data = dataset._data["all"]
        puzzle_start = set_data["puzzle_indices"][puzzle_index]
        puzzle_end = set_data["puzzle_indices"][puzzle_index + 1]
        puzzle_data = {
            "inputs": torch.from_numpy(set_data["inputs"][puzzle_start:puzzle_end]),
            "labels": torch.from_numpy(set_data["labels"][puzzle_start:puzzle_end]),
            "puzzle_identifiers": torch.from_numpy(np.array([set_data["puzzle_identifiers"][puzzle_index]]))
        }
        dummy_metadata = PuzzleDatasetMetadata(pad_id=0, ignore_label_id=-1, blank_identifier_id=0, vocab_size=32, seq_len=puzzle_data['inputs'].shape[1], num_puzzle_identifiers=1, total_groups=1, mean_puzzle_examples=1.0, sets=['all'])
        return puzzle_data, dummy_metadata

    def _initialize_trainer_for_puzzle(self, algo_config: AlgorithmConfig, metadata: PuzzleDatasetMetadata, challenge_dataset_config: Dict[str, Any]) -> Trainer:
        """Initializes a trainer instance for a given algorithm and puzzle metadata."""
        run_config = {"study_name": f"interactive_{algo_config.name}", "output_dir": "experiments"}
        trainer = Trainer(self.default_training_config, algo_config.config, challenge_dataset_config, run_config)
        trainer.train_metadata = metadata
        trainer.build_model()
        return trainer

    # Helper methods
    def _get_challenge_data(self, challenge_id: str) -> ChallengeSchema:
        """Fetches and validates challenge data from the registry."""
        challenge_data = self.challenge_registry.get_challenge_by_id(challenge_id)
        if not challenge_data:
            raise ValueError(f"Challenge with ID '{challenge_id}' not found")
        return challenge_data

    def _create_challenge_config(self, challenge_schema: ChallengeSchema, smoke_test: bool, dataset_override: Optional[str] = None) -> ChallengeConfig:
        """Creates the final ChallengeConfig object from the schema."""
        dataset_name = dataset_override or challenge_schema.dataset.dataset
        dataset_path = dataset_manager.get_dataset_path(dataset_name, smoke_test=smoke_test)
        data_config = {"dataset": dataset_path, "smoke_test": smoke_test}
        return ChallengeConfig(
            name=challenge_schema.name, id=challenge_schema.id, description=challenge_schema.description,
            dataset=data_config, scientific_question=challenge_schema.scientific_question or "",
            hypothesis_space=challenge_schema.hypothesis_space or []
        )

    def _load_algorithms(self, algorithm_names: List[str], arch_overrides: Optional[str] = None) -> List[AlgorithmConfig]:
        """Load algorithm configurations, applying overrides if provided."""
        overrides = json.loads(arch_overrides) if arch_overrides and isinstance(arch_overrides, str) else (arch_overrides or {})
        algorithms = []
        for name in algorithm_names:
            model_schema = self.model_configs.get(name)
            if not model_schema:
                console.print(f"[yellow]⚠️  Model '{name}' not found in registry[/yellow]")
                continue
            model_dict = model_schema.model_dump()
            model_dict.update(overrides)
            algorithms.append(AlgorithmConfig(
                name=model_schema.name, algorithm_class=model_schema.algorithm_class,
                search_space=self.search_spaces.get(name, {}),
                theoretical_advantages=model_schema.theoretical_advantages,
                theoretical_limitations=model_schema.theoretical_limitations,
                config=model_dict
            ))
        return algorithms

    def _display_results(self, results: DiscoveryResults):
        """Display the results of the discovery session."""
        console.print("\n[bold green]=== SCIENTIFIC DISCOVERY RESULTS ===[/bold green]")
        console.print(f"\n[bold]Challenge:[/bold] {results.challenge.name}")
        if results.insights:
            console.print(f"\n[bold]Scientific Insights:[/bold]")
            for i, insight in enumerate(results.insights, 1):
                console.print(f"  {i}. [{insight.type.upper()}] (Confidence: {insight.confidence:.2f}) - {insight.description}")
                for implication in insight.implications:
                    console.print(f"      - {implication}")
        else:
            console.print(f"\n[yellow]No significant scientific insights generated[/yellow]")
        console.print(f"\n[bold]Session Summary:[/bold]")
        console.print(f"  Total Time: {results.metadata.get('total_elapsed_time', 0):.2f}s")
        if (plot_path := results.metadata.get('plot_path')):
            console.print(f"\n[bold]Performance Plot:[/bold] 📊")
            console.print(f"  A plot of the training performance has been saved to: [u]{plot_path}[/u]")

    def send_progress(self, event_type: str, data: Dict = None):
        if self.progress_handler:
            self.progress_handler.on_progress(event_type, data if data is not None else {})
