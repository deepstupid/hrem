import time
from typing import Dict, Any, List, Optional, Tuple
import json
import yaml
import threading
from rich.console import Console

from .config import ChallengeConfig, AlgorithmConfig, PatienceBudget
from .config_manager import ConfigManager
from .patience_manager import AdaptivePatienceManager, ExplorationPhase, ScientificInsight
from sc_engine.utils.plotting import generate_performance_plot
from .challenge_registry import ChallengeRegistry
from .schemas import ChallengeSchema
from .progress_handler import ProgressHandler
from .evaluation import ModelEvaluator
from .components import OptunaOptimizer, ScientificInsightGenerator, ScientificReportGenerator, ScientificTimingManager
from .experiment import DiscoveryResults
from dataset_manager import dataset_manager
from .trainer import Trainer

console = Console()

class ScientificDiscoveryEngine:
    """Orchestrates the scientific discovery process."""

    def __init__(self,
                 config_manager: ConfigManager,
                 progress_handler: Optional[ProgressHandler] = None,
                 cancel_event: Optional[threading.Event] = None,
                 optimizer: Optional[OptunaOptimizer] = None,
                 timing_manager: Optional[ScientificTimingManager] = None):
        self.config_manager = config_manager
        self.challenge_registry = ChallengeRegistry(self.config_manager)
        self.model_configs = self.config_manager.load_model_configs()
        self.search_spaces = self.config_manager.load_search_spaces()
        with open("config/training/default.yaml", 'r') as f:
            self.default_training_config = yaml.safe_load(f)

        self.progress_handler = progress_handler
        self.cancel_event = cancel_event or threading.Event()
        self.optimizer = optimizer or OptunaOptimizer()
        self.timing_manager = timing_manager or ScientificTimingManager()

    def run(self, **kwargs):
        """Main entry point for kicking off an experiment."""
        challenge_id = kwargs.get("challenge_id")
        if not challenge_id:
            raise ValueError("A 'challenge_id' must be provided.")

        challenge_schema = self._get_challenge_data(challenge_id)

        if not kwargs.get("patience_level"):
            raise ValueError("A 'patience_level' must be provided.")
        models = kwargs.get("models", [])
        if not models:
            raise ValueError("At least one model must be specified.")

        available_models = self.model_configs.keys()
        for model_name in models:
            if model_name not in available_models:
                raise ValueError(f"Model '{model_name}' not found. Available: {', '.join(available_models)}")

        return self._run_discovery_session(**kwargs)

    def _run_discovery_session(self, **kwargs):
        """Configures and runs the scientific discovery engine."""
        challenge_schema = self._get_challenge_data(kwargs.get("challenge_id"))
        models_to_run = kwargs.get("models")

        challenge = self._create_challenge_config(challenge_schema, kwargs.get("smoke_test", False), kwargs.get("dataset"))
        algorithms = self._load_algorithms(models_to_run, arch_overrides=kwargs.get("arch_overrides"))
        patience_budget = PatienceBudget(level=kwargs.get("patience_level"))

        # Initialize engine components
        self.patience_manager = AdaptivePatienceManager(patience_budget, self.timing_manager)
        self.insight_generator = ScientificInsightGenerator(challenge, algorithms, "config/insight_config.yaml")

        self.send_progress('start_session', {'challenge': challenge.name})

        baseline_results, baseline_histories = self._run_baseline_evaluation(challenge, algorithms)
        optimization_results = self._run_hyperparameter_optimization(algorithms, challenge, baseline_results)
        final_results, final_histories = self._run_final_evaluation(challenge, algorithms, optimization_results)

        log_histories = {**baseline_histories, **final_histories}
        plot_path = generate_performance_plot(log_histories) if log_histories else None
        report_path = "scientific_report.md"

        insights = self._generate_insights(final_results, plot_path, report_path, log_histories, algorithms, challenge)

        results = DiscoveryResults(
            challenge=challenge,
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

    def _run_baseline_evaluation(self, challenge, algorithms) -> Tuple[Dict[str, Any], Dict[str, List]]:
        self.send_progress('start_phase', {'phase': 'baseline_evaluation'})
        self.patience_manager.allocate_for_phase(ExplorationPhase.BASELINE_EVALUATION, 0.5)
        start_time = time.time()

        model_evaluator = ModelEvaluator(self.default_training_config, self.progress_handler, self.cancel_event, algorithms, challenge.id, challenge.dataset)
        results, histories = model_evaluator.run_evaluation(
            title="⚡ Running Baseline Evaluation", phase=ExplorationPhase.BASELINE_EVALUATION,
            run_suffix="baseline", model_config_fn=lambda alg: alg.config,
            result_key_fn=lambda alg: alg.name
        )
        elapsed_time = time.time() - start_time
        self.patience_manager.update_patience_consumption(elapsed_time, ExplorationPhase.BASELINE_EVALUATION)
        self.timing_manager.record_discovery_timing("baseline_evaluation", elapsed_time, 0)
        self.send_progress('end_phase', {'phase': 'baseline_evaluation', 'results': results})
        return results, histories

    def _run_final_evaluation(self, challenge, algorithms, optimization_results: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, List]]:
        self.send_progress('start_phase', {'phase': 'final_evaluation'})
        self.patience_manager.allocate_for_phase(ExplorationPhase.FINAL_EVALUATION, 0.7)
        start_time = time.time()
        def model_config_fn(alg):
            config = alg.config.copy()
            config['arch_overrides'] = optimization_results.get(alg.name, {}).get('best_params', {})
            return config

        model_evaluator = ModelEvaluator(self.default_training_config, self.progress_handler, self.cancel_event, algorithms, challenge.id, challenge.dataset)
        results, histories = model_evaluator.run_evaluation(
            title="🏆 Running Final Evaluation", phase=ExplorationPhase.FINAL_EVALUATION,
            run_suffix="final", model_config_fn=model_config_fn,
            result_key_fn=lambda alg: f"{alg.name}_optimized"
        )
        elapsed_time = time.time() - start_time
        self.patience_manager.update_patience_consumption(elapsed_time, ExplorationPhase.FINAL_EVALUATION)
        self.timing_manager.record_discovery_timing("final_evaluation", elapsed_time, 1)
        self.send_progress('end_phase', {'phase': 'final_evaluation', 'results': results})
        return results, histories

    def _run_hyperparameter_optimization(self, algorithms, challenge, baseline_results: Dict[str, Any]) -> Dict[str, Any]:
        self.send_progress('start_phase', {'phase': 'optimization'})
        optimization_results = {}
        for alg in algorithms:
            if self.cancel_event.is_set():
                self.send_progress('optimization_cancelled', {'reason': 'Cancelled by user.'})
                break
            if not alg.search_space:
                self.send_progress('skip_optimization', {'algorithm': alg.name, 'reason': 'No search space defined.'})
                optimization_results[alg.name] = baseline_results.get(alg.name, {})
                continue
            self.send_progress('start_optimization_alg', {'algorithm': alg.name})
            best_params = self.optimizer.optimize(objective=lambda hparams: self._optimization_objective(alg, hparams, challenge), search_space=alg.search_space, n_trials=10)
            self.send_progress('end_optimization_alg', {'algorithm': alg.name, 'best_params': best_params})
            optimization_results[alg.name] = {'best_params': best_params}
        self.send_progress('end_phase', {'phase': 'optimization', 'results': optimization_results})
        return optimization_results

    def _optimization_objective(self, alg: AlgorithmConfig, hparams: Dict[str, Any], challenge) -> float:
        self.send_progress('start_trial', {'params': hparams})
        model_config = alg.config.copy()
        model_config['arch_overrides'] = hparams
        run_config = {"study_name": f"{challenge.id}_optimize_{alg.name}", "output_dir": "experiments"}
        trainer = Trainer(self.default_training_config, model_config, challenge.dataset, run_config, progress_handler=self.progress_handler)
        metrics, _ = trainer.train_and_evaluate()
        loss = metrics.get('all/lm_loss', float('inf'))
        self.send_progress('end_trial', {'params': hparams, 'loss': loss})
        return loss

    def _generate_insights(self, final_results: Dict[str, Any], plot_path: Optional[str], report_path: str, log_histories: Optional[Dict[str, List[Dict[str, Any]]]], algorithms, challenge) -> List[ScientificInsight]:
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
            report_generator = ScientificReportGenerator(challenge_name=challenge.name, algorithm_names=[alg.name for alg in algorithms], final_results=final_results, plot_path=plot_path)
            report_content = report_generator.generate_report(insights)
            with open(report_path, "w") as f: f.write(report_content)
        else:
            self.send_progress('no_insights')
        self.send_progress('end_phase', {'phase': 'insight_generation'})
        return insights

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
