import time
from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass
from rich.console import Console
import yaml

from .config import ChallengeConfig, AlgorithmConfig, PatienceBudget
from .patience_manager import AdaptivePatienceManager, ExplorationPhase, ScientificInsight
from .scheduler import DiscoveryAwareScheduler
from .insight_generator import ScientificInsightGenerator
from .report_generator import ScientificReportGenerator
from .timing_manager import ScientificTimingManager
from .optimization import HyperparameterOptimizer, OptunaOptimizer
from .trainer import Trainer
from .progress_handler import ProgressHandler
from sc_engine.utils.plotting import generate_performance_plot

console = Console()

@dataclass
class DiscoveryResults:
    """Results from a scientific discovery session."""
    challenge: ChallengeConfig
    algorithm_results: Dict[str, Any]
    insights: List[ScientificInsight]
    timing_data: Dict[str, Any]
    metadata: Dict[str, Any]
    log_histories: Dict[str, List[Dict[str, Any]]] = None

class ScientificDiscoveryEngine:
    """Central orchestrator for scientific exploration process."""
    
    def __init__(self, challenge: ChallengeConfig, algorithms: List[AlgorithmConfig], config: Dict[str, Any] = None, progress_handler: Optional[ProgressHandler] = None):
        self.challenge = challenge
        self.algorithms = algorithms
        self.config = config or {}
        self.patience_manager: Optional[AdaptivePatienceManager] = None
        self.scheduler = DiscoveryAwareScheduler(algorithms, challenge)
        self.insight_generator = ScientificInsightGenerator(challenge, algorithms, self.config.get("insights", "config/insight_config.yaml"))
        self.timing_manager = ScientificTimingManager()
        self.results: Dict[str, Any] = {}
        self.optimizer: HyperparameterOptimizer = OptunaOptimizer()
        self.progress_handler = progress_handler
        
        with open("config/training/default.yaml", 'r') as f:
            self.default_training_config = yaml.safe_load(f)

    def execute_discovery_session(self, patience_budget: PatienceBudget) -> DiscoveryResults:
        """Execute a scientifically-driven comparison within patience constraints."""
        self._send_progress('start_session', {'challenge': self.challenge.name})
        
        self.patience_manager = AdaptivePatienceManager(patience_budget, self.timing_manager)
        
        baseline_results, baseline_histories = self._run_baseline_evaluation()
        optimization_results = self._run_hyperparameter_optimization(baseline_results)
        final_results, final_histories = self._run_final_evaluation(optimization_results)

        log_histories = {**baseline_histories, **final_histories}
        plot_path = generate_performance_plot(log_histories) if log_histories else None

        insights = self._generate_insights(final_results, plot_path)
        
        discovery_results = DiscoveryResults(
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
        return discovery_results

    def _run_baseline_evaluation(self) -> Tuple[Dict[str, Any], Dict[str, List]]:
        self._send_progress('start_phase', {'phase': 'baseline_evaluation'})
        results, histories = self._run_evaluation(
            title="⚡ Running Baseline Evaluation",
            phase=ExplorationPhase.BASELINE_EVALUATION,
            patience_allocation=0.5,
            run_suffix="baseline",
            model_config_fn=lambda alg: alg.config,
            result_key_fn=lambda alg: alg.name,
            timing_value=0
        )
        self._send_progress('end_phase', {'phase': 'baseline_evaluation', 'results': results})
        return results, histories

    def _run_evaluation(self,
                        title: str,
                        phase: ExplorationPhase,
                        patience_allocation: float,
                        run_suffix: str,
                        model_config_fn: Callable[[AlgorithmConfig], Dict[str, Any]],
                        result_key_fn: Callable[[AlgorithmConfig], str],
                        timing_value: int
                        ) -> Tuple[Dict[str, Any], Dict[str, List]]:
        self._send_progress('start_evaluation', {'title': title})
        self.patience_manager.allocate_for_phase(phase, patience_allocation)
        start_time = time.time()
        results = {}
        histories = {}
        smoke_test = self.config.get("smoke_test", False)

        for i, alg in enumerate(self.algorithms):
            self._send_progress('start_algorithm', {'algorithm': alg.name, 'progress': (i + 1) / len(self.algorithms)})
            run_config = {
                "study_name": f"{self.challenge.id}_{run_suffix}_{alg.name}",
                "output_dir": "experiments",
                "smoke_test": smoke_test
            }
            training_config = self.default_training_config
            model_config = model_config_fn(alg)

            trainer = Trainer(training_config, model_config, self.challenge.dataset, run_config, progress_handler=self.progress_handler)
            metrics, history = trainer.train_and_evaluate()
            results[result_key_fn(alg)] = metrics
            histories[result_key_fn(alg)] = history
            self._send_progress('end_algorithm', {'algorithm': alg.name, 'metrics': metrics})

        elapsed_time = time.time() - start_time
        self.patience_manager.update_patience_consumption(elapsed_time, phase)
        self.timing_manager.record_discovery_timing(f"{run_suffix}_evaluation", elapsed_time, timing_value)
        self._send_progress('end_evaluation', {'title': title, 'results': results})
        return results, histories

    def _run_hyperparameter_optimization(self, baseline_results: Dict[str, Any]) -> Dict[str, Any]:
        self._send_progress('start_phase', {'phase': 'optimization'})
        optimization_results = {}
        for alg in self.algorithms:
            if not alg.search_space:
                self._send_progress('skip_optimization', {'algorithm': alg.name, 'reason': 'No search space defined.'})
                optimization_results[alg.name] = baseline_results.get(alg.name, {})
                continue
            self._send_progress('start_optimization_alg', {'algorithm': alg.name})
            def objective(hparams: Dict[str, Any]) -> float:
                self._send_progress('start_trial', {'params': hparams})
                model_config = alg.config.copy()
                model_config['arch_overrides'] = hparams
                run_config = {"study_name": f"{self.challenge.id}_optimize_{alg.name}", "output_dir": "experiments"}
                training_config = self.default_training_config
                trainer = Trainer(training_config, model_config, self.challenge.dataset, run_config, progress_handler=self.progress_handler)
                metrics, _ = trainer.train_and_evaluate()
                loss = metrics.get('all/lm_loss', float('inf'))
                self._send_progress('end_trial', {'params': hparams, 'loss': loss})
                return loss
            best_params = self.optimizer.optimize(objective=objective, search_space=alg.search_space, n_trials=self.config.get("n_trials", 10))
            self._send_progress('end_optimization_alg', {'algorithm': alg.name, 'best_params': best_params})
            optimization_results[alg.name] = {'best_params': best_params}
        self._send_progress('end_phase', {'phase': 'optimization', 'results': optimization_results})
        return optimization_results

    def _run_final_evaluation(self, optimization_results: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, List]]:
        self._send_progress('start_phase', {'phase': 'final_evaluation'})
        def model_config_fn(alg):
            config = alg.config.copy()
            config['arch_overrides'] = optimization_results.get(alg.name, {}).get('best_params', {})
            return config
        results, histories = self._run_evaluation(
            title="🏆 Running Final Evaluation",
            phase=ExplorationPhase.FINAL_EVALUATION,
            patience_allocation=0.7,
            run_suffix="final",
            model_config_fn=model_config_fn,
            result_key_fn=lambda alg: f"{alg.name}_optimized",
            timing_value=1
        )
        self._send_progress('end_phase', {'phase': 'final_evaluation', 'results': results})
        return results, histories
    
    def _generate_insights(self, final_results: Dict[str, Any], plot_path: Optional[str]) -> List[ScientificInsight]:
        self._send_progress('start_phase', {'phase': 'insight_generation'})
        allocation = self.patience_manager.allocate_for_phase(ExplorationPhase.INSIGHT_GENERATION, discovery_potential=0.9)
        start_time = time.time()
        insights = self.insight_generator.extract_insights(final_results)
        for insight in insights:
            insight.discovery_potential = self.insight_generator.classify_discovery_potential(insight.discovery_potential)
        elapsed_time = time.time() - start_time
        self.patience_manager.update_patience_consumption(elapsed_time, ExplorationPhase.INSIGHT_GENERATION)
        self.timing_manager.record_discovery_timing("insight_generation", elapsed_time, len(insights))
        if insights:
            self._send_progress('insights_generated', {'insights': insights})
            report_generator = ScientificReportGenerator(
                challenge_name=self.challenge.name,
                algorithm_names=[alg.name for alg in self.algorithms],
                final_results=final_results,
                plot_path=plot_path
            )
            report_content = report_generator.generate_report(insights)
            report_path = "scientific_report.md"
            with open(report_path, "w") as f:
                f.write(report_content)
        else:
            self._send_progress('no_insights')
        self._send_progress('end_phase', {'phase': 'insight_generation'})
        return insights

    def _send_progress(self, event_type: str, data: Dict = None):
        if self.progress_handler:
            self.progress_handler.on_progress(event_type, data if data is not None else {})