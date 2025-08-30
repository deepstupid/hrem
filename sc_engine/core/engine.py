"""Scientific Discovery Engine for algorithm comparison."""

import time
import os
import torch
import torch.distributed as dist
import tqdm
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from rich.console import Console

from .config import ChallengeConfig, AlgorithmConfig, PatienceBudget
from .patience_manager import AdaptivePatienceManager, ExplorationPhase, ScientificInsight
from .scheduler import DiscoveryAwareScheduler
from .insight_generator import ScientificInsightGenerator
from .report_generator import ScientificReportGenerator
from .timing_manager import ScientificTimingManager
from .optimization import HyperparameterOptimizer, OptunaOptimizer
from .utils import (
    LocalLogger,
    create_dataloader,
)
from .trainer import Trainer
from puzzle_dataset import PuzzleDatasetMetadata
import importlib
import yaml

console = Console()

@dataclass
class DiscoveryResults:
    """Results from a scientific discovery session."""
    challenge: ChallengeConfig
    algorithm_results: Dict[str, Any]
    insights: List[ScientificInsight]
    timing_data: Dict[str, Any]
    metadata: Dict[str, Any]

class ScientificDiscoveryEngine:
    """Central orchestrator for scientific exploration process."""
    
    def __init__(self, challenge: ChallengeConfig, algorithms: List[AlgorithmConfig], config: Dict[str, Any] = None):
        """Initialize with a scientific challenge and algorithms to compare."""
        self.challenge = challenge
        self.algorithms = algorithms
        self.config = config or {}
        self.patience_manager: Optional[AdaptivePatienceManager] = None
        self.scheduler = DiscoveryAwareScheduler(algorithms, challenge)
        self.insight_generator = ScientificInsightGenerator(challenge, algorithms, self.config.get("insights", "config/insight_config.yaml"))
        self.timing_manager = ScientificTimingManager()
        self.results: Dict[str, Any] = {}
        self.optimizer: HyperparameterOptimizer = OptunaOptimizer()
        
        with open("config/training/default.yaml", 'r') as f:
            self.default_training_config = yaml.safe_load(f)

    def execute_discovery_session(self, patience_budget: PatienceBudget) -> DiscoveryResults:
        """Execute a scientifically-driven comparison within patience constraints."""
        console.print(f"[bold blue]🔬 Starting Scientific Discovery Session[/bold blue]")
        console.print(f"[cyan]Challenge: {self.challenge.name}[/cyan]")
        
        self.patience_manager = AdaptivePatienceManager(patience_budget, self.timing_manager)
        
        baseline_results = self._run_baseline_evaluation()
        optimization_results = self._run_hyperparameter_optimization(baseline_results)
        final_results = self._run_final_evaluation(optimization_results)
        insights = self._generate_insights(final_results)
        
        discovery_results = DiscoveryResults(
            challenge=self.challenge,
            algorithm_results=final_results,
            insights=insights,
            timing_data=self.timing_manager.export_metrics(),
            metadata={
                'session_start_time': self.patience_manager.start_time,
                'total_elapsed_time': self.patience_manager.get_elapsed_time(),
            }
        )
        
        console.print("[green]✅ Scientific discovery session completed![/green]")
        return discovery_results

    def _run_baseline_evaluation(self) -> Dict[str, Any]:
        """Run baseline evaluation for all algorithms."""
        return self._run_evaluation(
            title="⚡ Running Baseline Evaluation",
            phase=ExplorationPhase.BASELINE_EVALUATION,
            patience_allocation=0.5,
            run_suffix="baseline",
            model_config_fn=lambda alg: alg.config,
            result_key_fn=lambda alg: alg.name,
            timing_value=0
        )

    def _run_evaluation(self,
                        title: str,
                        phase: ExplorationPhase,
                        patience_allocation: float,
                        run_suffix: str,
                        model_config_fn: Callable[[AlgorithmConfig], Dict[str, Any]],
                        result_key_fn: Callable[[AlgorithmConfig], str],
                        timing_value: int
                        ) -> Dict[str, Any]:
        """Generic method to run an evaluation phase."""
        console.print(f"[bold blue]{title}...[/bold blue]")
        self.patience_manager.allocate_for_phase(phase, patience_allocation)
        start_time = time.time()
        results = {}
        smoke_test = self.config.get("smoke_test", False)

        for alg in self.algorithms:
            run_config = {
                "study_name": f"{self.challenge.id}_{run_suffix}_{alg.name}",
                "output_dir": "experiments",
                "smoke_test": smoke_test
            }
            training_config = self.default_training_config
            model_config = model_config_fn(alg)

            trainer = Trainer(training_config, model_config, self.challenge.dataset, run_config)
            metrics = trainer.train_and_evaluate()
            results[result_key_fn(alg)] = metrics

        elapsed_time = time.time() - start_time
        self.patience_manager.update_patience_consumption(elapsed_time, phase)
        self.timing_manager.record_discovery_timing(f"{run_suffix}_evaluation", elapsed_time, timing_value)
        return results

    def _run_hyperparameter_optimization(self, baseline_results: Dict[str, Any]) -> Dict[str, Any]:
        """Run hyperparameter optimization with adaptive depth."""
        console.print("[bold blue]🔍 Running Hyperparameter Optimization...[/bold blue]")
        
        optimization_results = {}

        for alg in self.algorithms:
            if not alg.search_space:
                console.print(f"[yellow]No search space defined for {alg.name}, skipping optimization.[/yellow]")
                optimization_results[alg.name] = baseline_results.get(alg.name, {})
                continue

            def objective(hparams: Dict[str, Any]) -> float:
                model_config = alg.config.copy()
                model_config['arch_overrides'] = hparams
                run_config = {"study_name": f"{self.challenge.id}_optimize_{alg.name}", "output_dir": "experiments"}
                training_config = self.default_training_config

                trainer = Trainer(training_config, model_config, self.challenge.dataset, run_config)
                metrics = trainer.train_and_evaluate()
                return metrics.get('all/lm_loss', float('inf'))

            best_params = self.optimizer.optimize(
                objective=objective,
                search_space=alg.search_space,
                n_trials=self.config.get("n_trials", 10)
            )

            console.print(f"[green]Best parameters for {alg.name}: {best_params}[/green]")
            optimization_results[alg.name] = {'best_params': best_params}

        return optimization_results

    def _run_final_evaluation(self, optimization_results: Dict[str, Any]) -> Dict[str, Any]:
        """Run final evaluation with optimized parameters."""
        def model_config_fn(alg):
            config = alg.config.copy()
            config['arch_overrides'] = optimization_results.get(alg.name, {}).get('best_params', {})
            return config

        return self._run_evaluation(
            title="🏆 Running Final Evaluation",
            phase=ExplorationPhase.FINAL_EVALUATION,
            patience_allocation=0.7,
            run_suffix="final",
            model_config_fn=model_config_fn,
            result_key_fn=lambda alg: f"{alg.name}_optimized",
            timing_value=1
        )
    
    def _generate_insights(self, final_results: Dict[str, Any]) -> List[ScientificInsight]:
        """Generate scientific insights from comparison results."""
        console.print("[bold blue]💡 Generating Scientific Insights...[/bold blue]")
        
        allocation = self.patience_manager.allocate_for_phase(ExplorationPhase.INSIGHT_GENERATION, discovery_potential=0.9)
        start_time = time.time()
        
        insights = self.insight_generator.extract_insights(final_results)
        
        for insight in insights:
            insight.discovery_potential = self.insight_generator.classify_discovery_potential(insight.discovery_potential)
        
        elapsed_time = time.time() - start_time
        self.patience_manager.update_patience_consumption(elapsed_time, ExplorationPhase.INSIGHT_GENERATION)
        self.timing_manager.record_discovery_timing("insight_generation", elapsed_time, len(insights))
        
        if insights:
            console.print(f"[green]✅ Generated {len(insights)} scientific insights.[/green]")
            report_generator = ScientificReportGenerator(challenge_name=self.challenge.name, algorithm_names=[alg.name for alg in self.algorithms])
            report_content = report_generator.generate_report(insights)

            report_path = "scientific_report.md"
            with open(report_path, "w") as f:
                f.write(report_content)
            console.print(f"[bold green]📄 Scientific report saved to {report_path}[/bold green]")
        else:
            console.print("[yellow]⚠️ No significant insights generated[/yellow]")
        
        return insights