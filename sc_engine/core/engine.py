"""Scientific Discovery Engine for algorithm comparison."""

import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from rich.console import Console
from .config import ChallengeConfig, AlgorithmConfig, PatienceBudget, DiscoveryConfig
from .patience_manager import AdaptivePatienceManager, ExplorationPhase, ScientificInsight
from .scheduler import DiscoveryAwareScheduler
from .insight_generator import ScientificInsightGenerator
from .report_generator import ScientificReportGenerator
from .timing_manager import ScientificTimingManager
from hrm_system.config import TrainingConfig

console = Console()

@dataclass
class DiscoveryResults:
    """Results from a scientific discovery session."""
    challenge: ChallengeConfig
    algorithm_results: Dict[str, Any]
    insights: List[ScientificInsight]
    timing_data: Dict[str, Any]
    metadata: Dict[str, Any]

from hrm_system.config import RunConfig, ModelConfig
from typing import Callable

class ScientificDiscoveryEngine:
    """Central orchestrator for scientific exploration process."""
    
    def __init__(self, challenge: ChallengeConfig, algorithms: List[AlgorithmConfig], model_runner: Callable, optimization_runner: Callable, config: Dict[str, Any] = None):
        """Initialize with a scientific challenge and algorithms to compare."""
        self.challenge = challenge
        self.algorithms = algorithms
        self.model_runner = model_runner
        self.optimization_runner = optimization_runner
        self.config = config or {}
        self.patience_manager: Optional[AdaptivePatienceManager] = None
        self.scheduler = DiscoveryAwareScheduler(algorithms, challenge)
        self.insight_generator = ScientificInsightGenerator(challenge, algorithms, self.config.get("insights", "config/insight_config.yaml"))
        self.timing_manager = ScientificTimingManager()
        self.results: Dict[str, Any] = {}
        
    def execute_discovery_session(self, patience_budget: PatienceBudget) -> DiscoveryResults:
        """
        Execute a scientifically-driven comparison within patience constraints.
        """
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
        console.print("[bold blue]⚡ Running Baseline Evaluation...[/bold blue]")
        allocation = self.patience_manager.allocate_for_phase(ExplorationPhase.BASELINE_EVALUATION, 0.5)
        start_time = time.time()
        baseline_results = {}
        
        for alg in self.algorithms:
            run_config = RunConfig(study_name=f"{self.challenge.id}_baseline")
            training_config = TrainingConfig()
            model_config = ModelConfig(name=alg.name, algorithm_class=alg.algorithm_class)

            metrics = self.model_runner(model_config, self.challenge.dataset, run_config, training_config)
            baseline_results[alg.name] = metrics
        
        elapsed_time = time.time() - start_time
        self.patience_manager.update_patience_consumption(elapsed_time, ExplorationPhase.BASELINE_EVALUATION)
        self.timing_manager.record_discovery_timing("baseline_evaluation", elapsed_time, 0)
        return baseline_results

    def _run_hyperparameter_optimization(self, baseline_results: Dict[str, Any]) -> Dict[str, Any]:
        """Run hyperparameter optimization with adaptive depth."""
        console.print("[bold blue]🔍 Running Hyperparameter Optimization...[/bold blue]")
        allocation = self.patience_manager.allocate_for_phase(ExplorationPhase.HYPERPARAMETER_OPTIMIZATION, 0.8)
        start_time = time.time()
        optimization_results = {}
        
        for alg in self.algorithms:
            model_config = ModelConfig(name=alg.name, algorithm_class=alg.algorithm_class)
            best_params = self.optimization_runner(model_config, self.challenge.dataset, alg.search_space)
            optimization_results[alg.name] = {'best_params': best_params}
        
        elapsed_time = time.time() - start_time
        self.patience_manager.update_patience_consumption(elapsed_time, ExplorationPhase.HYPERPARAMETER_OPTIMIZATION)
        self.timing_manager.record_discovery_timing("hyperparameter_optimization", elapsed_time, 2)
        return optimization_results

    def _run_final_evaluation(self, optimization_results: Dict[str, Any]) -> Dict[str, Any]:
        """Run final evaluation with optimized parameters."""
        console.print("[bold blue]🏆 Running Final Evaluation...[/bold blue]")
        allocation = self.patience_manager.allocate_for_phase(ExplorationPhase.FINAL_EVALUATION, 0.7)
        start_time = time.time()
        final_results = {}
        
        for alg in self.algorithms:
            run_config = RunConfig(study_name=f"{self.challenge.id}_final")
            training_config = TrainingConfig()

            # Optimized run
            model_config = ModelConfig(name=alg.name, algorithm_class=alg.algorithm_class, arch_overrides=optimization_results[alg.name]['best_params'])
            metrics = self.model_runner(model_config, self.challenge.dataset, run_config, training_config)
            final_results[f"{alg.name}_optimized"] = metrics

        elapsed_time = time.time() - start_time
        self.patience_manager.update_patience_consumption(elapsed_time, ExplorationPhase.FINAL_EVALUATION)
        self.timing_manager.record_discovery_timing("final_evaluation", elapsed_time, 1)
        return final_results
    
    def _generate_insights(self, final_results: Dict[str, Any]) -> List[ScientificInsight]:
        """Generate scientific insights from comparison results."""
        console.print("[bold blue]💡 Generating Scientific Insights...[/bold blue]")
        
        # Allocate time for insight generation
        allocation = self.patience_manager.allocate_for_phase(
            ExplorationPhase.INSIGHT_GENERATION,
            discovery_potential=0.9  # High potential
        )
        
        start_time = time.time()
        
        # Generate insights using the insight generator
        insights = self.insight_generator.extract_insights(final_results)
        
        # Classify discovery potential for each insight
        for insight in insights:
            insight.discovery_potential = self.insight_generator.classify_discovery_potential(insight.discovery_potential)
        
        elapsed_time = time.time() - start_time
        self.patience_manager.update_patience_consumption(elapsed_time, ExplorationPhase.INSIGHT_GENERATION)
        self.timing_manager.record_discovery_timing("insight_generation", elapsed_time, len(insights))
        
        # Generate and save the report
        if insights:
            console.print(f"[green]✅ Generated {len(insights)} scientific insights.[/green]")
            report_generator = ScientificReportGenerator(
                challenge_name=self.challenge.name,
                algorithm_names=[alg.name for alg in self.algorithms]
            )
            report_content = report_generator.generate_report(insights)

            # Save the report
            report_path = "scientific_report.md"
            with open(report_path, "w") as f:
                f.write(report_content)
            console.print(f"[bold green]📄 Scientific report saved to {report_path}[/bold green]")
        else:
            console.print("[yellow]⚠️ No significant insights generated[/yellow]")
        
        return insights