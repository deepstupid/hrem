"""Scientific Discovery Engine for algorithm comparison."""

import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from rich.console import Console
from .config import ChallengeConfig, AlgorithmConfig, PatienceBudget, DiscoveryConfig
from .patience_manager import AdaptivePatienceManager, ExplorationPhase, ScientificInsight
from .scheduler import DiscoveryAwareScheduler
from .insight_generator import ScientificInsightGenerator
from .timing_manager import ScientificTimingManager
from demo_model_runner import run_model_with_fallback
from hrm_system.config import ExperimentConfig, OptimizationConfig, RunConfig, TrainingConfig
from demo_models import get_model_config, get_model_search_space
from hrm_system import logger_callback
from hrm_system.optimization import run_optimization


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
    
    def __init__(self, challenge: ChallengeConfig, algorithms: List[AlgorithmConfig], smoke_test: bool = False):
        """Initialize with a scientific challenge and algorithms to compare."""
        self.challenge = challenge
        self.algorithms = algorithms
        self.smoke_test = smoke_test
        self.patience_manager: Optional[AdaptivePatienceManager] = None
        self.scheduler = DiscoveryAwareScheduler(algorithms, challenge)
        self.insight_generator = ScientificInsightGenerator(challenge)
        self.timing_manager = ScientificTimingManager()
        self.results: Dict[str, Any] = {}
        
    def execute_discovery_session(self, patience_budget: PatienceBudget) -> DiscoveryResults:
        """
        Execute a scientifically-driven comparison within patience constraints.
        
        Args:
            patience_budget: User's patience budget
            
        Returns:
            DiscoveryResults object with comparison results and insights
        """
        console.print(f"[bold blue]🔬 Starting Scientific Discovery Session[/bold blue]")
        console.print(f"[cyan]Challenge: {self.challenge.name}[/cyan]")
        console.print(f"[cyan]Scientific Question: {self.challenge.scientific_question}[/cyan]")
        
        # Initialize patience manager
        self.patience_manager = AdaptivePatienceManager(patience_budget)
        
        # Execute baseline evaluation
        baseline_results = self._run_baseline_evaluation()
        
        # Execute hyperparameter optimization
        optimization_results = self._run_hyperparameter_optimization(baseline_results)
        
        # Execute final evaluation
        final_results = self._run_final_evaluation(optimization_results)
        
        # Generate insights
        insights = self._generate_insights(final_results)
        
        # Compile results
        discovery_results = DiscoveryResults(
            challenge=self.challenge,
            algorithm_results=final_results,
            insights=insights,
            timing_data=self.timing_manager.export_metrics(),
            metadata={
                'session_start_time': self.patience_manager.start_time,
                'total_elapsed_time': self.patience_manager.get_elapsed_time(),
                'total_budget': self.patience_manager.total_budget,
                'remaining_budget': self.patience_manager.get_remaining_budget()
            }
        )
        
        console.print("[green]✅ Scientific discovery session completed![/green]")
        return discovery_results
    
    def _run_baseline_evaluation(self) -> Dict[str, Any]:
        """Run baseline evaluation for all algorithms."""
        console.print("[bold blue]⚡ Running Baseline Evaluation...[/bold blue]")
        
        # Allocate time for baseline evaluation
        priority = self.scheduler.prioritize_exploration()
        allocation = self.patience_manager.allocate_for_phase(
            ExplorationPhase.BASELINE_EVALUATION, 
            discovery_potential=0.5  # Moderate potential
        )
        
        start_time = time.time()
        baseline_results = {}
        
        # Run baseline for each algorithm
        for algorithm_name in priority.algorithm_order:
            try:
                console.print(f"[cyan]Running baseline for {algorithm_name}...[/cyan]")
                
                metrics = self._run_model(algorithm_name, "baseline")
                baseline_results[algorithm_name] = metrics
                
                console.print(f"[green]✅ {algorithm_name} baseline completed[/green]")
            except Exception as e:
                console.print(f"[red]❌ {algorithm_name} baseline failed: {str(e)}[/red]")
        
        elapsed_time = time.time() - start_time
        self.patience_manager.update_patience_consumption(elapsed_time, ExplorationPhase.BASELINE_EVALUATION)
        self.timing_manager.record_discovery_timing("baseline_evaluation", elapsed_time, 0)
        
        return baseline_results
    
    def _run_hyperparameter_optimization(self, baseline_results: Dict[str, Any]) -> Dict[str, Any]:
        """Run hyperparameter optimization with adaptive depth."""
        console.print("[bold blue]🔍 Running Hyperparameter Optimization...[/bold blue]")
        
        optimization_results = {}
        
        # Allocate time for optimization
        allocation = self.patience_manager.allocate_for_phase(
            ExplorationPhase.HYPERPARAMETER_OPTIMIZATION,
            discovery_potential=0.8  # High potential
        )
        
        start_time = time.time()
        
        # Get optimization priority
        priority = self.scheduler.prioritize_exploration(baseline_results)
        
        # Run optimization for each algorithm based on priority
        for algorithm_name in priority.algorithm_order:
            try:
                console.print(f"[cyan]Optimizing {algorithm_name}...[/cyan]")
                
                # Adjust exploration depth
                exploration_depth = self.scheduler.adjust_algorithm_depth(algorithm_name, None)
                
                # Run optimization
                best_params = self._run_optimization(algorithm_name, exploration_depth)
                optimization_results[algorithm_name] = {
                    'best_params': best_params,
                    'exploration_depth': exploration_depth
                }
                
                console.print(f"[green]✅ {algorithm_name} optimization completed[/green]")
            except Exception as e:
                console.print(f"[red]❌ {algorithm_name} optimization failed: {str(e)}[/red]")
        
        elapsed_time = time.time() - start_time
        self.patience_manager.update_patience_consumption(elapsed_time, ExplorationPhase.HYPERPARAMETER_OPTIMIZATION)
        self.timing_manager.record_discovery_timing("hyperparameter_optimization", elapsed_time, 2)  # Simulated insights
        
        return optimization_results
    
    def _run_final_evaluation(self, optimization_results: Dict[str, Any]) -> Dict[str, Any]:
        """Run final evaluation with optimized parameters."""
        console.print("[bold blue]🏆 Running Final Evaluation...[/bold blue]")
        
        final_results = {}
        
        # Allocate time for final evaluation
        allocation = self.patience_manager.allocate_for_phase(
            ExplorationPhase.FINAL_EVALUATION,
            discovery_potential=0.7  # Good potential
        )
        
        start_time = time.time()
        
        # Evaluate both baseline and optimized versions
        for algorithm_name in optimization_results.keys():
            try:
                console.print(f"[cyan]Running final evaluation for {algorithm_name}...[/cyan]")
                
                # Run baseline evaluation
                baseline_metrics = self._run_model(algorithm_name, "final_baseline")
                final_results[f"{algorithm_name}_baseline"] = baseline_metrics
                
                # Run optimized evaluation
                # For now, we'll just re-run the baseline until optimization is integrated
                optimized_metrics = self._run_model(algorithm_name, "final_optimized")
                final_results[f"{algorithm_name}_optimized"] = optimized_metrics
                
                console.print(f"[green]✅ {algorithm_name} final evaluation completed[/green]")
            except Exception as e:
                console.print(f"[red]❌ {algorithm_name} final evaluation failed: {str(e)}[/red]")
        
        elapsed_time = time.time() - start_time
        self.patience_manager.update_patience_consumption(elapsed_time, ExplorationPhase.FINAL_EVALUATION)
        self.timing_manager.record_discovery_timing("final_evaluation", elapsed_time, 1)  # Simulated insights
        
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
            insight.discovery_potential = self.insight_generator.classify_discovery_potential(insight)
        
        elapsed_time = time.time() - start_time
        self.patience_manager.update_patience_consumption(elapsed_time, ExplorationPhase.INSIGHT_GENERATION)
        self.timing_manager.record_discovery_timing("insight_generation", elapsed_time, len(insights))
        
        # Display insights
        if insights:
            console.print(f"[green]✅ Generated {len(insights)} scientific insights:[/green]")
            for i, insight in enumerate(insights, 1):
                console.print(f"  {i}. [{insight.type.upper()}] Confidence: {insight.confidence:.2f}, "
                             f"Potential: {insight.discovery_potential:.2f}")
        else:
            console.print("[yellow]⚠️ No significant insights generated[/yellow]")
        
        return insights

    def _run_model(self, algorithm_name: str, run_type: str) -> Dict[str, Any]:
        """Run a single model evaluation."""
        model_config = get_model_config(algorithm_name)
        if not model_config:
            raise ValueError(f"Could not find configuration for model: {algorithm_name}")

        run_config = RunConfig(
            smoke_test=self.smoke_test,
            study_name=f"{self.challenge.id}_{run_type}",
            logger_callback=logger_callback,
        )
        
        training_config = TrainingConfig() # Use default training config
        
        metrics = run_model_with_fallback(
            model_config=model_config,
            run_config=run_config,
            data_config=self.challenge.dataset,
            training_config=training_config,
            run_identifier=f"{algorithm_name}_{run_type}"
        )
        return metrics
    
    def _run_optimization(self, algorithm_name: str, exploration_depth: Any) -> Dict[str, Any]:
        """Run hyperparameter optimization for a single algorithm."""
        console.print(f"Running optimization for {algorithm_name} with {exploration_depth.max_trials} trials...")
        
        model_to_optimize = get_model_config(algorithm_name)
        if not model_to_optimize:
            raise ValueError(f"Could not find configuration for model: {algorithm_name}")
            
        search_space = get_model_search_space(algorithm_name)
        if not search_space:
            console.print(f"[yellow]No search space defined for {algorithm_name}, skipping optimization.[/yellow]")
            return {}

        opt_config = OptimizationConfig(
            n_trials=exploration_depth.max_trials,
            n_jobs=1, # For now, run sequentially
            storage="sqlite:///:memory:", # Use in-memory DB for simplicity
            model_to_optimize=model_to_optimize,
            search_space=search_space,
        )

        run_config = RunConfig(
            smoke_test=self.smoke_test,
            study_name=f"{self.challenge.id}_{algorithm_name}_opt",
            logger_callback=logger_callback,
        )

        config = ExperimentConfig(
            mode="optimize",
            run_config=run_config,
            data_config=self.challenge.dataset,
            training_config=TrainingConfig(), # Use default
            optimization_config=opt_config,
        )

        try:
            results = run_optimization(config)
            return results.get("best_params", {})
        except Exception as e:
            console.print(f"[red]Error during optimization for {algorithm_name}: {e}[/red]")
            return {}
    
    def generate_insights(self) -> List[ScientificInsight]:
        """
        Generate insights from the comparison results.
        
        Returns:
            List of ScientificInsight objects
        """
        # This method would be used to generate additional insights after the main session
        if not self.results:
            console.print("[yellow]⚠️ No results available for insight generation[/yellow]")
            return []
            
        return self.insight_generator.extract_insights(self.results)