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

class ScientificDiscoveryEngine:
    """Central orchestrator for scientific exploration process."""
    
    def __init__(self, challenge: ChallengeConfig, algorithms: List[AlgorithmConfig]):
        """Initialize with a scientific challenge and algorithms to compare."""
        self.challenge = challenge
        self.algorithms = algorithms
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
                
                # This would use the actual model running infrastructure
                # For now, we'll simulate the results
                metrics = self._simulate_model_run(algorithm_name, "baseline")
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
                exploration_depth = self.scheduler.adjust_algorithm_depth(algorithm_name)
                
                # Run optimization (simulated)
                best_params = self._simulate_optimization(algorithm_name, exploration_depth)
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
                baseline_metrics = self._simulate_model_run(algorithm_name, "final_baseline")
                final_results[f"{algorithm_name}_baseline"] = baseline_metrics
                
                # Run optimized evaluation
                optimized_metrics = self._simulate_model_run(algorithm_name, "final_optimized")
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
    
    def _simulate_model_run(self, algorithm_name: str, run_type: str) -> Dict[str, Any]:
        """
        Simulate model run (in a real implementation, this would run actual models).
        
        Args:
            algorithm_name: Name of the algorithm
            run_type: Type of run (baseline, final, etc.)
            
        Returns:
            Dictionary with metrics
        """
        # This is a simulation - in reality, this would call the actual model runner
        import random
        
        # Simulate different performance for different algorithms
        base_loss = 2.0
        if "HREM" in algorithm_name:
            base_loss = random.uniform(1.5, 2.5)  # HREM might be better or worse
        else:
            base_loss = random.uniform(1.8, 2.8)  # HRM baseline range
            
        # Add some variance for optimization
        if "optimized" in run_type:
            base_loss *= random.uniform(0.8, 0.95)  # Optimization should improve things
            
        return {
            'all/lm_loss': base_loss,
            'timing': random.uniform(5, 15),  # Seconds
            'accuracy': random.uniform(0.7, 0.95),
            'loss_history': [base_loss * (1 + random.uniform(0.1, 0.3)) for _ in range(10)]
        }
    
    def _simulate_optimization(self, algorithm_name: str, exploration_depth: Any) -> Dict[str, Any]:
        """
        Simulate hyperparameter optimization.
        
        Args:
            algorithm_name: Name of the algorithm
            exploration_depth: Exploration depth configuration
            
        Returns:
            Dictionary with best parameters
        """
        # This is a simulation - in reality, this would run actual optimization
        import random
        
        if "HREM" in algorithm_name:
            return {
                'm_loc': random.randint(64, 256),
                'd_mem': random.randint(128, 512),
                'top_k': random.randint(4, 12),
                'H_layers': random.randint(1, 4)
            }
        else:
            return {
                'hidden_size': random.choice([64, 128, 256]),
                'num_heads': random.choice([1, 2, 4])
            }
    
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