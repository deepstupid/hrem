"""Enhanced adaptive demo runner for the HRM System TUI with comprehensive control and instrumentation.

This module provides an adaptive demo runner that adjusts its behavior based on:
- Available time based on user patience levels
- Model performance characteristics
- Hardware capabilities
"""

import time
from typing import Dict, Any, Optional
from rich.console import Console

from hrm_system import (
    ExperimentConfig,
    RunConfig,
    DataConfig,
    TrainingConfig,
    EvaluationConfig,
    OptimizationConfig,
)
from hrm_system.config import HREMParams
from demo_timing_utils import EnhancedTimingManager, EnhancedTimingContext, LoopTracker, GenerationTracker, TrialTracker
from demo_models import get_model_config, get_model_search_space
from hrm_system.reporting import display_final_comparison, display_optimization_results
from demo_shared import get_best_trial_info
from demo_model_runner import run_model_with_fallback, run_trial_with_timing
from demo_enhanced_config import EnhancedDemoConfig

console = Console()

class EnhancedAdaptiveDemoRunner(EnhancedTimingManager):
    """An enhanced adaptive demo runner with comprehensive control and instrumentation.
    
    This runner automatically adapts to:
    - Available time based on user patience levels
    - Model performance characteristics
    - Hardware capabilities
    - Selected demo mode (lightning, adaptive, comprehensive)
    
    The runner provides detailed instrumentation and metrics collection
    for performance analysis and optimization.
    """
    
    def __init__(self, config: ExperimentConfig, enhanced_config: Optional[EnhancedDemoConfig] = None):
        """Initialize the enhanced adaptive demo runner.
        
        Args:
            config: Experiment configuration containing all demo settings
            enhanced_config: Enhanced configuration with detailed control parameters
        """
        super().__init__(collect_detailed_metrics=enhanced_config.instrumentation.collect_detailed_metrics if enhanced_config else True)
        self.config = config
        self.enhanced_config = enhanced_config or EnhancedDemoConfig()
        self.start_time = time.time()
        self.max_demo_time = self._get_max_demo_time()
        self.min_trials = 1  # Minimum trials for meaningful optimization
        self.max_trials = self.enhanced_config.loop_control.max_trials  # Maximum trials from config
        
    def _get_max_demo_time(self) -> int:
        """Calculate maximum demo time based on patience level.
        
        Returns:
            Maximum time in seconds for the demo to run
        """
        # For demo purposes, we'll use a fixed time limit
        if hasattr(self.config.run_config, 'smoke_test') and self.config.run_config.smoke_test:
            return 60  # 1 minute for smoke test
        # Check patience level from config
        patience_level = getattr(self.config.optimization_config, 'patience_level', 'medium')
        if patience_level == 'low':
            return 60  # 1 minute for low patience
        elif patience_level == 'medium':
            return 120  # 2 minutes for medium patience
        else:
            return 180  # 3 minutes for high patience
            
    def get_elapsed_time(self) -> float:
        """Get total elapsed time since demo start.
        
        Returns:
            Elapsed time in seconds
        """
        return time.time() - self.start_time
        
    def get_remaining_time(self) -> float:
        """Get estimated remaining time for demo.
        
        Returns:
            Remaining time in seconds
        """
        return max(0, self.max_demo_time - self.get_elapsed_time())
        
    def get_adaptive_trials(self, model_name: str) -> int:
        """Determine number of trials based on model timing and user patience.
        
        Args:
            model_name: Name of the model to optimize
            
        Returns:
            Number of trials to run for optimization
        """
        # Calculate based on remaining time and model timing
        model_time = self.estimate_model_time(model_name, self.config.run_config.smoke_test)
        remaining_time = self.get_remaining_time()
        
        # Calculate maximum trials we can afford with a safety margin
        # Reserve some time for final comparison
        time_for_optimization = remaining_time * 0.7
        max_trials_by_time = max(1, int(time_for_optimization / (model_time * 1.2)))  # 1.2x buffer
        
        # Apply limits from enhanced config
        trials = max(self.min_trials, max_trials_by_time)
        return min(trials, self.max_trials)
        
    def should_continue_demo(self) -> bool:
        """Check if we should continue the demo based on time constraints.
        
        Returns:
            True if demo should continue, False otherwise
        """
        return self.get_elapsed_time() < self.max_demo_time

    def run_model_with_timing(self, model_config, run_config, data_config, training_config, run_identifier):
        """Run a model and track its execution time with enhanced instrumentation.
        
        Args:
            model_config: Configuration for the model to run
            run_config: Configuration for the run
            data_config: Configuration for the data
            training_config: Configuration for training
            run_identifier: Unique identifier for this run
            
        Returns:
            Tuple of (metrics, elapsed_time)
        """
        operation_name = f"run_{model_config.name}"
        with EnhancedTimingContext(self, operation_name) as timer:
            try:
                metrics = run_model_with_fallback(
                    model_config=model_config,
                    run_config=run_config,
                    data_config=data_config,
                    training_config=training_config,
                    run_identifier=run_identifier
                )
                return metrics, self.end_timer(timer.start_time)
            except Exception as e:
                raise e

    def run_baseline_evaluation(self) -> Dict[str, Any]:
        """Run baseline evaluation for all models with visual feedback and enhanced tracking.
        
        Returns:
            Dictionary mapping model names to their baseline metrics
        """
        console.print("[bold blue]⚡ Running Baseline Evaluation...[/bold blue]")
        
        baseline_results = {}
        model_configs = self.config.evaluation_config.get_models() if self.config.evaluation_config else []
        
        # Run baselines for each model
        run_config = self.config.run_config
        data_config = self.config.data_config
        training_config = self.config.training_config or TrainingConfig()
        
        # Track the entire baseline evaluation loop
        with LoopTracker(self, "baseline_evaluation"):
            for i, model_config in enumerate(model_configs):
                # Track each individual model run
                with LoopTracker(self, f"baseline_model_{model_config.name}"):
                    try:
                        metrics, elapsed = self.run_model_with_timing(
                            model_config=model_config,
                            run_config=run_config,
                            data_config=data_config,
                            training_config=training_config,
                            run_identifier=f"baseline_{model_config.name}"
                        )
                        baseline_results[model_config.name] = metrics
                        
                        console.print(f"[green]✅ {model_config.name} baseline completed ({elapsed:.1f}s)[/green]")
                    except Exception as e:
                        console.print(f"[red]❌ {model_config.name} baseline failed: {str(e)}[/red]")
        
        return baseline_results
        
    def run_hyperparameter_optimization(self) -> Dict[str, Any]:
        """Run hyperparameter optimization with real-time updates and enhanced tracking.
        
        Returns:
            Dictionary mapping model names to their optimization results
        """
        console.print("[bold blue]🔍 Running Hyperparameter Optimization...[/bold blue]")
        
        optimized_results = {}
        opt_config = self.config.optimization_config
        if not opt_config or not opt_config.model_to_optimize or not opt_config.search_space:
            console.print("[yellow]⚠️  No optimization configuration found. Skipping optimization.[/yellow]")
            return optimized_results
            
        model_to_optimize = opt_config.model_to_optimize
        search_space = opt_config.search_space
        
        # Determine adaptive number of trials
        n_trials = self.get_adaptive_trials(model_to_optimize.name)
        
        console.print(f"[cyan]Optimizing {model_to_optimize.name} with {n_trials} trials...[/cyan]")
        
        # Track the entire optimization process
        with LoopTracker(self, f"optimization_process_{model_to_optimize.name}"):
            with EnhancedTimingContext(self, f"optimization_{model_to_optimize.name}") as timer:
                try:
                    import optuna
                    
                    # Setup optimization config
                    config = ExperimentConfig(
                        mode="optimize",
                        run_config=self.config.run_config,
                        data_config=self.config.data_config,
                        training_config=self.config.training_config or TrainingConfig(),
                        optimization_config=opt_config
                    )
                    
                    # Run optimization with trial-by-trial updates
                    study = optuna.create_study(direction="minimize", study_name=f"demo_opt_{model_to_optimize.name}")
                    
                    # Track best value for updates
                    best_value = float('inf')
                    
                    # Custom optimization loop for updates with enhanced tracking
                    for trial_num in range(n_trials):
                        # Track each trial iteration
                        with TrialTracker(self, f"{model_to_optimize.name}_trial_{trial_num}"):
                            if not self.should_continue_demo():
                                console.print("[yellow]⚠️  Demo time limit reached. Stopping optimization.[/yellow]")
                                break
                                
                            # Run a single trial
                            trial = study.ask()
                            try:
                                value, elapsed = run_trial_with_timing(trial, config, self, f"trial_{trial.number}")
                                study.tell(trial, value)
                                if value < best_value:
                                    best_value = value
                                    console.print(f"[bright_green]New best loss: {best_value:.4f}[/bright_green]")
                            except optuna.TrialPruned:
                                console.print(f"[yellow]Trial {trial.number} pruned[/yellow]")
                                study.tell(trial, state=optuna.trial.TrialState.PRUNED)
                            except Exception as e:
                                console.print(f"[red]Trial {trial.number} failed: {str(e)}[/red]")
                                study.tell(trial, state=optuna.trial.TrialState.FAIL)
                    
                    # Store best results
                    best_info = get_best_trial_info(study)
                    if best_info:
                        optimized_results[model_to_optimize.name] = best_info
                        console.print(f"[green]✅ {model_to_optimize.name} optimization completed ({self.end_timer(timer.start_time):.1f}s)![/green]")
                        console.print(f"[bright_green]Best Loss: {best_info['value']:.4f}[/bright_green]")
                    else:
                        console.print(f"[yellow]⚠️  {model_to_optimize.name} optimization completed with no valid trials[/yellow]")
                        
                except Exception as e:
                    console.print(f"[red]❌ {model_to_optimize.name} optimization failed: {str(e)}[/red]")
        
        return optimized_results
        
    def run_final_evaluation(self, optimized_results: Dict[str, Any]) -> Dict[str, Any]:
        """Run final evaluation with both baseline and optimized models with enhanced tracking.
        
        Args:
            optimized_results: Results from optimization
            
        Returns:
            Dictionary mapping model names to their final evaluation metrics
        """
        console.print("[bold blue]🏆 Running Final Evaluation...[/bold blue]")
        
        # Prepare models for final evaluation (baseline + optimized)
        final_results = {}
        
        # Get baseline models
        model_configs = self.config.evaluation_config.get_models() if self.config.evaluation_config else []
        
        # Add baseline results
        run_config = self.config.run_config
        data_config = self.config.data_config
        training_config = self.config.training_config or TrainingConfig()
        
        # Track the entire final evaluation loop
        with LoopTracker(self, "final_evaluation"):
            # Add baseline results
            for model_config in model_configs:
                # Track each baseline model evaluation
                with LoopTracker(self, f"final_baseline_{model_config.name}"):
                    try:
                        metrics, elapsed = self.run_model_with_timing(
                            model_config=model_config,
                            run_config=run_config,
                            data_config=data_config,
                            training_config=training_config,
                            run_identifier=f"final_{model_config.name}"
                        )
                        final_results[model_config.name] = metrics
                        console.print(f"[green]✅ {model_config.name} final evaluation completed ({elapsed:.1f}s)[/green]")
                    except Exception as e:
                        console.print(f"[red]❌ {model_config.name} final evaluation failed: {str(e)}[/red]")
            # Add optimized results with actual evaluation
            for model_name in optimized_results.keys():
                # Track each optimized model evaluation
                with LoopTracker(self, f"final_optimized_{model_name}"):
                    try:
                        # Create model config with optimized parameters
                        best_params = optimized_results[model_name]["params"]
                        model_config = get_model_config(model_name).model_copy(deep=True)
                        
                        if "hrem" in model_config.algorithm_class.lower():
                            model_config.hrem_params = HREMParams(**best_params)
                        else:
                            model_config.arch_overrides = best_params
                            
                        # Run evaluation with optimized parameters
                        metrics, elapsed = self.run_model_with_timing(
                            model_config=model_config,
                            run_config=run_config,
                            data_config=data_config,
                            training_config=training_config,
                            run_identifier=f"optimized_{model_name}"
                        )
                        
                        final_results[f"{model_name}_optimized"] = metrics
                        console.print(f"[green]✅ {model_name} optimized evaluation completed ({elapsed:.1f}s)[/green]")
                    except Exception as e:
                        console.print(f"[red]❌ {model_name} optimized evaluation failed: {str(e)}[/red]")
                        # Fallback to baseline metrics if optimization evaluation fails
                        if model_name in final_results:
                            final_results[f"{model_name}_optimized"] = final_results[model_name]
        
        console.print("[green]✅ Final evaluation completed![/green]")
        return final_results
                
    def display_timing_summary(self):
        """Display a comprehensive summary of timing information."""
        elapsed_total = time.time() - self.start_time
        console.print(f"[bold]⏱️  Demo completed in {elapsed_total:.1f} seconds[/bold]")
        
        # Display comprehensive metrics if enabled
        if self.collect_detailed_metrics:
            self.display_comprehensive_summary()
            
    def export_detailed_metrics(self, filepath: str = "demo_metrics.json"):
        """Export detailed metrics to a file.
        
        Args:
            filepath: Path to save the metrics file
        """
        if self.enhanced_config.instrumentation.export_metrics:
            self.save_metrics_to_file(filepath)