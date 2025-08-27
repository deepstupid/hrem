"""Enhanced aggressively optimized demo runner for the HRM System with comprehensive control and instrumentation.

This module provides an aggressively optimized demo runner that focuses on:
- Ultra-fast execution with minimal iterations
- Real-time progress updates and visualization
- Efficient resource utilization
- Comprehensive instrumentation for performance analysis
"""

import time
from typing import Dict, Any, Optional
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.live import Live
from rich.panel import Panel

from hrm_system import (
    ExperimentConfig,
    RunConfig,
    DataConfig,
    EvaluationConfig,
    OptimizationConfig,
    TrainingConfig,
)
from hrm_system.config import HREMParams
from hrm_system.config_demo import TrainingConfig as DemoTrainingConfig
from demo_models import get_model_config, get_model_search_space
from hrm_system.reporting import display_final_comparison, display_optimization_results
from demo_shared import get_best_trial_info
from demo_model_runner import run_model_with_fallback
from demo_timing_utils import EnhancedTimingManager, EnhancedTimingContext, LoopTracker, GenerationTracker, TrialTracker
from demo_enhanced_config import EnhancedDemoConfig

console = Console()

class EnhancedAggressiveDemoRunner(EnhancedTimingManager):
    """An enhanced aggressively optimized demo runner with comprehensive control and instrumentation.
    
    This runner provides an ultra-fast demo experience with:
    - Minimal iterations for quick results
    - Real-time progress visualization
    - Efficient optimization algorithms
    - Comprehensive metrics collection
    
    The runner is designed for users who want immediate feedback
    and are willing to trade some accuracy for speed.
    """
    
    def __init__(self, config: ExperimentConfig, enhanced_config: Optional[EnhancedDemoConfig] = None):
        """Initialize the enhanced aggressive demo runner.
        
        Args:
            config: Experiment configuration containing all demo settings
            enhanced_config: Enhanced configuration with detailed control parameters
        """
        super().__init__(collect_detailed_metrics=enhanced_config.instrumentation.collect_detailed_metrics if enhanced_config else True)
        self.config = config
        self.enhanced_config = enhanced_config or EnhancedDemoConfig()
        self.start_time = time.time()
        self.max_demo_time = self._get_max_demo_time()
        self.min_trials = 1
        self.max_trials = self.enhanced_config.loop_control.max_trials  # Use config value
        
    def _get_max_demo_time(self) -> int:
        """Calculate maximum demo time based on patience level.
        
        Returns:
            Maximum time in seconds for the demo to run
        """
        # For demo purposes, we'll use a fixed time limit
        if hasattr(self.config.run_config, 'smoke_test') and self.config.run_config.smoke_test:
            return 30  # 30 seconds for smoke test
        # Check patience level from config
        patience_level = getattr(self.config.optimization_config, 'patience_level', 'medium')
        if patience_level == 'low':
            return 60  # 1 minute for low patience
        elif patience_level == 'medium':
            return 90  # 1.5 minutes for medium patience
        else:
            return 120  # 2 minutes for high patience
            
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
        console.print("[bold blue]⚡ Running Aggressive Baseline Evaluation...[/bold blue]")
        
        baseline_results = {}
        model_configs = self.config.evaluation_config.get_models() if self.config.evaluation_config else []
        
        # Track the entire baseline evaluation process
        with LoopTracker(self, "aggressive_baseline_evaluation"):
            # Use progress bar for exciting visualization
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TimeElapsedColumn(),
                console=console,
            ) as progress:
                baseline_tasks = {}
                
                # Create progress tasks for each model
                for model_config in model_configs:
                    baseline_tasks[model_config.name] = progress.add_task(
                        f"[cyan]Running {model_config.name} baseline...[/cyan]", 
                        total=None
                    )
                
                # Run baselines for each model
                for model_config in model_configs:
                    # Track each individual model run
                    with LoopTracker(self, f"aggressive_baseline_{model_config.name}"):
                        try:
                            metrics, elapsed = self.run_model_with_timing(
                                model_config=model_config,
                                run_config=self.config.run_config,
                                data_config=self.config.data_config,
                                training_config=DemoTrainingConfig(),  # Use our optimized config
                                run_identifier=f"baseline_{model_config.name}"
                            )
                            baseline_results[model_config.name] = metrics
                            
                            progress.update(
                                baseline_tasks[model_config.name], 
                                description=f"[green]✅ {model_config.name} baseline completed ({elapsed:.1f}s)[/green]"
                            )
                        except Exception as e:
                            console.print(f"[red]Error running {model_config.name} baseline: {str(e)}[/red]")
                            progress.update(
                                baseline_tasks[model_config.name], 
                                description=f"[red]❌ {model_config.name} baseline failed[/red]"
                            )
        
        return baseline_results
        
    def run_hyperparameter_optimization(self) -> Dict[str, Any]:
        """Run hyperparameter optimization with aggressive limits and enhanced tracking.
        
        Returns:
            Dictionary mapping model names to their optimization results
        """
        console.print("[bold blue]🔍 Running Aggressive Hyperparameter Optimization...[/bold blue]")
        
        optimized_results = {}
        
        # Get model to optimize from config
        opt_config = self.config.optimization_config
        if not opt_config or not opt_config.model_to_optimize or not opt_config.search_space:
            console.print("[yellow]⚠️  No optimization configuration found. Skipping optimization.[/yellow]")
            return optimized_results
            
        model_name = opt_config.model_to_optimize.name
        
        # Track the entire optimization process
        with LoopTracker(self, "aggressive_hyperparameter_optimization"):
            # Create a live display for optimization progress
            with Live(console=console, refresh_per_second=4) as live_display:
                optimization_panels = []
                
                # Check time before starting optimization
                if not self.should_continue_demo():
                    console.print("[yellow]⚠️  Demo time limit reached. Stopping optimization.[/yellow]")
                    return optimized_results
                    
                optimization_panels.append(f"[cyan]Optimizing {model_name}...[/cyan]")
                live_display.update(Panel("\n".join(optimization_panels), title="Optimization Status"))
                
                # Track optimization for the model
                with LoopTracker(self, f"aggressive_optimization_{model_name}"):
                    # Run actual optimization with aggressive limits
                    try:
                        model_to_optimize = opt_config.model_to_optimize
                        search_space = opt_config.search_space
                        
                        # Import optuna only when needed
                        import optuna
                        
                        # Setup optimization config with aggressive limits
                        opt_config = OptimizationConfig(
                            n_trials=self.max_trials,
                            n_jobs=1,
                            n_final_runs=1,
                            storage=f"sqlite:///experiments/optuna_{model_name}_demo.db",
                            model_to_optimize=model_to_optimize,
                            search_space=search_space
                        )
                        
                        # Create study
                        study = optuna.create_study(direction="minimize", study_name=f"demo_opt_{model_name}")
                        
                        # Track best value for live updates
                        best_value = float('inf')
                        
                        # Custom optimization loop with aggressive limits and enhanced tracking
                        for trial_num in range(self.max_trials):
                            # Track each trial
                            with TrialTracker(self, f"{model_name}_trial_{trial_num}"):
                                if not self.should_continue_demo():
                                    console.print("[yellow]⚠️  Demo time limit reached. Stopping optimization.[/yellow]")
                                    break
                                    
                                # Update display with trial progress
                                optimization_panels[-1] = (
                                    f"[cyan]Optimizing {model_name} (Trial {trial_num+1}/{self.max_trials})...[/cyan]\n"
                                    f"[bright_green]Current Best: {best_value:.4f}[/bright_green]"
                                )
                                live_display.update(Panel("\n".join(optimization_panels), title="Optimization Status"))
                                
                                # Run a single trial
                                trial = study.ask()
                                try:
                                    # Create a temporary experiment config for this trial
                                    trial_config = ExperimentConfig(
                                        mode="optimize",
                                        run_config=self.config.run_config,
                                        data_config=self.config.data_config,
                                        training_config=DemoTrainingConfig(),  # Use our optimized config
                                        optimization_config=opt_config
                                    )
                                    
                                    # Import the trial runner function
                                    from demo_model_runner import run_trial_with_timing
                                    
                                    value, elapsed = run_trial_with_timing(trial, trial_config)
                                    study.tell(trial, value)
                                    if value < best_value:
                                        best_value = value
                                        console.print(f"[bright_green]New best loss: {best_value:.4f}[/bright_green]")
                                except optuna.TrialPruned:
                                    study.tell(trial, state=optuna.trial.TrialState.PRUNED)
                                except Exception as e:
                                    console.print(f"[red]Trial {trial.number} failed: {str(e)}[/red]")
                                    study.tell(trial, state=optuna.trial.TrialState.FAIL)
                        
                        # Store best results
                        try:
                            best_info = get_best_trial_info(study)
                            if best_info:
                                optimized_results[model_name] = best_info
                                optimization_panels[-1] = (
                                    f"[green]✅ {model_name} optimization completed![/green]\n"
                                    f"[bright_green]Best Loss: {best_info['value']:.4f}[/bright_green]"
                                )
                            else:
                                optimization_panels[-1] = f"[yellow]⚠️  {model_name} optimization completed with no valid trials[/yellow]"
                                
                        except Exception as e:
                            console.print(f"[red]Error getting best trial info: {str(e)}[/red]")
                            optimization_panels[-1] = f"[red]❌ {model_name} optimization failed: {str(e)}[/red]"
                            
                    except Exception as e:
                        optimization_panels[-1] = f"[red]❌ {model_name} optimization failed: {str(e)}[/red]"
                        
                    live_display.update(Panel("\n".join(optimization_panels), title="Optimization Status"))
        
        return optimized_results
        
    def run_final_evaluation(self, baseline_results, optimized_results) -> Dict[str, Any]:
        """Run final evaluation with both baseline and optimized models with enhanced tracking.
        
        Args:
            baseline_results: Results from baseline evaluation
            optimized_results: Results from optimization
            
        Returns:
            Dictionary mapping model names to their final evaluation metrics
        """
        console.print("[bold blue]🏆 Running Final Lightning Evaluation...[/bold blue]")
        
        # Prepare models for final evaluation (baseline + optimized)
        final_results = {}
        
        # Track the entire final evaluation process
        with LoopTracker(self, "aggressive_final_evaluation"):
            # Copy baseline results
            for model_name, metrics in baseline_results.items():
                final_results[model_name] = metrics
                
            # Add optimized results with actual evaluation
            for model_name in optimized_results.keys():
                # Track each optimized model evaluation
                with LoopTracker(self, f"aggressive_final_optimized_{model_name}"):
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
                            run_config=self.config.run_config,
                            data_config=self.config.data_config,
                            training_config=DemoTrainingConfig(),  # Use our optimized config
                            run_identifier=f"optimized_{model_name}"
                        )
                        
                        final_results[f"{model_name}_optimized"] = metrics
                        console.print(f"[green]✅ {model_name} optimized evaluation completed ({elapsed:.1f}s)[/green]")
                    except Exception as e:
                        console.print(f"[red]❌ {model_name} optimized evaluation failed: {str(e)}[/red]")
                        # Fallback to baseline metrics if optimization evaluation fails
                        if model_name in final_results:
                            final_results[f"{model_name}_optimized"] = final_results[model_name]
        
        console.print("[green]✅ Final lightning evaluation completed![/green]")
        return final_results
                
    def display_timing_summary(self):
        """Display a summary of timing information."""
        elapsed_total = time.time() - self.start_time
        console.print(f"\n[bold]⏱️  Demo completed in {elapsed_total:.1f} seconds[/bold]")
        
        # Display comprehensive metrics if enabled
        if self.collect_detailed_metrics:
            # Call the parent class method directly to avoid recursion
            super().display_comprehensive_summary()
            
    def export_detailed_metrics(self, filepath: str = "demo_metrics.json"):
        """Export detailed metrics to a file.
        
        Args:
            filepath: Path to save the metrics file
        """
        if self.enhanced_config.instrumentation.export_metrics:
            self.save_metrics_to_file(filepath)