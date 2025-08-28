"""Unified demo runner for the HRM/HREM system.

This module provides a unified interface for running demos in different modes
(lightning, adaptive, comprehensive) with consistent APIs and enhanced features.
"""

import time
import optuna
from typing import Dict, Any, List, Optional
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

from hrm_system import (
    ExperimentConfig,
    RunConfig,
    DataConfig,
    EvaluationConfig,
    OptimizationConfig,
    TrainingConfig,
)
from hrm_system.config import HREMParams
from hrm_system.reporting import display_final_comparison, display_optimization_results

from demo_config import DemoConfig, DemoMode
from demo_models import get_model_configs, get_model_config, get_model_search_space
from demo_shared import get_best_trial_info
from demo_model_runner import run_model_with_fallback, run_trial_with_timing, get_dataset_config
from demo_patience_manager import PatienceManager
from demo_visualization import plot_hyperparameter_pca

console = Console()

class DemoRunner:
    """Unified demo runner for all demo modes."""
    
    def __init__(self, config: DemoConfig):
        """Initialize the demo runner."""
        self.config = config
        self.start_time = time.time()
        self.metrics = {}
        self.timing_metrics = {}
        self.patience_manager = None
        self.demo_state = {
            "current_step": "Initializing",
            "progress": 0,
            "details": "Setting up the demo environment."
        }
        
    def get_elapsed_time(self) -> float:
        """Get total elapsed time since demo start."""
        return time.time() - self.start_time
        
    def should_continue_demo(self) -> bool:
        """Check if we should continue the demo based on time constraints."""
        return self.get_elapsed_time() < self.config.loop_control.time_limit

    def run_model_with_timing(self, model_config, run_config, data_config, training_config, run_identifier):
        """Run a model and track its execution time."""
        start_time = time.time()
        try:
            metrics = run_model_with_fallback(
                model_config=model_config,
                run_config=run_config,
                data_config=data_config,
                training_config=training_config,
                run_identifier=run_identifier
            )
            elapsed = time.time() - start_time
            return metrics, elapsed
        except Exception as e:
            elapsed = time.time() - start_time
            raise e

    def run_baseline_evaluation(self) -> Dict[str, Any]:
        """Run baseline evaluation for all models."""
        self.demo_state.update({
            "current_step": "Baseline Evaluation",
            "progress": 0,
            "details": "Running baseline evaluation for all models to establish initial performance and timing metrics."
        })
        console.print(Panel(f"[bold blue]📊 {self.demo_state['current_step']} ({self.config.demo_mode.value.title()} Mode)[/bold blue]\n[dim]{self.demo_state['details']}[/dim]", expand=False))
        
        model_configs = get_model_configs(self.config.models)
        
        run_config = RunConfig(
            smoke_test=self.config.smoke_test,
            study_name=f"{self.config.demo_mode.value}_baseline"
        )
        
        try:
            data_config = get_dataset_config(
                f"{self.config.dataset}-{self.config.task}" if self.config.dataset == "synthetic" else self.config.dataset,
                self.config.smoke_test,
                self.config.num_aug
            )
        except Exception as e:
            console.print(f"[red]Error accessing dataset: {str(e)}[/red]")
            return {}
        
        training_config = TrainingConfig(
            epochs=self.config.loop_control.max_epochs,
            eval_interval=self.config.loop_control.eval_interval,
            global_batch_size=self.config.loop_control.batch_size,
            smoke_test=self.config.smoke_test
        )
        
        eval_config_dict = {"n_runs": 1}
        for i, model_config in enumerate(model_configs):
            eval_config_dict[f"model_{chr(ord('a') + i)}"] = model_config
        eval_config = EvaluationConfig(**eval_config_dict)
        
        baseline_results = {}
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            total_models = len(model_configs)
            main_task = progress.add_task("[cyan]Baseline Evaluation Progress[/cyan]", total=total_models)

            for i, model_config in enumerate(model_configs):
                self.demo_state["details"] = f"Running baseline for {model_config.name}..."
                progress.update(main_task, description=f"[cyan]Running {model_config.name} baseline...[/cyan]")

                try:
                    metrics, elapsed = self.run_model_with_timing(
                        model_config=model_config,
                        run_config=run_config,
                        data_config=data_config,
                        training_config=training_config,
                        run_identifier=f"baseline_{model_config.name}"
                    )
                    baseline_results[model_config.name] = metrics
                    
                    # Store timing metrics
                    if 'avg_epoch_time' in metrics:
                        self.timing_metrics[model_config.name] = metrics['avg_epoch_time']
                        console.print(f"[green]✅ {model_config.name} baseline completed in {elapsed:.1f}s (avg epoch: {metrics['avg_epoch_time']:.2f}s)[/green]")
                    else:
                         console.print(f"[green]✅ {model_config.name} baseline completed in {elapsed:.1f}s[/green]")

                except Exception as e:
                    console.print(f"[red]Error running {model_config.name} baseline: {str(e)}[/red]")

                progress.update(main_task, advance=1)
                self.demo_state["progress"] = (i + 1) / total_models

        return baseline_results

    def run_hyperparameter_optimization(self, model_name: str, num_trials: int) -> Dict[str, Any]:
        """Run hyperparameter optimization for a specific model."""
        self.demo_state.update({
            "current_step": "Hyperparameter Optimization",
            "progress": 0,
            "details": f"Optimizing {model_name} to find the best hyperparameters."
        })
        console.print(Panel(f"[bold blue]🔍 {self.demo_state['current_step']} for {model_name} ({self.config.demo_mode.value.title()} Mode)[/bold blue]\n[dim]{self.demo_state['details']}[/dim]", expand=False))
        
        # Get model configuration
        model_config = get_model_config(model_name)
        if not model_config:
            console.print(f"[yellow]⚠️  Model {model_name} not found. Skipping optimization.[/yellow]")
            return {}
        
        # Get search space
        search_space = get_model_search_space(model_name)
        if not search_space:
            console.print(f"[yellow]⚠️  No search space defined for {model_name}. Skipping optimization.[/yellow]")
            return {}
        
        # Create experiment configurations
        run_config = RunConfig(
            smoke_test=self.config.smoke_test,
            study_name=f"{self.config.demo_mode.value}_optimization"
        )
        
        try:
            data_config = get_dataset_config(
                f"{self.config.dataset}-{self.config.task}" if self.config.dataset == "synthetic" else self.config.dataset,
                self.config.smoke_test,
                self.config.num_aug
            )
        except Exception as e:
            console.print(f"[red]Error accessing dataset: {str(e)}[/red]")
            return {}
        
        training_config = TrainingConfig(
            epochs=self.config.loop_control.max_epochs,
            eval_interval=self.config.loop_control.eval_interval,
            global_batch_size=self.config.loop_control.batch_size,
            smoke_test=self.config.smoke_test
        )
        
        # Setup optimization configuration
        opt_config = OptimizationConfig(
            n_trials=num_trials,
            n_jobs=self.config.loop_control.n_jobs,
            n_final_runs=self.config.loop_control.n_final_runs,
            storage=f"sqlite:///experiments/optuna_{model_name}_{self.config.demo_mode.value}_demo.db",
            model_to_optimize=model_config,
            search_space=search_space
        )
        
        # Create experiment configuration
        experiment_config = ExperimentConfig(
            mode="optimize",
            run_config=run_config,
            data_config=data_config,
            training_config=training_config,
            optimization_config=opt_config
        )
        
        # Run optimization with live updates
        optimized_results = {}
        
        with Live(console=console, refresh_per_second=4) as live_display:
            optimization_panels = []
            
            # Check time before starting optimization
            if not self.should_continue_demo():
                console.print("[yellow]⚠️  Demo time limit reached. Stopping optimization.[/yellow]")
                return optimized_results
                
            optimization_panels.append(f"[cyan]Optimizing {model_name}...[/cyan]")
            live_display.update(Panel("\\n".join(optimization_panels), title="Optimization Status"))
            
            # Create study
            study = optuna.create_study(direction="minimize", study_name=f"demo_opt_{model_name}")
            
            # Track best value for live updates
            best_value = float('inf')
            
            # Custom optimization loop for live updates
            for trial_num in range(num_trials):
                if not self.should_continue_demo():
                    console.print("[yellow]⚠️  Demo time limit reached. Stopping optimization.[/yellow]")
                    break
                    
                # Update display with trial progress
                optimization_panels[-1] = (
                    f"[cyan]Optimizing {model_name} (Trial {trial_num+1}/{num_trials})...[/cyan]\n"
                    f"[bright_green]Current Best: {best_value:.4f}[/bright_green]"
                )
                live_display.update(Panel("\n".join(optimization_panels), title="Optimization Status"))
                
                # Run a single trial
                trial = study.ask()
                try:
                    value, elapsed = run_trial_with_timing(trial, experiment_config)
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
            best_info = get_best_trial_info(study)
            if best_info:
                optimized_results[model_name] = best_info
                optimization_panels[-1] = (
                    f"[green]✅ {model_name} optimization completed![/green]\n"
                    f"[bright_green]Best Loss: {best_info['value']:.4f}[/bright_green]"
                )
            else:
                optimization_panels[-1] = f"[yellow]⚠️  {model_name} optimization completed with no valid trials[/yellow]"
                
            live_display.update(Panel("\n".join(optimization_panels), title="Optimization Status"))
        
        # Generate and display PCA plot
        plot_path = plot_hyperparameter_pca(study, model_name)
        if plot_path:
            console.print(f"[bold green]📊 PCA plot of hyperparameter space saved to:[/bold green] [cyan]{plot_path}[/cyan]")

        return optimized_results

    def run_final_evaluation(self, baseline_results: Dict[str, Any], optimized_results: Dict[str, Any]) -> Dict[str, Any]:
        """Run final evaluation with both baseline and optimized models."""
        self.demo_state.update({
            "current_step": "Final Evaluation",
            "progress": 0,
            "details": "Running final evaluation with optimized models to measure improvement."
        })
        console.print(Panel(f"[bold blue]🏆 {self.demo_state['current_step']} ({self.config.demo_mode.value.title()} Mode)[/bold blue]\n[dim]{self.demo_state['details']}[/dim]", expand=False))
        
        # Prepare models for final evaluation (baseline + optimized)
        final_results = {}
        
        # Copy baseline results
        for model_name, metrics in baseline_results.items():
            final_results[model_name] = metrics
            
        # Add optimized results with actual evaluation
        for model_name in optimized_results.keys():
            try:
                # Create model config with optimized parameters
                best_params = optimized_results[model_name]["params"]
                model_config = get_model_config(model_name).model_copy(deep=True)
                
                if "hrem" in model_config.algorithm_class.lower():
                    model_config.hrem_params = HREMParams(**best_params)
                else:
                    model_config.arch_overrides = best_params
                    
                # Create experiment configurations
                run_config = RunConfig(
                    smoke_test=self.config.smoke_test,
                    study_name=f"{self.config.demo_mode.value}_final"
                )
                
                try:
                    data_config = get_dataset_config(
                        f"{self.config.dataset}-{self.config.task}" if self.config.dataset == "synthetic" else self.config.dataset,
                        self.config.smoke_test,
                        self.config.num_aug
                    )
                except Exception as e:
                    console.print(f"[red]Error accessing dataset: {str(e)}[/red]")
                    continue
                
                training_config = TrainingConfig(
                    epochs=self.config.loop_control.max_epochs,
                    eval_interval=self.config.loop_control.eval_interval,
                    global_batch_size=self.config.loop_control.batch_size,
                    smoke_test=self.config.smoke_test
                )
                
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
        
        return final_results

    def display_summary(self):
        """Display a summary of the demo execution."""
        elapsed_total = self.get_elapsed_time()
        console.print(f"[bold]⏱️  Demo completed in {elapsed_total:.1f} seconds[/bold]")
        
        # Display mode-specific summary
        mode_descriptions = {
            DemoMode.LIGHTNING: "⚡ Lightning-fast demo with minimal iterations",
            DemoMode.ADAPTIVE: "🧠 Adaptive demo that adjusts based on available time",
            DemoMode.COMPREHENSIVE: "🔬 Comprehensive demo with detailed analysis"
        }
        
        description = mode_descriptions.get(self.config.demo_mode, "Demo")
        console.print(f"[italic]{description}[/italic]")

    def export_metrics(self, filepath: str = None):
        """Export metrics to a file."""
        if not self.config.instrumentation.export_metrics:
            return
            
        if filepath is None:
            filepath = f"{self.config.demo_mode.value}_demo_metrics.json"
            
        try:
            import json
            with open(filepath, 'w') as f:
                json.dump(self.metrics, f, indent=2)
            console.print(f"[green]✅ Metrics saved to {filepath}[/green]")
        except Exception as e:
            console.print(f"[red]❌ Failed to save metrics to {filepath}: {str(e)}[/red]")

def run_demo(config: DemoConfig):
    """Run a demo with the specified configuration."""
    console.clear()
    
    # Display mode header
    mode_headers = {
        DemoMode.LIGHTNING: "[bold blue]⚡ HRM vs HREM: Lightning-Fast Algorithm Comparison[/bold blue]",
        DemoMode.ADAPTIVE: "[bold blue]🧠 HRM vs HREM: Adaptive Algorithm Comparison[/bold blue]",
        DemoMode.COMPREHENSIVE: "[bold blue]🔬 HRM vs HREM: Comprehensive Algorithm Comparison[/bold blue]"
    }
    
    header = mode_headers.get(config.demo_mode, "[bold blue]HRM vs HREM Demo[/bold blue]")
    console.print(Panel(header, expand=False))
    
    # Display configuration
    console.print("\n[bold]Demo Configuration:[/bold]")
    config_table = Table(show_header=True, header_style="bold magenta")
    config_table.add_column("Parameter", style="cyan")
    config_table.add_column("Value")
    
    config_table.add_row("Demo Mode", config.demo_mode.value)
    config_table.add_row("Dataset", f"{config.dataset}-{config.task}" if config.dataset == "synthetic" else config.dataset)
    config_table.add_row("Models", ", ".join(config.models))
    config_table.add_row("Max Epochs", str(config.loop_control.max_epochs))
    config_table.add_row("Max Trials", str(config.loop_control.max_trials))
    config_table.add_row("Batch Size", str(config.loop_control.batch_size))
    config_table.add_row("Export Metrics", str(config.instrumentation.export_metrics))
    
    console.print(config_table)
    
    # Initialize demo runner
    demo_runner = DemoRunner(config)
    
    # --- STEP 1: Baseline Evaluation ---
    baseline_results = demo_runner.run_baseline_evaluation()
    
    # Display baseline results
    display_final_comparison("📊 Baseline Results", baseline_results)
    console.print("[bold green]✅ Baseline evaluation completed![/bold green]\n")
    
    # Check if we should continue with optimization
    if not demo_runner.should_continue_demo():
        console.print("[yellow]⚠️  Demo time limit reached. Skipping optimization phase.[/yellow]")
        demo_runner.display_summary()
        demo_runner.export_metrics()
        return
    
    # --- STEP 2: Hyperparameter Optimization ---
    demo_runner.patience_manager = PatienceManager(
        patience_in_seconds=int(config.patience_level),
        timing_metrics=demo_runner.timing_metrics
    )

    # Run optimization for each model that has a search space
    optimized_results = {}
    for model_name in config.models:
        if not demo_runner.should_continue_demo():
            console.print("[yellow]⚠️  Demo time limit reached. Stopping optimization.[/yellow]")
            break
            
        model_search_space = get_model_search_space(model_name)
        if model_search_space:
            num_trials = demo_runner.patience_manager.get_optimization_trial_budget(
                model_name=model_name,
                epochs_per_trial=config.loop_control.max_epochs,
                time_spent_so_far=demo_runner.get_elapsed_time()
            )
            console.print(f"[bold cyan]Dynamic trial budget for {model_name}: {num_trials} trials[/bold cyan]")

            model_opt_results = demo_runner.run_hyperparameter_optimization(model_name, num_trials)
            optimized_results.update(model_opt_results)
    
    # Display optimization results
    if optimized_results:
        console.print("[bold]Optimization Results:[/bold]")
        for model_name, best_info in optimized_results.items():
            console.print(f"  {model_name}: Best Loss = {best_info['value']:.4f}")
    else:
        console.print("[yellow]No optimization performed.[/yellow]")
    
    console.print("[bold green]✅ Hyperparameter optimization completed![/bold green]\n")
    
    # Check if we should continue with final evaluation
    if not demo_runner.should_continue_demo():
        console.print("[yellow]⚠️  Demo time limit reached. Skipping final evaluation.[/yellow]")
        demo_runner.display_summary()
        demo_runner.export_metrics()
        return
    
    # --- STEP 3: Final Comparison ---
    console.print(Panel("[bold]🏆 Step 3: Final Comparison[/bold]", expand=False))
    
    final_results = demo_runner.run_final_evaluation(baseline_results, optimized_results)
    
    # Display final results
    console.print("\n[bold magenta]🎨 Final Results:[/bold magenta]")
    display_final_comparison("🏆 Final Comparison", final_results)
    
    # Show exciting conclusion
    demo_runner.display_summary()
    demo_runner.export_metrics()
    
    # Display mode-specific conclusion
    mode_conclusions = {
        DemoMode.LIGHTNING: (
            f"[bold green]🎉 Lightning Demo Finished Successfully![/bold green]\n"
            f"[italic]Ultra-fast optimization provides immediate feedback![/italic]\n"
            f"[bold blue]💡 Key insight: Less is more - faster iterations, quicker insights![/bold blue]"
        ),
        DemoMode.ADAPTIVE: (
            f"[bold green]🎉 Adaptive Demo Finished Successfully![/bold green]\n"
            f"[italic]Smart optimization adapts to your time constraints![/italic]\n"
            f"[bold blue]💡 Key insight: Balance between speed and thoroughness![/bold blue]"
        ),
        DemoMode.COMPREHENSIVE: (
            f"[bold green]🎉 Comprehensive Demo Finished Successfully![/bold green]\n"
            f"[italic]Thorough analysis provides deep insights![/italic]\n"
            f"[bold blue]💡 Key insight: Detailed exploration reveals hidden patterns![/bold blue]"
        )
    }
    
    conclusion = mode_conclusions.get(config.demo_mode, "[bold green]🎉 Demo Finished Successfully![/bold green]")
    console.print(Panel(conclusion, expand=False))