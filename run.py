import os
import json
import time
import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from rich.live import Live
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
import optuna
from typing import List, Dict, Any
from collections import defaultdict

from hrm_system import (
    ExperimentConfig,
    RunConfig,
    DataConfig,
    TrainingConfig,
    EvaluationConfig,
    OptimizationConfig,
    run_evaluation,
    run_optimization,
    run_single_model,
)
from hrm_system.config import HREMParams
from demo_models import (
    list_available_models,
    get_model_configs,
    get_model_config,
    get_model_search_space,
)
from hrm_system.reporting import display_final_comparison, display_optimization_results
from dataset_manager import dataset_manager
from demo_timing_utils import TimingManager, TimingContext

# Import AdaptiveDemoRunner
from adaptive_demo_runner import AdaptiveDemoRunner

console = Console()

# Import the centralized logger callback
from hrm_system import logger_callback

# Import shared function
from demo_model_runner import get_dataset_config

# Import shared function
from demo_model_runner import run_model_with_fallback

@click.group()
def cli():
    """HRM System: A unified interface for evaluation, optimization, and demos."""
    pass

@cli.command()
@click.option("--dataset", default="synthetic", type=click.Choice(["arc", "sudoku", "maze", "synthetic"] + 
                                                                   [f"synthetic-{task}" for task in ["copy", "reverse", "sort", "parity", "duplicate"]]), 
              help="Dataset to use.")
@click.option("--num-aug", default=0, type=int, help="Number of augmentations.")
@click.option("--n-runs", default=1, type=int, help="Number of runs for statistical significance.")
@click.option("--smoke-test", is_flag=True, default=False, help="Run in smoke test mode.")
@click.option("--study-name", default="cli_evaluation", type=str, help="Name for the study.")
@click.option(
    "--models",
    "-m",
    multiple=True,
    default=["HRM", "HREM"],
    type=click.Choice(list_available_models()),
    help="Models to evaluate.",
)
@click.option("--arch-overrides", type=str, help="JSON string for architecture overrides.")
def evaluate(dataset, num_aug, n_runs, smoke_test, study_name, models, arch_overrides):
    """Run a side-by-side comparison of specified models."""
    console.print(f"[bold blue]Starting Evaluation: {study_name}[/bold blue]")
    model_configs = get_model_configs(list(models))
    if not model_configs:
        console.print("[red]Error: No valid models specified for evaluation.[/red]")
        return

    if arch_overrides:
        try:
            overrides = json.loads(arch_overrides)
            for config in model_configs:
                config.arch_overrides.update(overrides)
        except json.JSONDecodeError:
            console.print("[red]Error: Invalid JSON string for --arch-overrides.[/red]")
            return

    eval_config_dict = {"n_runs": n_runs}
    for i, model_config in enumerate(model_configs):
        eval_config_dict[f"model_{chr(ord('a') + i)}"] = model_config
    eval_config = EvaluationConfig(**eval_config_dict)

    # A base training config can be provided for the experiment
    training_config = TrainingConfig()

    # Get dataset configuration
    try:
        data_config = get_dataset_config(dataset, smoke_test, num_aug)
    except Exception as e:
        console.print(f"[red]Error accessing dataset: {str(e)}[/red]")
        return

    config = ExperimentConfig(
        mode="evaluate",
        run_config=RunConfig(smoke_test=smoke_test, study_name=study_name, logger_callback=logger_callback),
        data_config=data_config,
        training_config=training_config,
        evaluation_config=eval_config,
    )
    run_evaluation(config)
    console.print(f"[bold green]Evaluation '{study_name}' finished.[/bold green]")


@cli.command()
@click.option("--dataset", default="synthetic", type=click.Choice(["arc", "sudoku", "maze", "synthetic"] + 
                                                                   [f"synthetic-{task}" for task in ["copy", "reverse", "sort", "parity", "duplicate"]]), 
              help="Dataset to use.")
@click.option("--n-trials", default=10, type=int, help="Number of optimization trials.")
@click.option("--n-jobs", default=1, type=int, help="Number of parallel jobs for Optuna.")
@click.option("--n-final-runs", default=1, type=int, help="Number of final comparison runs.")
@click.option("--storage", default=f"sqlite:///{os.path.abspath('experiments')}/optuna_cli.db", help="Optuna storage URL.")
@click.option("--smoke-test", is_flag=True, default=False, help="Run in smoke test mode.")
@click.option("--study-name", default="cli_optimization", type=str, help="Name for the study.")
@click.option("--model", default="HREM", type=click.Choice(list_available_models()), help="Model to optimize.")
def optimize(dataset, n_trials, n_jobs, n_final_runs, storage, smoke_test, study_name, model):
    """Run hyperparameter optimization for a specified model."""
    console.print(f"[bold blue]Starting Optimization for {model}: {study_name}[/bold blue]")
    model_to_optimize = get_model_config(model)
    if not model_to_optimize:
        console.print(f"[red]Error: Model '{model}' not found.[/red]")
        return
    search_space = get_model_search_space(model)
    if not search_space:
        console.print(f"[red]Error: No search space defined for model '{model}'.[/red]")
        return

    # Get dataset configuration
    try:
        data_config = get_dataset_config(dataset, smoke_test)
    except Exception as e:
        console.print(f"[red]Error accessing dataset: {str(e)}[/red]")
        return

    config = ExperimentConfig(
        mode="optimize",
        run_config=RunConfig(smoke_test=smoke_test, study_name=study_name, logger_callback=logger_callback),
        data_config=data_config,
        optimization_config=OptimizationConfig(n_trials=n_trials, n_jobs=n_jobs, n_final_runs=n_final_runs, storage=storage, model_to_optimize=model_to_optimize, search_space=search_space)
    )
    run_optimization(config)
    console.print(f"[bold green]Optimization '{study_name}' finished.[/bold green]")

# Import shared function
from demo_shared import clear_optuna_studies

@cli.command()
@click.option("--dataset", default="synthetic", type=click.Choice(["arc", "sudoku", "maze", "synthetic"] + 
                                                                   [f"synthetic-{task}" for task in ["copy", "reverse", "sort", "parity", "duplicate"]]), 
              help="Dataset to use.")
@click.option("--num-aug", default=0, type=int, help="Number of augmentations.")
@click.option("--n-trials", default=1, type=int, help="Number of optimization trials.")
@click.option("--n-jobs", default=1, type=int, help="Number of parallel jobs for Optuna.")
@click.option("--n-final-runs", default=1, type=int, help="Number of final comparison runs.")
@click.option("--smoke-test", is_flag=True, default=False, help="Run in smoke test mode.")
@click.option("--study-name", default="interactive_demo", type=str, help="Name for the study.")
@click.option("--patience", type=click.Choice(["low", "medium", "high"]), default="low", help="User patience level.")
@click.option(
    "--models",
    "-m",
    multiple=True,
    default=["HRM", "HREM"],
    type=click.Choice(list_available_models()),
    help="Models to evaluate.",
)
def demo(dataset, num_aug, n_trials, n_jobs, n_final_runs, smoke_test, study_name, patience, models):
    """Run an exciting, continuous side-by-side algorithm comparison with immediate animated results."""
    console.clear()
    console.print(Panel("[bold blue]🚀 HRM vs HREM: Real-Time Algorithm Comparison[/bold blue]", expand=False))
    
    # Clear any existing Optuna studies
    clear_optuna_studies(study_name)
    
    # Get configurations
    model_configs = get_model_configs(list(models))
    run_config = RunConfig(smoke_test=smoke_test, study_name=study_name, logger_callback=logger_callback)
    training_config = TrainingConfig()
    
    # Get dataset configuration
    try:
        data_config = get_dataset_config(dataset, smoke_test, num_aug)
    except Exception as e:
        console.print(f"[red]Error accessing dataset: {str(e)}[/red]")
        return
    
    # Create experiment config
    eval_config_dict = {"n_runs": 1}
    for i, model_config in enumerate(model_configs):
        eval_config_dict[f"model_{chr(ord('a') + i)}"] = model_config
    eval_config = EvaluationConfig(**eval_config_dict)
    
    experiment_config = ExperimentConfig(
        mode="evaluate",
        run_config=run_config,
        data_config=data_config,
        training_config=training_config,
        evaluation_config=eval_config,
        optimization_config=OptimizationConfig(
            n_trials=n_trials,
            n_jobs=n_jobs,
            n_final_runs=n_final_runs,
            storage=f"sqlite:///{os.path.abspath('experiments')}/optuna_demo.db"
        )
    )
    
    # Initialize adaptive demo runner
    demo_runner = AdaptiveDemoRunner(experiment_config, results_displayer=None)
    
    # Show exciting intro
    console.print("[bold green]⚡ Real-time optimization with immediate results![/bold green]")
    console.print("[italic]Watch as algorithms compete side-by-side...[/italic]\n")
    
    # --- STEP 1: Continuous Baseline Evaluation ---
    console.print(Panel("[bold]⚡ Step 1: Continuous Baseline Evaluation[/bold]", expand=False))
    
    # Use progress bar for exciting visualization
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        baseline_tasks = {}
        baseline_results = {}
        
        # Create progress tasks for each model
        for model_name in models:
            baseline_tasks[model_name] = progress.add_task(
                f"[cyan]Running {model_name} baseline...[/cyan]", 
                total=None
            )
        
        # Run baselines for each model
        for model_config in model_configs:
            try:
                metrics, elapsed = demo_runner.run_model_with_timing(
                    model_config=model_config,
                    run_config=run_config,
                    data_config=data_config,
                    training_config=training_config,
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
    
    # Display baseline results immediately
    display_final_comparison("📊 Baseline Results", baseline_results)
    console.print("[bold green]⚡ Baseline results generated instantly![/bold green]\n")
    
    # Check if we should continue with optimization
    if not demo_runner.should_continue_demo():
        console.print("[yellow]⚠️  Demo time limit reached. Skipping optimization phase.[/yellow]")
        return
    
    # --- STEP 2: Real-Time Optimization with Live Updates ---
    console.print(Panel("[bold]🔍 Step 2: Real-Time Hyperparameter Optimization[/bold]", expand=False))
    console.print("[italic]💡 Watch as each iteration improves performance...[/italic]\n")
    
    optimized_results = {}
    
    # Create a live display for optimization progress
    with Live(console=console, refresh_per_second=4) as live_display:
        optimization_panels = []
        
        for i, model_name in enumerate(models):
            # Check time before starting optimization
            if not demo_runner.should_continue_demo():
                console.print("[yellow]⚠️  Demo time limit reached. Stopping optimization.[/yellow]")
                break
                
            # Determine adaptive number of trials based on remaining time and model timing
            n_trials = demo_runner.get_adaptive_trials(model_name)
            
            optimization_panels.append(f"[cyan]Optimizing {model_name} ({n_trials} trials)...[/cyan]")
            live_display.update(Panel("\n".join(optimization_panels), title="Optimization Status"))
            
            # Run actual optimization
            try:
                model_to_optimize = get_model_config(model_name)
                search_space = get_model_search_space(model_name)
                
                if not model_to_optimize or not search_space:
                    optimization_panels[-1] = f"[red]❌ {model_name} optimization failed (missing config)[/red]"
                    live_display.update(Panel("\n".join(optimization_panels), title="Optimization Status"))
                    continue
                
                # Setup optimization config
                opt_config = OptimizationConfig(
                    n_trials=n_trials,
                    n_jobs=n_jobs,
                    n_final_runs=n_final_runs,
                    storage=f"sqlite:///{os.path.abspath('experiments')}/optuna_{model_name}_{study_name}.db",
                    model_to_optimize=model_to_optimize,
                    search_space=search_space
                )
                
                config = ExperimentConfig(
                    mode="optimize",
                    run_config=run_config,
                    data_config=data_config,
                    optimization_config=opt_config
                )
                
                # Run optimization with trial-by-trial updates
                start_time = time.time()
                study = optuna.create_study(direction="minimize", study_name=study_name)
                
                # Track best value for live updates
                best_value = float('inf')
                
                # Custom optimization loop for live updates
                for trial_num in range(n_trials):
                    if not demo_runner.should_continue_demo():
                        console.print("[yellow]⚠️  Demo time limit reached. Stopping optimization.[/yellow]")
                        break
                        
                    # Update display with trial progress
                    optimization_panels[-1] = (
                        f"[cyan]Optimizing {model_name} (Trial {trial_num+1}/{n_trials})...[/cyan]\n"
                        f"[bright_green]Current Best: {best_value:.4f}[/bright_green]"
                    )
                    live_display.update(Panel("\n".join(optimization_panels), title="Optimization Status"))
                    
                    # Run a single trial
                    trial = study.ask()
                    try:
                        value = run_trial(trial, config)
                        study.tell(trial, value)
                        if value < best_value:
                            best_value = value
                            console.print(f"[bright_green]New best loss: {best_value:.4f}[/bright_green]")
                    except optuna.TrialPruned:
                        study.tell(trial, state=optuna.trial.TrialState.PRUNED)
                    except Exception as e:
                        study.tell(trial, state=optuna.trial.TrialState.FAIL)
                
                elapsed = time.time() - start_time
                
                # Store best results
                best_info = get_best_trial_info(study)
                if best_info:
                    optimized_results[model_name] = best_info
                    optimization_panels[-1] = (
                        f"[green]✅ {model_name} optimization completed ({elapsed:.1f}s)![/green]\n"
                        f"[bright_green]Best Loss: {best_info['value']:.4f}[/bright_green]"
                    )
                else:
                    optimization_panels[-1] = f"[yellow]⚠️  {model_name} optimization completed with no valid trials[/yellow]"
                    
            except Exception as e:
                elapsed = time.time() - start_time
                optimization_panels[-1] = f"[red]❌ {model_name} optimization failed ({elapsed:.1f}s): {str(e)}[/red]"
                
            live_display.update(Panel("\n".join(optimization_panels), title="Optimization Status"))
    
    # Display optimization results
    for model_name, best_info in optimized_results.items():
        display_optimization_results(model_name, {"best_params": best_info['params']})
    
    console.print("[bold green]⚡ Optimization results generated in real-time![/bold green]\n")
    
    # --- STEP 3: Animated Final Comparison ---
    console.print(Panel("[bold]🏆 Step 3: Animated Final Comparison[/bold]", expand=False))
    
    # Prepare models for final evaluation (baseline + optimized)
    final_results = {}
    
    # Copy baseline results
    for model_name, metrics in baseline_results.items():
        final_results[model_name] = metrics
        
    # Add optimized results with actual evaluation
    for model_name in models:
        if model_name in optimized_results:
            try:
                # Create model config with optimized parameters
                best_params = optimized_results[model_name]["params"]
                model_config = get_model_config(model_name).model_copy(deep=True)
                
                if "hrem" in model_config.algorithm_class.lower():
                    model_config.hrem_params = HREMParams(**best_params)
                else:
                    model_config.arch_overrides = best_params
                    
                # Run evaluation with optimized parameters
                metrics, elapsed = demo_runner.run_model_with_timing(
                    model_config=model_config,
                    run_config=run_config,
                    data_config=data_config,
                    training_config=training_config,
                    run_identifier=f"optimized_{model_name}"
                )
                
                final_results[f"{model_name}_optimized"] = metrics
            except Exception as e:
                console.print(f"[red]Error evaluating optimized {model_name}: {str(e)}[/red]")
                # Fallback to baseline metrics if optimization evaluation fails
                final_results[f"{model_name}_optimized"] = baseline_results[model_name]
    
    # Display animated final results
    console.print("\n[bold magenta]🎨 Presenting final results...[/bold magenta]")
    time.sleep(0.5)  # Small pause for dramatic effect
    display_final_comparison("🏆 Final Animated Comparison", final_results)
    
    # Show exciting conclusion
    total_time = demo_runner.get_elapsed_time()
    console.print(Panel(
        f"[bold green]🎉 Demo Finished Successfully![/bold green]\n"
        f"[italic]Total time: {total_time:.1f} seconds[/italic]\n"
        f"[bold blue]💡 Key insight: Real-time optimization provides immediate feedback![/bold blue]",
        expand=False
    ))


@cli.command()
@click.pass_context
def demoui(ctx):
    """Launch the interactive demo UI with menu for choosing challenges."""
    console.clear()
    console.print(Panel("[bold blue]🚀 HRM vs HREM: Interactive Demo Selector[/bold blue]", expand=False))
    
    # Menu for dataset selection
    console.print("[bold]Select a challenge:[/bold]")
    console.print("  1. Synthetic Copy (default) - Fast, simple task")
    console.print("  2. Synthetic Reverse - Medium complexity")
    console.print("  3. Synthetic Sort - Higher complexity")
    console.print("  4. Synthetic Parity - Logical reasoning")
    console.print("  5. Synthetic Duplicate - Pattern recognition")
    console.print("  6. ARC Dataset - Abstract reasoning")
    console.print("  7. Sudoku Dataset - Constraint satisfaction")
    console.print("  8. Maze Dataset - Path finding")
    
    # Get user choice
    while True:
        try:
            choice = int(console.input("\n[yellow]Enter your choice (1-8): [/yellow]"))
            if 1 <= choice <= 8:
                selected_dataset = ["synthetic", "synthetic-reverse", "synthetic-sort", 
                                  "synthetic-parity", "synthetic-duplicate", 
                                  "arc", "sudoku", "maze"][choice - 1]
                break
            else:
                console.print("[red]Invalid choice. Please enter a number between 1 and 8.[/red]")
        except ValueError:
            console.print("[red]Invalid input. Please enter a number.[/red]")
    
    # Menu for patience level
    patience_levels = ["low", "medium", "high"]
    console.print("\n[bold]Select patience level:[/bold]")
    console.print("  1. Quick Demo (faster results, ~45 seconds)")
    console.print("  2. Balanced Demo (moderate exploration, ~2 minutes)")
    console.print("  3. Deep Dive (thorough exploration, ~5 minutes)")
    
    # Get user choice
    while True:
        try:
            choice = int(console.input("\n[yellow]Enter your choice (1-3): [/yellow]"))
            if 1 <= choice <= len(patience_levels):
                selected_patience = patience_levels[choice - 1]
                break
            else:
                console.print("[red]Invalid choice. Please enter a number between 1 and 3.[/red]")
        except ValueError:
            console.print("[red]Invalid input. Please enter a number.[/red]")
    
    # Menu for model selection
    available_models = list_available_models()
    console.print("\n[bold]Select models to compare (comma-separated, e.g., 1,2):[/bold]")
    for i, model in enumerate(available_models, 1):
        console.print(f"  {i}. {model}")
    
    # Get user choices
    while True:
        try:
            choices = console.input("\n[yellow]Enter your choices (e.g., 1,2): [/yellow]")
            selected_indices = [int(x.strip()) for x in choices.split(",")]
            if all(1 <= idx <= len(available_models) for idx in selected_indices):
                selected_models = [available_models[idx - 1] for idx in selected_indices]
                break
            else:
                console.print(f"[red]Invalid choice. Please enter numbers between 1 and {len(available_models)}.[/red]")
        except ValueError:
            console.print("[red]Invalid input. Please enter comma-separated numbers.[/red]")
    
    # Confirm choices
    console.print(f"\n[bold]Your selections:[/bold]")
    console.print(f"  Challenge: {selected_dataset}")
    console.print(f"  Patience: {selected_patience}")
    console.print(f"  Models: {', '.join(selected_models)}")
    
    confirm = console.input("\n[yellow]Start demo with these settings? (y/n): [/yellow]")
    if confirm.lower() != 'y':
        console.print("[red]Demo cancelled.[/red]")
        return
    
    # Run the demo with selected parameters
    console.print("[green]Starting demo...[/green]")
    time.sleep(1)
    
    # Invoke the demo command with our selected arguments
    ctx.invoke(demo, 
               dataset=selected_dataset,
               num_aug=0,
               n_trials=1,
               n_jobs=1,
               n_final_runs=1,
               smoke_test=False,  # Use full demo for better results
               study_name="interactive_demo",
               patience=selected_patience,
               models=selected_models)


@cli.command()
def tui():
    """Launch the Textual User Interface."""
    from tui.main import main as tui_main
    tui_main()

if __name__ == "__main__":
    cli()