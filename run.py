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


# Import unified demo runner
from unified_demo_runner import DemoRunner

console = Console()

# Import the centralized logger callback
from hrm_system import logger_callback

from utils.functions import prepare_data_config, prepare_run_config
from demo_model_runner import run_model_with_fallback
from demo_shared import get_best_trial_info

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

    training_config = TrainingConfig()

    try:
        data_config = prepare_data_config(dataset, smoke_test, num_aug)
        run_config = prepare_run_config(smoke_test, study_name)
    except Exception as e:
        return

    config = ExperimentConfig(
        mode="evaluate",
        run_config=run_config,
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

    try:
        data_config = prepare_data_config(dataset, smoke_test)
        run_config = prepare_run_config(smoke_test, study_name)
    except Exception as e:
        return

    config = ExperimentConfig(
        mode="optimize",
        run_config=run_config,
        data_config=data_config,
        optimization_config=OptimizationConfig(n_trials=n_trials, n_jobs=n_jobs, n_final_runs=n_final_runs, storage=storage, model_to_optimize=model_to_optimize, search_space=search_space)
    )
    run_optimization(config)
    console.print(f"[bold green]Optimization '{study_name}' finished.[/bold green]")

# Import shared function
from demo_shared import clear_optuna_studies

@cli.command()
@click.option("--dataset", default="synthetic", type=click.Choice(["arc", "sudoku", "maze", "synthetic"]), 
              help="Dataset to use.")
@click.option("--task", default="reverse", 
              type=click.Choice(["copy", "reverse", "sort", "parity", "duplicate"]), 
              help="Task for synthetic dataset.")
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
@click.option("--export-metrics", is_flag=True, default=False, help="Export detailed metrics to file.")
def demo(dataset, task, num_aug, n_trials, n_jobs, n_final_runs, smoke_test, study_name, patience, models, export_metrics):
    """Run an exciting, continuous side-by-side algorithm comparison with immediate animated results."""
    from demo_config import DemoConfig, DemoMode, DemoConfigManager
    
    console.clear()
    console.print(Panel("[bold blue]🚀 HRM vs HREM: Real-Time Algorithm Comparison[/bold blue]", expand=False))
    
    # Create demo configuration
    config = DemoConfigManager.create_config(
        mode=DemoMode.ADAPTIVE,  # Use adaptive mode for the demo
        models=list(models),
        dataset=dataset,
        task=task,
        smoke_test=smoke_test,
        export_metrics=export_metrics
    )
    
    # Update with command line options
    config.loop_control.max_trials = n_trials
    config.loop_control.n_jobs = n_jobs
    config.loop_control.n_final_runs = n_final_runs
    config.study_name = study_name
    config.patience_level = patience
    config.num_aug = num_aug
    
    # Import and run the unified demo runner
    from unified_demo_runner import run_demo
    run_demo(config)


@cli.command()
def demoui():
    """Launch the interactive demo UI with menu for choosing challenges."""
    from comprehensive_demo import interactive_config_setup
    from unified_demo_runner import run_demo
    
    console.clear()
    console.print(Panel("[bold blue]🚀 HRM vs HREM: Interactive Demo Selector[/bold blue]", expand=False))
    
    # Use the interactive configuration setup from comprehensive_demo
    config = interactive_config_setup()
    
    # Run the demo
    run_demo(config)


@cli.command()
@click.option("--challenge", type=str, help="Challenge ID to run")
@click.option("--patience", type=click.Choice(["low", "medium", "high"]), default="medium", 
              help="Patience level for the comparison")
@click.option("--smoke-test", is_flag=True, default=False, help="Run in smoke test mode")
def compare(challenge, patience, smoke_test):
    """Run scientific algorithm comparison between HRM and HREM."""
    console.print("[bold blue]🔬 Starting Scientific Algorithm Comparison[/bold blue]")
    
    # Import our scientific comparison components
    try:
        from scientific_comparison.model_runner import ScientificModelRunner
    except ImportError as e:
        console.print(f"[red]Error importing scientific comparison modules: {e}[/red]")
        console.print("[yellow]Make sure the scientific_comparison package is properly installed.[/yellow]")
        return
    
    try:
        runner = ScientificModelRunner()
        
        # If no challenge specified, use default
        if not challenge:
            challenge = "synthetic_sort"  # Default challenge
        
        # Run the comparison
        results = runner.run_comparison_from_config(
            challenge_id=challenge,
            patience_level=patience
        )
        
        console.print("[green]✅ Scientific comparison completed![/green]")
        
    except Exception as e:
        console.print(f"[red]Error running scientific comparison: {e}[/red]")
        import traceback
        console.print(traceback.format_exc())


@cli.command()
def tui():
    """Launch the Textual User Interface."""
    from tui.main import main as tui_main
    tui_main()

if __name__ == "__main__":
    cli()