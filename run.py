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


console = Console()

# Import the centralized logger callback
from hrm_system import logger_callback

# Import shared function
from demo_model_runner import get_dataset_config

# Import shared function
from demo_model_runner import run_model_with_fallback

# Import shared function for trial info
from demo_shared import get_best_trial_info


@click.group()
def cli():
    """HRM System: A unified interface for evaluation, optimization, and scientific comparison."""
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
            challenge = "quick_comparison"  # Default challenge
        
        # Run the comparison
        results = runner.run_comparison_from_config(
            challenge_id=challenge,
            patience_level=patience,
            smoke_test=smoke_test
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