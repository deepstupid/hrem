import click
from rich.console import Console
from typing import List

from hrm_system import (
    ExperimentConfig,
    RunConfig,
    DataConfig,
    ModelConfig,
    TrainingConfig,
    EvaluationConfig,
    OptimizationConfig,
    run_evaluation,
    run_optimization,
)
from demo_config import PatienceLevel
from run_demo_cli_parameterized import run_demonstration
from demo_models import list_available_models

console = Console()

def logger_callback(message: str):
    """A simple logger callback that prints to the console."""
    console.print(message)

@click.group()
def cli():
    """HRM System: A unified interface for evaluation, optimization, and demonstration."""
    pass

@cli.command()
@click.option("--dataset", default="synthetic", type=click.Choice(["arc", "sudoku", "maze", "synthetic"]), help="Dataset to use.")
@click.option("--num-aug", default=0, type=int, help="Number of augmentations.")
@click.option("--n-runs", default=1, type=int, help="Number of runs for statistical significance.")
@click.option("--smoke-test", is_flag=True, default=False, help="Run in smoke test mode.")
@click.option("--study-name", default="cli_evaluation", type=str, help="Name for the study.")
@click.option("--models", "-m", multiple=True, default=["HRM", "HREM"], help="Models to evaluate.")
def evaluate(dataset, num_aug, n_runs, smoke_test, study_name, models):
    """Run a side-by-side comparison of specified models."""
    console.print(f"[bold blue]Starting Evaluation: {study_name}[/bold blue]")

    model_configs = []
    available_models = {
        "HRM": ModelConfig(name="HRM", algorithm_class="hrm_system.algorithms.hrm.HRMAlgorithm", base_arch_config="hrm_v1"),
        "HREM": ModelConfig(name="HREM", algorithm_class="hrm_system.algorithms.hrem.HREMAlgorithm", base_arch_config="hrem_v1"),
        "EnhancedHREM": ModelConfig(name="EnhancedHREM", algorithm_class="hrm_system.algorithms.hrem.HREMAlgorithm", base_arch_config="enhanced_hrem_v1"),
    }
    for model_name in models:
        if model_name in available_models:
            model_configs.append(available_models[model_name])
        else:
            console.print(f"[yellow]Warning: Model '{model_name}' not found. Skipping.[/yellow]")

    if len(model_configs) < 1:
        console.print("[red]Error: No valid models specified for evaluation.[/red]")
        return

    eval_config_dict = {"n_runs": n_runs}
    for i, model_config in enumerate(model_configs):
        eval_config_dict[f"model_{chr(ord('a') + i)}"] = model_config

    eval_config = EvaluationConfig(**eval_config_dict)

    # Use enhanced training config if EnhancedHREM is being evaluated
    training_config = TrainingConfig()
    if "EnhancedHREM" in models:
        training_config = TrainingConfig(
            optimizer="AdamW",
            optimizer_eps=1e-5,
            lr=3e-4,
            puzzle_emb_lr=3e-3,
            weight_decay=0.01,
            global_batch_size=1024,
            eval_interval=5000,
        )

    config = ExperimentConfig(
        mode="evaluate",
        run_config=RunConfig(
            smoke_test=smoke_test,
            study_name=study_name,
            logger_callback=logger_callback
        ),
        data_config=DataConfig(
            dataset=dataset,
            num_aug=num_aug
        ),
        training_config=training_config,
        evaluation_config=eval_config,
    )

    run_evaluation(config)
    console.print(f"[bold green]Evaluation '{study_name}' finished.[/bold green]")


@cli.command()
@click.option("--dataset", default="synthetic", type=click.Choice(["arc", "sudoku", "maze", "synthetic"]), help="Dataset to use.")
@click.option("--n-trials", default=10, type=int, help="Number of optimization trials.")
@click.option("--n-jobs", default=1, type=int, help="Number of parallel jobs for Optuna.")
@click.option("--n-final-runs", default=1, type=int, help="Number of final comparison runs.")
@click.option("--storage", default="sqlite:///experiments/optuna_cli.db", help="Optuna storage URL.")
@click.option("--smoke-test", is_flag=True, default=False, help="Run in smoke test mode.")
@click.option("--study-name", default="cli_optimization", type=str, help="Name for the study.")
def optimize(dataset, n_trials, n_jobs, n_final_runs, storage, smoke_test, study_name):
    """Run hyperparameter optimization for the HREM model."""
    console.print(f"[bold blue]Starting Optimization: {study_name}[/bold blue]")

    config = ExperimentConfig(
        mode="optimize",
        run_config=RunConfig(
            smoke_test=smoke_test,
            study_name=study_name,
            logger_callback=logger_callback
        ),
        data_config=DataConfig(
            dataset=dataset
        ),
        optimization_config=OptimizationConfig(
            n_trials=n_trials,
            n_jobs=n_jobs,
            n_final_runs=n_final_runs,
            storage=storage
        )
    )

    run_optimization(config)
    console.print(f"[bold green]Optimization '{study_name}' finished.[/bold green]")

@cli.command()
@click.option("--patience", type=click.Choice(["low", "medium", "high"]), default="medium", help="Set the patience level for the demo.")
@click.option("--interactive", is_flag=True, help="Enable interactive mode with user prompts.")
@click.option("--challenge-key", type=str, help="Specify challenge key directly (skips menu).")
@click.option("--models", "-m", multiple=True, default=["HRM", "HREM"], help=f"Specify models to compare.")
@click.option("--storage-path", type=str, help="Custom storage path for optimization database.")
def demonstration(patience, interactive, challenge_key, models, storage_path):
    """Run an interactive demonstration with baseline, optimization, and final evaluation."""
    console.print(f"[bold blue]Starting Demonstration[/bold blue]")
    run_demonstration(
        patience=PatienceLevel(patience),
        interactive=interactive,
        challenge_key=challenge_key,
        model_names=models,
        storage_path=storage_path
    )
    console.print(f"[bold green]Demonstration finished.[/bold green]")


if __name__ == "__main__":
    cli()
