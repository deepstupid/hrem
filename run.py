import os
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
from demo_models import (
    list_available_models,
    get_model_configs,
    get_model_config,
    get_model_search_space,
)

console = Console()

def logger_callback(message: str):
    """A simple logger callback that prints to the console."""
    console.print(message)

@click.group()
def cli():
    """HRM System: A unified interface for evaluation and optimization."""
    pass

@cli.command()
@click.option("--dataset", default="synthetic", type=click.Choice(["arc", "sudoku", "maze", "synthetic"]), help="Dataset to use.")
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
def evaluate(dataset, num_aug, n_runs, smoke_test, study_name, models):
    """Run a side-by-side comparison of specified models."""
    console.print(f"[bold blue]Starting Evaluation: {study_name}[/bold blue]")

    model_configs = get_model_configs(list(models))

    if not model_configs:
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
@click.option(
    "--storage",
    default=f"sqlite:///{os.path.abspath('experiments')}/optuna_cli.db",
    help="Optuna storage URL.",
)
@click.option("--smoke-test", is_flag=True, default=False, help="Run in smoke test mode.")
@click.option("--study-name", default="cli_optimization", type=str, help="Name for the study.")
@click.option(
    "--model",
    default="HREM",
    type=click.Choice(list_available_models()),
    help="Model to optimize.",
)
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
            storage=storage,
            model_to_optimize=model_to_optimize,
            search_space=search_space,
        )
    )

    run_optimization(config)
    console.print(f"[bold green]Optimization '{study_name}' finished.[/bold green]")


if __name__ == "__main__":
    cli()
