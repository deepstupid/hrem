import click
from rich.console import Console

from hrm_system import (
    ExperimentConfig,
    RunConfig,
    DataConfig,
    ModelConfig,
    TrainingConfig,
    EvaluationConfig,
    OptimizationConfig,
    HREMParams,
    run_evaluation,
    run_optimization,
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
def evaluate(dataset, num_aug, n_runs, smoke_test, study_name):
    """Run a side-by-side comparison of the HRM and HREM models."""
    console.print(f"[bold blue]Starting Evaluation: {study_name}[/bold blue]")

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
        evaluation_config=EvaluationConfig(
            n_runs=n_runs
        )
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


if __name__ == "__main__":
    cli()
