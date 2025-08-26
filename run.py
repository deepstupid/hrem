import os
import json
import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from rich.live import Live
import optuna
from typing import List, Dict, Any

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

console = Console()

def logger_callback(message: str):
    """A simple logger callback that prints to the console."""
    if ("it/s" not in message and "%" not in message and
        "TensorFloat32" not in message and "Online softmax" not in message and
        "torch._prims_common.check" not in message and
        "FutureWarning" not in message and "UserWarning" not in message):
        console.print(message)

@click.group()
def cli():
    """HRM System: A unified interface for evaluation, optimization, and demos."""
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
    if "EnhancedHREM" in models:
        training_config = TrainingConfig(optimizer="AdamW", optimizer_eps=1e-5, lr=3e-4, puzzle_emb_lr=3e-3, weight_decay=0.01, global_batch_size=1024, eval_interval=5000)

    config = ExperimentConfig(
        mode="evaluate",
        run_config=RunConfig(smoke_test=smoke_test, study_name=study_name, logger_callback=logger_callback),
        data_config=DataConfig(dataset=dataset, num_aug=num_aug),
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

    config = ExperimentConfig(
        mode="optimize",
        run_config=RunConfig(smoke_test=smoke_test, study_name=study_name, logger_callback=logger_callback),
        data_config=DataConfig(dataset=dataset),
        optimization_config=OptimizationConfig(n_trials=n_trials, n_jobs=n_jobs, n_final_runs=n_final_runs, storage=storage, model_to_optimize=model_to_optimize, search_space=search_space)
    )
    run_optimization(config)
    console.print(f"[bold green]Optimization '{study_name}' finished.[/bold green]")


def _run_trial(trial: optuna.trial.Trial, config: ExperimentConfig) -> float:
    """Execute a single trial for a given model."""
    opt_config = config.optimization_config
    model_config = opt_config.model_to_optimize
    search_space = opt_config.search_space or {}

    params = {}
    for param_name, definition in search_space.items():
        param_type = definition.get("type")
        if param_type == "categorical":
            params[param_name] = trial.suggest_categorical(param_name, definition["choices"])
        elif param_type == "int":
            params[param_name] = trial.suggest_int(param_name, definition["low"], definition["high"])

    trial_model_config = model_config.model_copy(deep=True)
    if "hrem" in trial_model_config.algorithm_class.lower():
        trial_model_config.hrem_params = HREMParams(**params)
    else:
        trial_model_config.arch_overrides = params

    try:
        metrics = run_single_model(
            run_config=config.run_config,
            data_config=config.data_config,
            model_config=trial_model_config,
            training_config=config.training_config,
            run_identifier=f"trial_{trial.number}"
        )
        return metrics.get('all/lm_loss', float('inf'))
    except Exception:
        raise optuna.TrialPruned()

def _get_best_trial_info(study: optuna.Study) -> Dict[str, Any]:
    """Safely retrieves information about the best trial from a study."""
    try:
        return {"number": study.best_trial.number, "params": study.best_trial.params, "value": study.best_trial.value}
    except ValueError:
        return None

@cli.command()
@click.option("--dataset", default="synthetic", type=click.Choice(["arc", "sudoku", "maze", "synthetic"]), help="Dataset to use.")
@click.option("--num-aug", default=0, type=int, help="Number of augmentations.")
@click.option("--n-trials", default=10, type=int, help="Number of optimization trials.")
@click.option("--n-jobs", default=1, type=int, help="Number of parallel jobs for Optuna.")
@click.option("--n-final-runs", default=1, type=int, help="Number of final comparison runs.")
@click.option("--smoke-test", is_flag=True, default=False, help="Run in smoke test mode.")
@click.option("--study-name", default="interactive_demo", type=str, help="Name for the study.")
@click.option(
    "--models",
    "-m",
    multiple=True,
    default=["HRM", "HREM"],
    type=click.Choice(list_available_models()),
    help="Models to evaluate.",
)
def demo(dataset, num_aug, n_trials, n_jobs, n_final_runs, smoke_test, study_name, models):
    """Run an interactive demo with baseline, optimization, and final evaluation."""
    console.clear()
    console.print(Panel("[bold blue]Welcome to the HRM/HREM Interactive Demo[/bold blue]", expand=False))

    # Baseline Evaluation
    console.print(Panel("[bold]Step 1: Baseline Evaluation[/bold]", expand=False))
    baseline_model_configs = get_model_configs(list(models))
    baseline_results = {}
    for model_config in baseline_model_configs:
        console.print(f"Running baseline for [cyan]{model_config.name}[/cyan]...")
        metrics = run_single_model(
            run_config=RunConfig(smoke_test=smoke_test, study_name=f"{study_name}_baseline", logger_callback=logger_callback),
            data_config=DataConfig(dataset=dataset, num_aug=num_aug),
            model_config=model_config,
            training_config=TrainingConfig(),
            run_identifier=f"baseline_{model_config.name}"
        )
        baseline_results[model_config.name] = metrics

    # Display baseline results
    display_final_comparison("Baseline Results", baseline_results)

    # Hyperparameter Optimization
    console.print(Panel("\n[bold]Step 2: Hyperparameter Optimization[/bold]", expand=False))
    optimized_results = {}
    for model_name in models:
        model_to_optimize = get_model_config(model_name)
        search_space = get_model_search_space(model_name)
        if not search_space:
            console.print(f"Skipping optimization for [cyan]{model_name}[/cyan]: No search space defined.")
            continue

        console.print(f"Optimizing [cyan]{model_name}[/cyan]...")

        storage_path = f"sqlite:///{os.path.abspath('experiments')}/{study_name}_{model_name}.db"
        study = optuna.create_study(study_name=f"{study_name}_{model_name}", storage=storage_path, direction="minimize", load_if_exists=True)

        config = ExperimentConfig(
            mode="optimize",
            run_config=RunConfig(smoke_test=smoke_test, study_name=f"{study_name}_{model_name}", logger_callback=logger_callback),
            data_config=DataConfig(dataset=dataset),
            optimization_config=OptimizationConfig(n_trials=n_trials, n_jobs=n_jobs, n_final_runs=n_final_runs, storage=storage_path, model_to_optimize=model_to_optimize, search_space=search_space)
        )

        study.optimize(lambda trial: _run_trial(trial, config), n_trials=n_trials, n_jobs=n_jobs)

        best_trial_info = _get_best_trial_info(study)
        if best_trial_info:
            optimized_results[model_name] = best_trial_info
            display_optimization_results(model_name, {"best_params": best_trial_info['params']})

    # Final Evaluation
    console.print(Panel("\n[bold]Step 3: Final Evaluation[/bold]", expand=False))
    final_model_configs = []
    for model_name in models:
        final_model_configs.append(get_model_config(model_name)) # baseline
        if model_name in optimized_results:
            best_params = optimized_results[model_name]['params']
            optimized_config = get_model_config(model_name)
            if "hrem" in optimized_config.algorithm_class.lower():
                optimized_config.hrem_params = HREMParams(**best_params)
            else:
                optimized_config.arch_overrides = best_params
            optimized_config.name = f"{model_name}_best"
            final_model_configs.append(optimized_config)

    eval_config_dict = {"n_runs": n_final_runs}
    for i, model_config in enumerate(final_model_configs):
        eval_config_dict[f"model_{chr(ord('a') + i)}"] = model_config
    eval_config = EvaluationConfig(**eval_config_dict)

    final_eval_config = ExperimentConfig(
        mode="evaluate",
        run_config=RunConfig(smoke_test=smoke_test, study_name=f"{study_name}_final_eval", logger_callback=logger_callback),
        data_config=DataConfig(dataset=dataset, num_aug=num_aug),
        training_config=TrainingConfig(),
        evaluation_config=eval_config,
    )
    final_results = run_evaluation(final_eval_config)

    # Display final results
    if final_results and "results" in final_results:
        display_final_comparison("Final Comparison", final_results["results"])

    console.print(Panel("[bold green]Demo Finished![/bold green]", expand=False))


if __name__ == "__main__":
    cli()
