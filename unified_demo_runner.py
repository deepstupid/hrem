"""Unified demo runner for the HRM/HREM demo system."""

import time
import optuna
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from demo_config_manager import DemoConfig, ConfigManager, model_registry
from demo_timing_utils import TimingManager
from demo_ui import ResultsDisplay

console = Console()

def run_model_with_fallback(model_config, run_config, data_config, training_config, run_identifier):
    """Run a model with fallback to synthetic dataset if needed."""
    from hrm_system import run_single_model
    try:
        metrics = run_single_model(
            run_config=run_config,
            data_config=data_config,
            model_config=model_config,
            training_config=training_config,
            run_identifier=run_identifier
        )
        return metrics
    except Exception as e:
        if "not found" in str(e).lower() or "no such file" in str(e).lower():
            console.print(f"[yellow]⚠️  Dataset issue for {model_config.name}. Falling back to synthetic.[/yellow]")
            synthetic_data_config = ConfigManager.get_dataset_config("synthetic", run_config.smoke_test, data_config.num_aug)
            metrics = run_single_model(
                run_config=run_config,
                data_config=synthetic_data_config,
                model_config=model_config,
                training_config=training_config,
                run_identifier=run_identifier
            )
            return metrics
        else:
            raise

def run_trial_with_timing(trial: optuna.trial.Trial, config, timing_manager=None, operation_name=None):
    """Execute a single trial for a given model with optional timing."""
    from hrm_system.config import HREMParams
    start_time = time.time()

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
        metrics = run_model_with_fallback(
            trial_model_config,
            config.run_config,
            config.data_config,
            config.training_config,
            f"trial_{trial.number}"
        )
        elapsed_time = time.time() - start_time
        if timing_manager and operation_name:
            timing_manager.record_timing(operation_name, elapsed_time)
        return metrics.get('all/lm_loss', float('inf'))
    except Exception:
        raise optuna.TrialPruned()

def run_demo(config: DemoConfig):
    """Run a unified demo with adaptive instrumentation and comprehensive control."""
    console.print(Panel(f"[bold blue]🚀 Starting {config.demo_mode.value.capitalize()} Demo[/bold blue]", expand=False))

    timing_manager = TimingManager()
    display = ResultsDisplay(ui_config={})

    with timing_manager.get_context("total_demo_time"):
        # Create experiment config
        experiment_config = ConfigManager.create_experiment_config(config)

        # Run baseline evaluation
        baseline_results = {}
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("[cyan]Running baseline evaluation...", total=len(experiment_config.evaluation_config.models))
            for model_config in experiment_config.evaluation_config.models:
                with timing_manager.get_context(f"baseline_{model_config.name}"):
                    metrics = run_model_with_fallback(
                        model_config,
                        experiment_config.run_config,
                        experiment_config.data_config,
                        experiment_config.training_config,
                        f"baseline_{model_config.name}"
                    )
                    baseline_results[model_config.name] = metrics
                progress.update(task, advance=1)

        display.display_model_detailed_stats("Baseline Evaluation", baseline_results)

        # Run optimization if applicable
        optimized_params = None
        if experiment_config.mode == "optimize":
            opt_config = experiment_config.optimization_config
            model_to_optimize = opt_config.model_to_optimize

            console.print(Panel(f"[bold blue]🔬 Optimizing {model_to_optimize.name}[/bold blue]", expand=False))

            study = optuna.create_study(direction="minimize", study_name=experiment_config.run_config.study_name)
            with timing_manager.get_context(f"optimization_{model_to_optimize.name}"):
                study.optimize(
                    lambda trial: run_trial_with_timing(trial, experiment_config, timing_manager, f"trial_{model_to_optimize.name}"),
                    n_trials=opt_config.n_trials,
                    n_jobs=opt_config.n_jobs,
                )

            optimized_params = study.best_params
            console.print(f"\n[green]Best trial for {model_to_optimize.name}:[/green]")
            console.print(f"  Params: {study.best_params}")
            console.print(f"  Value: {study.best_value}")

        # Run final comparison
        final_results = {}
        final_models = experiment_config.evaluation_config.models
        if optimized_params:
            # Update the model config with the optimized params
            for model_config in final_models:
                if model_config.name == model_to_optimize.name:
                    if "hrem" in model_config.algorithm_class.lower():
                        model_config.hrem_params = HREMParams(**optimized_params)
                    else:
                        model_config.arch_overrides = optimized_params

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("[cyan]Running final comparison...", total=len(final_models))
            for model_config in final_models:
                with timing_manager.get_context(f"final_{model_config.name}"):
                    metrics = run_model_with_fallback(
                        model_config,
                        experiment_config.run_config,
                        experiment_config.data_config,
                        experiment_config.training_config,
                        f"final_{model_config.name}"
                    )
                    final_results[model_config.name] = metrics
                progress.update(task, advance=1)

        display.display_final_comparison("Final Comparison", final_results)
        display.display_final_leader(final_results)

    console.print(Panel(f"[bold green]✅ Demo Complete![/bold green]", expand=False))
    timing_manager.display_summary()
