"""Adaptive demo execution with timing and optimization for the HRM/HREM demo system."""

import optuna
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from demo_config_manager import DemoConfig, ConfigManager
from demo_metrics import MetricsManager
from demo_ui import ResultsDisplay
from demo_model_runner import run_model_with_fallback, run_trial_with_timing

console = Console()

def run_adaptive_demo(config: DemoConfig):
    """Run a unified demo with adaptive instrumentation and comprehensive control."""
    console.print(Panel(f"[bold blue]🚀 Starting {config.demo_mode.value.capitalize()} Demo[/bold blue]", expand=False))

    metrics_manager = MetricsManager()
    display = ResultsDisplay(ui_config={})

    with metrics_manager.get_context("total_demo_time"):
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
                with metrics_manager.get_context(f"baseline_{model_config.name}"):
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
            with metrics_manager.get_context(f"optimization_{model_to_optimize.name}"):
                study.optimize(
                    lambda trial: run_trial_with_timing(trial, experiment_config, metrics_manager, f"trial_{model_to_optimize.name}"),
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
            from hrm_system.config import HREMParams
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
                with metrics_manager.get_context(f"final_{model_config.name}"):
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
    metrics_manager.display_summary()
