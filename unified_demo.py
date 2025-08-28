#!/usr/bin/env python3
"""Unified demo script for HRM vs HREM comparison with comprehensive control and adaptive instrumentation."""

import click
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm

from demo_config_manager import DemoConfig, ConfigManager, model_registry, DemoMode
from unified_demo_runner import run_demo

console = Console()

# Available synthetic datasets
SYNTHETIC_DATASETS = [
    "synthetic-copy",
    "synthetic-reverse",
    "synthetic-sort",
    "synthetic-parity",
    "synthetic-duplicate"
]

def interactive_config_setup() -> DemoConfig:
    """Interactively configure the demo based on user preferences."""
    console.print(Panel("[bold blue]🔧 Interactive Demo Configuration[/bold blue]", expand=False))
    
    patience = Prompt.ask("How patient are you today?", choices=["low", "medium", "high"], default="low")
    hardware = Prompt.ask("What's your hardware capability?", choices=["low", "medium", "high"], default="low")
    detail = Prompt.ask("How detailed would you like the analysis?", choices=["basic", "adaptive", "comprehensive"], default="adaptive")
    demo_mode_str = Prompt.ask("What demo mode would you prefer?", choices=[e.value for e in DemoMode], default="adaptive")
    demo_mode = DemoMode(demo_mode_str)

    console.print("\n[bold]Available datasets:[/bold]")
    for i, ds in enumerate(SYNTHETIC_DATASETS, 1):
        console.print(f"  {i}. {ds}")
    dataset_choice = Prompt.ask("Select a dataset (enter number or name)", default="1")

    if dataset_choice.isdigit() and 0 <= int(dataset_choice) - 1 < len(SYNTHETIC_DATASETS):
        dataset = SYNTHETIC_DATASETS[int(dataset_choice) - 1]
    elif dataset_choice in SYNTHETIC_DATASETS:
        dataset = dataset_choice
    else:
        dataset = SYNTHETIC_DATASETS[0]

    available_models = model_registry.list_models()
    console.print(f"\n[bold]Available models:[/bold] {', '.join(available_models)}")
    models_input = Prompt.ask("Select models to compare (comma-separated, or 'all')", default="HRM,HREM")

    if models_input.lower() == "all":
        models = available_models
    else:
        models = [m.strip() for m in models_input.split(',') if m.strip() in available_models]
        if not models:
            models = ["HRM", "HREM"]

    export_metrics = Confirm.ask("Export detailed metrics to file?", default=False)

    config = ConfigManager.get_adaptive_config(user_patience=patience, hardware_capability=hardware, desired_detail=detail)
    config.demo_mode = demo_mode
    config.dataset = dataset
    config.models = models
    config.instrumentation.export_metrics = export_metrics
    config.instrumentation.collect_detailed_metrics = (detail == "comprehensive")
    
    return config

@click.command()
@click.option("--interactive", "-i", is_flag=True, default=False, help="Run in interactive configuration mode.")
@click.option("--dataset", default="synthetic-reverse", help="Dataset to use.")
@click.option("--smoke-test", is_flag=True, default=True, help="Run in smoke test mode.")
@click.option("--study-name", default="unified_demo", type=str, help="Name for the study.")
@click.option("--patience", type=click.Choice(["low", "medium", "high"]), default="low", help="User patience level.")
@click.option("--hardware", type=click.Choice(["low", "medium", "high"]), default="low", help="Hardware capability.")
@click.option("--detail", type=click.Choice(["basic", "adaptive", "comprehensive"]), default="adaptive", help="Detail level of analysis.")
@click.option("--demo-mode", type=click.Choice([e.value for e in DemoMode]), default="adaptive", help="Demo mode to run.")
@click.option("--models", "-m", multiple=True, default=["HRM", "HREM"], help="Models to evaluate.")
@click.option("--export-metrics", is_flag=True, default=False, help="Export detailed metrics to file.")
@click.option("--max-epochs", default=1, type=int, help="Max epochs for training.")
@click.option("--max-trials", default=1, type=int, help="Max trials for optimization.")
def unified_demo(interactive, dataset, smoke_test, study_name, patience, hardware, detail, demo_mode, models, export_metrics, max_epochs, max_trials):
    """Run a unified demo with adaptive instrumentation and comprehensive control."""
    if interactive:
        config = interactive_config_setup()
    else:
        task = ""
        if "synthetic" in dataset:
            parts = dataset.split("-", 1)
            if len(parts) > 1:
                task = parts[1]
                dataset = parts[0]

        config = DemoConfig(
            demo_mode=DemoMode(demo_mode),
            models=list(models),
            dataset=dataset,
            task=task,
            patience_level=patience,
            study_name=study_name,
            smoke_test=smoke_test,
        )
        config.loop_control.max_epochs = max_epochs
        config.loop_control.max_trials = max_trials
        config.instrumentation.export_metrics = export_metrics

    run_demo(config)

if __name__ == "__main__":
    unified_demo()