#!/usr/bin/env python3
"""Comprehensive demo script showcasing all features of the HRM/HREM system.

This script provides a full-featured demo experience with:
- Detailed analysis and comparison
- Adaptive timing management
- Comprehensive instrumentation
- Interactive configuration options
- Extensive metrics collection and reporting

The script is designed for users who want a thorough understanding
of the HRM and HREM models and their performance characteristics.
"""

import click
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.table import Table
from demo_models import list_available_models
from demo_config import DemoConfig, DemoMode, DemoConfigManager
from unified_demo_runner import run_demo

console = Console()

def interactive_config_setup():
    """Interactively configure the demo based on user preferences."""
    console.print(Panel("[bold blue]🔧 Interactive Demo Configuration[/bold blue]", expand=False))
    
    # Get user preferences
    console.print("\n[bold]Let's configure your demo experience:[/bold]\n")
    
    # Demo mode
    demo_mode = Prompt.ask(
        "Select demo mode:",
        choices=["lightning", "adaptive", "comprehensive"],
        default="comprehensive"
    )
    
    # Dataset selection
    console.print("\n[bold]Dataset options:[/bold]")
    console.print("  1. Synthetic Copy")
    console.print("  2. Synthetic Reverse")
    console.print("  3. Synthetic Sort")
    console.print("  4. Synthetic Parity")
    console.print("  5. Synthetic Duplicate")
    console.print("  6. ARC Dataset")
    console.print("  7. Sudoku Dataset")
    console.print("  8. Maze Dataset")
    
    dataset_choice = Prompt.ask(
        "Select a dataset (enter number)",
        default="2"
    )
    
    dataset_map = {
        "1": ("synthetic", "copy"),
        "2": ("synthetic", "reverse"),
        "3": ("synthetic", "sort"),
        "4": ("synthetic", "parity"),
        "5": ("synthetic", "duplicate"),
        "6": ("arc", None),
        "7": ("sudoku", None),
        "8": ("maze", None)
    }
    
    dataset, task = dataset_map.get(dataset_choice, ("synthetic", "reverse"))
    
    # Models selection
    available_models = list_available_models()
    console.print(f"\n[bold]Available models:[/bold] {', '.join(available_models)}")
    
    models_input = Prompt.ask(
        "Select models to compare (comma-separated, or 'all')",
        default="HRM,HREM"
    )
    
    if models_input.lower() == "all":
        models = available_models
    else:
        models = [m.strip() for m in models_input.split(",") if m.strip() in available_models]
        if not models:
            models = ["HRM", "HREM"]
    
    # Smoke test
    smoke_test = Confirm.ask(
        "Run in smoke test mode? (faster but less accurate)",
        default=True
    )
    
    # Export metrics
    export_metrics = Confirm.ask(
        "Export detailed metrics to file?",
        default=False
    )
    
    # Create configuration
    config = DemoConfigManager.create_config(
        mode=DemoMode(demo_mode),
        models=models,
        dataset=dataset,
        task=task or "reverse",
        smoke_test=smoke_test,
        export_metrics=export_metrics
    )
    
    return config

@click.command()
@click.option("--interactive", "-i", is_flag=True, default=False, 
              help="Run in interactive configuration mode.")
@click.option("--dataset", default="synthetic", 
              type=click.Choice(["arc", "sudoku", "maze", "synthetic"]), 
              help="Dataset to use.")
@click.option("--task", default="reverse", 
              type=click.Choice(["copy", "reverse", "sort", "parity", "duplicate"]), 
              help="Task for synthetic dataset.")
@click.option("--smoke-test", is_flag=True, default=True, help="Run in smoke test mode.")
@click.option("--mode", type=click.Choice(["lightning", "adaptive", "comprehensive"]), default="comprehensive",
              help="Demo mode to run.")
@click.option("--models", "-m", multiple=True, default=["HRM", "HREM"], 
              type=click.Choice(list_available_models()), help="Models to evaluate.")
@click.option("--export-metrics", is_flag=True, default=False, help="Export detailed metrics to file.")
def comprehensive_demo(interactive, dataset, task, smoke_test, mode, models, export_metrics):
    """Run a comprehensive demo with full control over all parameters."""
    console.clear()
    console.print(Panel("[bold magenta]🔬 Comprehensive HRM vs HREM Demo[/bold magenta]", expand=False))
    
    # Get configuration
    if interactive:
        config = interactive_config_setup()
    else:
        # Create configuration from CLI options
        config = DemoConfigManager.create_config(
            mode=DemoMode(mode),
            models=list(models),
            dataset=dataset,
            task=task,
            smoke_test=smoke_test,
            export_metrics=export_metrics
        )
    
    # Display configuration
    console.print("\n[bold]Demo Configuration:[/bold]")
    config_table = Table(show_header=True, header_style="bold magenta")
    config_table.add_column("Parameter", style="cyan")
    config_table.add_column("Value")
    
    config_table.add_row("Demo Mode", config.demo_mode.value)
    config_table.add_row("Dataset", f"{config.dataset}-{config.task}" if config.dataset == "synthetic" else config.dataset)
    config_table.add_row("Models", ", ".join(config.models))
    config_table.add_row("Max Epochs", str(config.loop_control.max_epochs))
    config_table.add_row("Max Trials", str(config.loop_control.max_trials))
    config_table.add_row("Batch Size", str(config.loop_control.batch_size))
    config_table.add_row("Smoke Test", str(config.smoke_test))
    config_table.add_row("Export Metrics", str(config.instrumentation.export_metrics))
    
    console.print(config_table)
    
    if not interactive and not Confirm.ask("\nProceed with this configuration?", default=True):
        console.print("[yellow]Demo cancelled.[/yellow]")
        return
    
    # Run the demo
    run_demo(config)

if __name__ == "__main__":
    comprehensive_demo()