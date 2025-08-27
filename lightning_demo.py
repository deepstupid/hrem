#!/usr/bin/env python3
"""Lightning-fast demo script for HRM vs HREM comparison.

This script provides an ultra-fast demo experience with:
- Minimal iterations for quick results
- Real-time progress visualization
- Efficient optimization algorithms
- Immediate feedback and results

The script is designed for users who want to quickly compare HRM and HREM models
without waiting for extensive optimization processes.
"""

import click
from rich.console import Console
from rich.panel import Panel
from demo_models import list_available_models
from demo_config import DemoConfig, DemoMode, DemoConfigManager
from unified_demo_runner import run_demo

console = Console()

@click.command()
@click.option("--dataset", default="synthetic", 
              type=click.Choice(["arc", "sudoku", "maze", "synthetic"]), 
              help="Dataset to use.")
@click.option("--task", default="reverse", 
              type=click.Choice(["copy", "reverse", "sort", "parity", "duplicate"]), 
              help="Task for synthetic dataset.")
@click.option("--smoke-test", is_flag=True, default=True, help="Run in smoke test mode.")
@click.option("--models", "-m", multiple=True, default=["HRM", "HREM"], 
              type=click.Choice(list_available_models()), help="Models to evaluate.")
@click.option("--export-metrics", is_flag=True, default=False, help="Export detailed metrics to file.")
def lightning_demo(dataset, task, smoke_test, models, export_metrics):
    """Run a lightning-fast demo.
    
    This function orchestrates the lightning-fast demo process using the unified demo runner.
    """
    # Create lightning demo configuration
    config = DemoConfigManager.create_config(
        mode=DemoMode.LIGHTNING,
        models=list(models),
        dataset=dataset,
        task=task,
        smoke_test=smoke_test,
        export_metrics=export_metrics
    )
    
    # Run the demo
    run_demo(config)

if __name__ == "__main__":
    lightning_demo()