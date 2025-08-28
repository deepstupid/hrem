#!/usr/bin/env python3
"""Unified demo script for HRM vs HREM comparison with comprehensive control and adaptive instrumentation.

This script provides a single entry point for running demos with different modes:
- Lightning: Ultra-fast demo with minimal iterations
- Adaptive: Time-adaptive demo that adjusts based on available time
- Comprehensive: Full-featured demo with detailed analysis

The script automatically adapts to:
- Available time based on user patience levels
- Hardware capabilities
- Selected demo mode (lightning, adaptive, comprehensive)
- Model configurations and optimization parameters
"""

import os
import time
import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.prompt import Prompt, Confirm
from rich.table import Table

from hrm_system import (
    RunConfig,
    DataConfig,
    EvaluationConfig,
    OptimizationConfig,
    TrainingConfig
)
from hrm_system.config import ExperimentConfig
from hrm_system.config_demo import TrainingConfig as DemoTrainingConfig
from demo_models import get_model_configs, list_available_models
from hrm_system.reporting import display_final_comparison
from dataset_manager import dataset_manager
from demo_model_runner import get_dataset_config

# Import our unified demo components
from demo_enhanced_config import EnhancedDemoConfig, EnhancedConfigManager, DemoMode
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

def interactive_config_setup():
    """Interactively configure the demo based on user preferences.
    
    This function guides the user through configuring the demo experience
    by asking questions about their preferences and capabilities.
    
    Returns:
        EnhancedDemoConfig: Configured demo settings
    """
    console.print(Panel("[bold blue]🔧 Interactive Demo Configuration[/bold blue]", expand=False))
    
    # Get user preferences
    console.print("\n[bold]Let's configure your demo experience:[/bold]\n")
    
    # Patience level
    patience = Prompt.ask(
        "How patient are you today?",
        choices=["low", "medium", "high"],
        default="low"
    )
    
    # Hardware capability
    hardware = Prompt.ask(
        "What's your hardware capability?",
        choices=["low", "medium", "high"],
        default="low"
    )
    
    # Desired detail level
    detail = Prompt.ask(
        "How detailed would you like the analysis?",
        choices=["basic", "adaptive", "comprehensive"],
        default="adaptive"
    )
    
    # Demo mode
    demo_mode = Prompt.ask(
        "What demo mode would you prefer?",
        choices=["lightning", "adaptive", "comprehensive"],
        default="adaptive"
    )
    
    # Dataset selection
    console.print("\n[bold]Available datasets:[/bold]")
    for i, ds in enumerate(SYNTHETIC_DATASETS, 1):
        console.print(f"  {i}. {ds}")
    
    dataset_choice = Prompt.ask(
        "Select a dataset (enter number or name)",
        default="1"
    )
    
    if dataset_choice.isdigit():
        dataset_idx = int(dataset_choice) - 1
        if 0 <= dataset_idx < len(SYNTHETIC_DATASETS):
            dataset = SYNTHETIC_DATASETS[dataset_idx]
        else:
            dataset = SYNTHETIC_DATASETS[0]
    elif dataset_choice in SYNTHETIC_DATASETS:
        dataset = dataset_choice
    else:
        dataset = SYNTHETIC_DATASETS[0]
    
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
    
    # Export metrics
    export_metrics = Confirm.ask(
        "Export detailed metrics to file?",
        default=False
    )
    
    # Create adaptive configuration
    enhanced_config = EnhancedConfigManager.get_adaptive_config(
        user_patience=patience,
        hardware_capability=hardware,
        desired_detail=detail
    )
    
    # Update with user selections
    enhanced_config.demo_mode = DemoMode(demo_mode)
    enhanced_config.dataset = dataset
    enhanced_config.models = models
    enhanced_config.instrumentation.export_metrics = export_metrics
    enhanced_config.instrumentation.collect_detailed_metrics = (detail == "comprehensive")
    
    return enhanced_config

@click.command()
@click.option("--interactive", "-i", is_flag=True, default=False, 
              help="Run in interactive configuration mode.")
@click.option("--dataset", default="synthetic-reverse", 
              type=click.Choice(["arc", "sudoku", "maze", "synthetic"] + 
                               [f"synthetic-{task}" for task in ["copy", "reverse", "sort", "parity", "duplicate"]]), 
              help="Dataset to use.")
@click.option("--smoke-test", is_flag=True, default=True, help="Run in smoke test mode.")
@click.option("--study-name", default="unified_demo", type=str, help="Name for the study.")
@click.option("--patience", type=click.Choice(["low", "medium", "high"]), default="low", 
              help="User patience level.")
@click.option("--hardware", type=click.Choice(["low", "medium", "high"]), default="low",
              help="Hardware capability.")
@click.option("--detail", type=click.Choice(["basic", "adaptive", "comprehensive"]), default="adaptive",
              help="Detail level of analysis.")
@click.option("--demo-mode", type=click.Choice(["lightning", "adaptive", "comprehensive"]), default="adaptive",
              help="Demo mode to run.")
@click.option("--models", "-m", multiple=True, default=["HRM", "HREM"], 
              type=click.Choice(list_available_models()), help="Models to evaluate.")
@click.option("--export-metrics", is_flag=True, default=False, help="Export detailed metrics to file.")
@click.option("--max-epochs", default=1, type=int, help="Max epochs for training.")
@click.option("--max-trials", default=1, type=int, help="Max trials for optimization.")
def unified_demo(interactive, dataset, smoke_test, study_name, patience, hardware, detail, demo_mode, models, export_metrics, max_epochs, max_trials):
    """Run a unified demo with adaptive instrumentation and comprehensive control."""
    from demo_config import DemoConfig
    
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