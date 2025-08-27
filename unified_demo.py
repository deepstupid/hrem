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
from unified_demo_runner import UnifiedDemoRunner

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
def unified_demo(interactive, dataset, smoke_test, study_name, patience, hardware, detail, demo_mode, models, export_metrics):
    """Run a unified demo with adaptive instrumentation and comprehensive control.
    
    This function orchestrates the entire demo process:
    1. Configuration setup (interactive or CLI)
    2. Experiment configuration creation
    3. Demo runner initialization
    4. Baseline evaluation
    5. Hyperparameter optimization (if applicable)
    6. Final evaluation and comparison
    7. Results display and metrics export
    
    Args:
        interactive: Whether to run in interactive mode
        dataset: Dataset to use for evaluation
        smoke_test: Whether to run in smoke test mode
        study_name: Name for the study
        patience: User patience level (low/medium/high)
        hardware: Hardware capability (low/medium/high)
        detail: Detail level of analysis (basic/adaptive/comprehensive)
        demo_mode: Demo mode to run (lightning/adaptive/comprehensive)
        models: Models to evaluate
        export_metrics: Whether to export detailed metrics to file
    """
    console.clear()
    console.print(Panel("[bold magenta]🔬 Unified HRM vs HREM Demo[/bold magenta]", expand=False))
    
    # Get configuration
    if interactive:
        enhanced_config = interactive_config_setup()
    else:
        # Create enhanced demo configuration from CLI options
        enhanced_config = EnhancedConfigManager.get_adaptive_config(
            user_patience=patience,
            hardware_capability=hardware,
            desired_detail=detail
        )
        enhanced_config.demo_mode = DemoMode(demo_mode)
        enhanced_config.dataset = dataset
        enhanced_config.smoke_test = smoke_test
        enhanced_config.study_name = study_name
        enhanced_config.models = list(models)
        enhanced_config.instrumentation.export_metrics = export_metrics
        enhanced_config.instrumentation.collect_detailed_metrics = (detail == "comprehensive")
    
    # Display configuration
    console.print("\n[bold]Demo Configuration:[/bold]")
    config_table = Table(show_header=True, header_style="bold magenta")
    config_table.add_column("Parameter", style="cyan")
    config_table.add_column("Value")
    
    config_table.add_row("Demo Mode", enhanced_config.demo_mode.value)
    config_table.add_row("Patience Level", enhanced_config.patience_level)
    config_table.add_row("Dataset", enhanced_config.dataset)
    config_table.add_row("Models", ", ".join(enhanced_config.models))
    config_table.add_row("Max Epochs", str(enhanced_config.loop_control.max_epochs))
    config_table.add_row("Max Trials", str(enhanced_config.loop_control.max_trials))
    config_table.add_row("Batch Size", str(enhanced_config.loop_control.batch_size))
    config_table.add_row("Export Metrics", str(enhanced_config.instrumentation.export_metrics))
    
    console.print(config_table)
    
    if not Confirm.ask("\nProceed with this configuration?", default=True):
        console.print("[yellow]Demo cancelled.[/yellow]")
        return
    
    # Create experiment configuration using enhanced config manager
    experiment_config = EnhancedConfigManager.create_experiment_config(enhanced_config)
    
    # Initialize unified demo runner
    demo_runner = UnifiedDemoRunner(experiment_config, enhanced_config)
    
    # Show demo intro
    mode_desc = {
        "lightning": "⚡ Lightning-Fast",
        "adaptive": "🧠 Adaptive",
        "comprehensive": "🔬 Comprehensive"
    }
    mode_text = mode_desc.get(enhanced_config.demo_mode.value, "🔬 Unified")
    console.print(f"\n[bold green]{mode_text} Demo Starting...[/bold green]")
    console.print("[italic]Sit back and watch the magic happen![/italic]\n")
    
    # --- STEP 1: Baseline Evaluation ---
    console.print(Panel("[bold]📊 Step 1: Baseline Evaluation[/bold]", expand=False))
    
    baseline_results = demo_runner.run_baseline_evaluation(enhanced_config.models)
    
    # Display baseline results
    display_final_comparison("📊 Baseline Results", baseline_results)
    console.print("[bold green]✅ Baseline evaluation completed![/bold green]\n")
    
    # Check if we should continue with optimization
    if not demo_runner.should_continue_demo():
        console.print("[yellow]⚠️  Demo time limit reached. Skipping optimization phase.[/yellow]")
        demo_runner.display_timing_summary()
        if export_metrics:
            demo_runner.export_detailed_metrics("unified_demo_metrics.json")
        return
    
    # --- STEP 2: Hyperparameter Optimization ---
    console.print(Panel("[bold]🔍 Step 2: Hyperparameter Optimization[/bold]", expand=False))
    
    optimized_results = demo_runner.run_hyperparameter_optimization(enhanced_config.models)
    
    # Display optimization results
    if optimized_results:
        console.print("[bold]Optimization Results:[/bold]")
        for model_name, best_info in optimized_results.items():
            console.print(f"  {model_name}: Best Loss = {best_info['value']:.4f}")
    else:
        console.print("[yellow]No optimization results available.[/yellow]")
    
    console.print("[bold green]✅ Hyperparameter optimization completed![/bold green]\n")
    
    # Check if we should continue with final evaluation
    if not demo_runner.should_continue_demo():
        console.print("[yellow]⚠️  Demo time limit reached. Skipping final evaluation.[/yellow]")
        demo_runner.display_timing_summary()
        if export_metrics:
            demo_runner.export_detailed_metrics("unified_demo_metrics.json")
        return
    
    # --- STEP 3: Final Comparison ---
    console.print(Panel("[bold]🏆 Step 3: Final Comparison[/bold]", expand=False))
    
    final_results = demo_runner.run_final_evaluation(baseline_results, optimized_results, enhanced_config.models)
    
    # Display final results
    console.print("\n[bold magenta]🎨 Final Results:[/bold magenta]")
    display_final_comparison("🏆 Final Comparison", final_results)
    
    # Show exciting conclusion
    demo_runner.display_timing_summary()
    if export_metrics:
        demo_runner.export_detailed_metrics("unified_demo_metrics.json")
    
    console.print(Panel(
        f"[bold magenta]🎉 Unified Demo Finished Successfully![/bold magenta]\n"
        f"[italic]Thanks for exploring HRM vs HREM with us![/italic]\n"
        f"[bold blue]💡 Key insight: Unified approach provides the best of all worlds![/bold blue]",
        expand=False
    ))

if __name__ == "__main__":
    unified_demo()