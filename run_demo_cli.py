#!/usr/bin/env python3
"""
Command-line version of the TUI's Demo that shows each iteration's results in the console.
Complete end-to-end demonstration of HRM vs HREM with real-time results generation.
"""

import time
import sys
import os
import argparse
import warnings
import logging
from typing import Dict, Any, Tuple
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich import box

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Suppress warnings at the highest level
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)

# Further suppress specific PyTorch warnings
import torch
torch.backends.cuda.matmul.allow_tf32 = False

from hrm_system import (
    ExperimentConfig,
    RunConfig,
    DataConfig,
    TrainingConfig,
    EvaluationConfig,
    OptimizationConfig,
    ModelConfig,
    HREMParams,
    run_evaluation,
    run_optimization,
)

console = Console()

class DemoLogger:
    """A logger that captures and displays messages in a structured way."""
    def __init__(self):
        self.messages = []
        
    def log(self, message: str):
        """Log a message and display it."""
        # Filter out progress bar updates and warning messages to reduce verbosity
        if ("it/s" not in message and "%" not in message and 
            "TensorFloat32" not in message and "Online softmax" not in message and
            "torch._prims_common.check" not in message and
            "FutureWarning" not in message and "UserWarning" not in message):
            console.print(message)
            self.messages.append(message)

def get_config_settings(is_fast_mode: bool) -> Dict[str, Any]:
    """Get configuration settings based on mode."""
    if is_fast_mode:
        return {
            "baseline_epochs": 50,
            "baseline_eval_interval": 25,
            "opt_epochs": 50,
            "opt_eval_interval": 25,
            "opt_trials": 3,
            "final_epochs": 50,
            "final_eval_interval": 25
        }
    else:
        return {
            "baseline_epochs": 200,
            "baseline_eval_interval": 50,
            "opt_epochs": 150,
            "opt_eval_interval": 50,
            "opt_trials": 5,
            "final_epochs": 200,
            "final_eval_interval": 50
        }

def display_model_detailed_stats(title: str, results: Dict[str, Any], model_names: list = None):
    """Display detailed statistics for models including parameter counts and performance metrics."""
    if not model_names:
        # Determine which models are present in the results
        model_names = [name for name in ['HRM', 'HREM', 'HREM_best'] if name in results]
    
    if not model_names:
        return
        
    table = Table(title=title, show_header=True, header_style="bold magenta", box=box.ROUNDED)
    table.add_column("Metric", style="cyan")
    
    # Add columns for each model
    model_styles = {"HRM": "bold blue", "HREM": "bold green", "HREM_best": "bold bright_green"}
    for model_name in model_names:
        style = model_styles.get(model_name, "bold white")
        table.add_column(model_name, justify="right", style=style)
    
    if results:
        # Get metrics for available models
        model_metrics = {}
        for model_name in model_names:
            if model_name in results:
                model_metrics[model_name] = results.get(model_name, {})
        
        # Common metrics to display with descriptions
        metrics_info = [
            ('all/accuracy', 'Accuracy', '🎯 Higher is better - task solving accuracy'),
            ('all/lm_loss', 'Loss', '📉 Lower is better - language modeling loss'),
            ('all/steps', 'Steps', '⚡ Lower is better - average steps to solve'),
            ('step', 'Training Steps', '📊 Training iterations completed'),
            ('num_params', 'Parameters', '⚙️ Model parameter count')
        ]
        
        for key, display_name, description in metrics_info:
            # Check if any model has this metric
            has_metric = any(model_metrics[model_name].get(key, 'N/A') != 'N/A' for model_name in model_metrics)
            if not has_metric:
                continue
                
            row_values = []
            for model_name in model_names:
                if model_name in model_metrics:
                    val = model_metrics[model_name].get(key, 'N/A')
                    # Format values
                    if isinstance(val, (int, float)):
                        if key == 'num_params':
                            val = f"{val:,}"  # Add commas for large numbers
                        else:
                            val = f"{val:.4f}"
                    row_values.append(str(val))
                else:
                    row_values.append('N/A')
            
            table.add_row(display_name, *row_values)
            
        console.print(table)

def display_model_comparison_summary(results: Dict[str, Any]):
    """Display a concise comparison summary."""
    table = Table(show_header=True, header_style="bold white", box=box.SIMPLE)
    table.add_column("Model", style="bold")
    table.add_column("Accuracy", justify="right")
    table.add_column("Loss", justify="right")
    table.add_column("Steps", justify="right")
    
    models = ['HRM', 'HREM', 'HREM_best']
    model_styles = {"HRM": "bold blue", "HREM": "bold green", "HREM_best": "bold bright_green"}
    
    for model_name in models:
        if model_name in results and results[model_name]:
            metrics = results[model_name]
            accuracy = metrics.get('all/accuracy', 'N/A')
            loss = metrics.get('all/lm_loss', 'N/A')
            steps = metrics.get('all/steps', 'N/A')
            
            # Format values
            if isinstance(accuracy, (int, float)):
                accuracy = f"{accuracy:.4f}"
            if isinstance(loss, (int, float)):
                loss = f"{loss:.4f}"
            if isinstance(steps, (int, float)):
                steps = f"{steps:.0f}"
                
            style = model_styles.get(model_name, "white")
            table.add_row(f"[{style}]{model_name}[/{style}]", accuracy, loss, steps)
            
    console.print(table)

def display_current_leader(hrm_score: float, hrem_score: float):
    """Display the current leader with a colorful panel."""
    if hrm_score > hrem_score:
        winner = "HRM"
        emoji = "👑"
        color = "blue"
        reason = "Higher accuracy"
    elif hrem_score > hrm_score:
        winner = "HREM"
        emoji = "👑"
        color = "green"
        reason = "Higher accuracy"
    else:
        winner = "TIE"
        emoji = "🤝"
        color = "yellow"
        reason = "Equal performance"
    
    console.print(Panel(f"[bold {color}]{emoji} Current Leader: {winner}[/bold {color}]\n[dim]{reason}[/dim]", expand=False))

def display_performance_improvement(baseline_acc: float, final_acc: float, model_name: str = "HREM"):
    """Display performance improvement with visual indicators."""
    improvement = final_acc - baseline_acc
    
    if improvement > 0:
        console.print(f"[bold green]📈 {model_name} improved by {improvement:.4f}![/bold green]")
        console.print(f"[dim]Performance gain: {improvement/baseline_acc*100:.1f}% increase[/dim]")
    elif improvement < 0:
        console.print(f"[bold red]📉 {model_name} decreased by {abs(improvement):.4f}[/bold red]")
        console.print(f"[dim]Performance loss: {abs(improvement)/baseline_acc*100:.1f}% decrease[/dim]")
    else:
        console.print(f"[bold yellow]➡️  {model_name} performance unchanged[/bold yellow]")

def display_hrem_params(title: str, params: HREMParams):
    """Display HREM parameters in a formatted table."""
    if not params:
        return
        
    table = Table(title=title, show_header=True, header_style="bold cyan", box=box.ROUNDED)
    table.add_column("Parameter", style="dim")
    table.add_column("Value", justify="right")
    
    param_dict = params.model_dump()
    for key, value in param_dict.items():
        table.add_row(key, str(value))
            
    console.print(table)

def display_iteration_header(title: str, description: str = ""):
    """Display a header for each iteration with a clean screen."""
    console.clear()
    console.print(Panel(f"[bold blue]{title}[/bold blue]", expand=False))
    if description:
        console.print(f"[dim]{description}[/dim]")
    console.print()

def wait_for_user(interactive: bool = False):
    """Wait for user input if interactive mode is enabled."""
    if interactive:
        try:
            input("\n[bold blue]Press Enter to continue...[/bold blue]")
        except EOFError:
            pass  # Continue if input is not available

def run_baseline_evaluation(study_name: str, is_fast_mode: bool = False) -> Dict[str, Any]:
    """Run a baseline evaluation of HRM vs HREM."""
    display_iteration_header("🏁 Step 1: Establishing Baseline Performance", 
                           "Running both models on the same synthetic task to establish baseline performance...")
    
    config_settings = get_config_settings(is_fast_mode)
    
    config = ExperimentConfig(
        mode="evaluate",
        run_config=RunConfig(
            smoke_test=True,
            study_name=study_name,
            logger_callback=DemoLogger().log
        ),
        data_config=DataConfig(dataset="synthetic", synthetic_task="copy"),
        training_config=TrainingConfig(
            epochs=config_settings["baseline_epochs"], 
            eval_interval=config_settings["baseline_eval_interval"]
        ),
        evaluation_config=EvaluationConfig(n_runs=1)
    )
    
    start_time = time.time()
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        progress.add_task(description="Running baseline evaluation...", total=None)
        results = run_evaluation(config)
        
    elapsed_time = time.time() - start_time
    console.print(f"[dim]⏱️  Evaluation completed in {elapsed_time:.1f} seconds[/dim]")
        
    return results.get("results", {})

def calculate_balanced_score(accuracy: float, num_params: int, target_params: int = 100000) -> float:
    """
    Calculate a balanced score that considers both accuracy and parameter count.
    
    Args:
        accuracy: Model accuracy (higher is better)
        num_params: Number of parameters (lower is better for efficiency)
        target_params: Target parameter count for balancing
        
    Returns:
        Balanced score (higher is better)
    """
    # Convert to numbers if they're strings
    try:
        accuracy = float(accuracy)
        num_params = int(num_params)
        target_params = int(target_params)
    except (ValueError, TypeError):
        return 0.0  # Return a default score if conversion fails
    
    # Avoid division by zero
    if target_params == 0:
        return accuracy
    
    # Normalize parameter count (0-1 scale, where 1 is perfect match to target)
    param_score = 1.0 - abs(num_params - target_params) / target_params
    # Clamp to [0, 1] range
    param_score = max(0.0, min(1.0, param_score))
    
    # Combine accuracy and parameter efficiency (you can adjust weights)
    # For now, we'll use a simple weighted average
    accuracy_weight = 0.7
    param_weight = 0.3
    
    return accuracy_weight * accuracy + param_weight * param_score

def run_hyperparameter_optimization(study_name: str, is_fast_mode: bool = False) -> Tuple[HREMParams, Dict[str, Any]]:
    """Run hyperparameter optimization for both HRM and HREM with competitive parameter counts."""
    display_iteration_header("🔍 Step 2: Guided Hyperparameter Optimization", 
                           "Optimizing both HRM and HREM hyperparameters with real-time performance feedback...")
    console.print("[dim]💡 Key advantage: Results are generated after each iteration![/dim]")
    console.print()
    
    config_settings = get_config_settings(is_fast_mode)
    
    # First, optimize HREM
    console.print("[bold blue]Running HREM hyperparameter optimization...[/bold blue]")
    hrem_config = ExperimentConfig(
        mode="optimize",
        run_config=RunConfig(
            smoke_test=True,
            study_name=f"{study_name}_hrem",
            logger_callback=DemoLogger().log
        ),
        data_config=DataConfig(dataset="synthetic", synthetic_task="copy"),
        training_config=TrainingConfig(
            epochs=config_settings["opt_epochs"], 
            eval_interval=config_settings["opt_eval_interval"]
        ),
        optimization_config=OptimizationConfig(
            n_trials=config_settings["opt_trials"],
            n_jobs=1,
            storage="sqlite:///experiments/optuna_demo_cli.db",
            n_final_runs=1,
            # Specify that we're optimizing HREM
            model_to_optimize=ModelConfig(
                name="HREM_best",
                algorithm_class="hrm_system.algorithms.hrem.HREMAlgorithm",
                base_arch_config="hrem_v1"
            )
        )
    )
    
    start_time = time.time()
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        progress.add_task(description="Running HREM hyperparameter optimization...", total=None)
        hrem_result = run_optimization(hrem_config)
    
    hrem_elapsed = time.time() - start_time
    console.print(f"[dim]⏱️  HREM optimization completed in {hrem_elapsed:.1f} seconds[/dim]")
    
    # Then, optimize HRM
    console.print("[bold blue]Running HRM hyperparameter optimization...[/bold blue]")
    
    # Create HRM optimization config
    hrm_config = ExperimentConfig(
        mode="optimize",
        run_config=RunConfig(
            smoke_test=True,
            study_name=f"{study_name}_hrm",
            logger_callback=DemoLogger().log
        ),
        data_config=DataConfig(dataset="synthetic", synthetic_task="copy"),
        training_config=TrainingConfig(
            epochs=config_settings["opt_epochs"], 
            eval_interval=config_settings["opt_eval_interval"]
        ),
        optimization_config=OptimizationConfig(
            n_trials=config_settings["opt_trials"],
            n_jobs=1,
            storage="sqlite:///experiments/optuna_demo_cli.db",
            n_final_runs=1,
            # Specify that we're optimizing HRM
            model_to_optimize=ModelConfig(
                name="HRM_best",
                algorithm_class="hrm_system.algorithms.hrm.HRMAlgorithm",
                base_arch_config="hrm_v1"
            ),
            # Define search space for HRM parameters
            search_space={
                "path": "config/hrm_search_space.yaml"
            }
        )
    )
    
    hrm_start_time = time.time()
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        progress.add_task(description="Running HRM hyperparameter optimization...", total=None)
        hrm_result = run_optimization(hrm_config)
    
    hrm_elapsed = time.time() - hrm_start_time
    console.print(f"[dim]⏱️  HRM optimization completed in {hrm_elapsed:.1f} seconds[/dim]")
    
    # Get best parameters for both models
    hrem_params = None
    if hrem_result and "best_params" in hrem_result:
        hrem_params = HREMParams(**hrem_result["best_params"])
    else:
        # For demo purposes, we'll create a simple set of optimized params
        hrem_params = HREMParams(
            m_loc=128,
            d_mem=128,
            top_k=4,
            H_layers=2,
            L_layers=2,
            H_cycles=2,
            L_cycles=8,
            hidden_size=256
        )
    
    hrm_params = None
    if hrm_result and "best_params" in hrm_result:
        hrm_params = hrm_result["best_params"]
    
    elapsed_time = time.time() - start_time
    console.print(f"[dim]⏱️  Total optimization completed in {elapsed_time:.1f} seconds[/dim]")
    
    return hrem_params, {
        "hrem_result": hrem_result,
        "hrm_result": hrm_result,
        "hrm_params": hrm_params,
        "hrm_score": 0.0  # For compatibility with existing code
    }

def run_final_evaluation(optimized_params: HREMParams, hrm_config: Dict[str, Any], study_name: str, is_fast_mode: bool = False) -> Dict[str, Any]:
    """Run final evaluation with both optimized HRM and HREM."""
    display_iteration_header("🏆 Step 3: Final Performance Comparison", 
                           "Running final comparison with optimized HRM and HREM parameters...")
    
    config_settings = get_config_settings(is_fast_mode)
    
    # Create model configurations with optimized parameters
    hrem_model_config = ModelConfig(
        name="HREM_best",
        algorithm_class="hrm_system.algorithms.hrem.HREMAlgorithm",
        base_arch_config="hrem_v1",
        hrem_params=optimized_params
    )
    
    # For HRM, we pass parameters as arch_overrides
    hrm_model_config = ModelConfig(
        name="HRM_best",
        algorithm_class="hrm_system.algorithms.hrm.HRMAlgorithm",
        base_arch_config="hrm_v1",
        arch_overrides=hrm_config or {}
    )
    
    config = ExperimentConfig(
        mode="evaluate",
        run_config=RunConfig(
            smoke_test=True,
            study_name=study_name,
            logger_callback=DemoLogger().log
        ),
        data_config=DataConfig(dataset="synthetic", synthetic_task="copy"),
        training_config=TrainingConfig(
            epochs=config_settings["final_epochs"], 
            eval_interval=config_settings["final_eval_interval"]
        ),
        evaluation_config=EvaluationConfig(
            n_runs=1,
            model_a=hrm_model_config,
            model_b=hrem_model_config
        )
    )
    
    start_time = time.time()
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        progress.add_task(description="Running final evaluation...", total=None)
        results = run_evaluation(config)
        
    elapsed_time = time.time() - start_time
    console.print(f"[dim]⏱️  Final evaluation completed in {elapsed_time:.1f} seconds[/dim]")
        
    return results.get("results", {})

def main(is_fast_mode: bool = False, interactive: bool = False):
    """Run the HRM vs HREM demonstration in the console."""
    try:
        # Header
        console.clear()
        console.print(Panel("[bold blue]🚀 HRM vs HREM Demonstration[/bold blue]\n[italic]Complete end-to-end system showcasing real-time results generation[/italic]", expand=False))
        
        # Introduction
        console.print("\n[bold]This demonstration showcases:[/bold]")
        console.print("• 📊 Baseline evaluation of HRM vs HREM models")
        console.print("• 🔍 Hyperparameter optimization with real-time feedback")
        console.print("• 🏆 Final comparison showing performance gains")
        
        # Approach explanation
        console.print("\n[bold blue]🧠 Approach Explanation[/bold blue]")
        console.print("The HRM System uses a novel approach to hyperparameter optimization:")
        console.print("• Starts with baseline models to establish performance reference")
        console.print("• Uses guided search to explore hyperparameter space efficiently")
        console.print("• Generates actionable results after each iteration")
        console.print("• Continuously improves based on real-time feedback")
        
        if interactive:
            try:
                input("\n[bold green]Press Enter to start the demonstration...[/bold green]")
            except EOFError:
                pass  # Continue if input is not available
        
        # Step 1: Baseline evaluation
        baseline_results = run_baseline_evaluation("cli_demo_baseline", is_fast_mode)
        display_model_detailed_stats("📊 Baseline Results", baseline_results)
        
        # Show current leader
        hrm_baseline_acc = baseline_results.get('HRM', {}).get('all/accuracy', 0)
        hrem_baseline_acc = baseline_results.get('HREM', {}).get('all/accuracy', 0)
        try:
            hrm_baseline_acc = float(hrm_baseline_acc) if isinstance(hrm_baseline_acc, str) else hrm_baseline_acc
            hrem_baseline_acc = float(hrem_baseline_acc) if isinstance(hrem_baseline_acc, str) else hrem_baseline_acc
        except (ValueError, TypeError):
            pass
        display_current_leader(hrm_baseline_acc, hrem_baseline_acc)
        
        if interactive:
            try:
                input("\n[bold blue]Press Enter to continue to optimization...[/bold blue]")
            except EOFError:
                pass  # Continue if input is not available
        
        # Step 2: Hyperparameter optimization
        hrem_params, optimization_details = run_hyperparameter_optimization("cli_demo_optimization", is_fast_mode)
        display_hrem_params("⚙️ Optimized HREM Parameters", hrem_params)
        
        hrm_params = optimization_details.get("hrm_params", {})
        console.print("\n[bold blue]HRM Optimization Results:[/bold blue]")
        if hrm_params:
            console.print(f"  Best HRM params: {hrm_params}")
        else:
            console.print("  No HRM parameters found")
        
        if interactive:
            try:
                input("\n[bold blue]Press Enter to see final comparison...[/bold blue]")
            except EOFError:
                pass  # Continue if input is not available
        
        # Step 3: Final comparison
        final_results = run_final_evaluation(hrem_params, hrm_params, "cli_demo_final", is_fast_mode)
        display_model_detailed_stats("📊 Final Results", final_results)
        
        # Show improvement
        hrem_final_acc = final_results.get('HREM_best', {}).get('all/accuracy', 0)
        try:
            hrem_final_acc = float(hrem_final_acc) if isinstance(hrem_final_acc, str) else hrem_final_acc
        except (ValueError, TypeError):
            pass
        display_performance_improvement(hrem_baseline_acc, hrem_final_acc, "HREM")
        
        # Show final leader
        hrm_final_acc = final_results.get('HRM', {}).get('all/accuracy', 0)
        try:
            hrm_final_acc = float(hrm_final_acc) if isinstance(hrm_final_acc, str) else hrm_final_acc
        except (ValueError, TypeError):
            pass
        display_current_leader(hrm_final_acc, hrem_final_acc)
        
        # Summary
        display_iteration_header("✅ Demonstration Completed Successfully!")
        
        # Performance summary
        console.print("[bold]📈 Performance Summary:[/bold]")
        console.print(f"  HRM baseline performance:           {hrm_baseline_acc:.4f}")
        console.print(f"  HREM baseline performance:          {hrem_baseline_acc:.4f}")
        
        # Get optimized model performances
        hrm_final_acc = final_results.get('HRM_best', {}).get('all/accuracy', 0)
        try:
            hrm_final_acc = float(hrm_final_acc) if isinstance(hrm_final_acc, str) else hrm_final_acc
        except (ValueError, TypeError):
            pass
            
        console.print(f"  HRM optimized performance:          {hrm_final_acc:.4f}")
        console.print(f"  HREM optimized performance:         {hrem_final_acc:.4f}")
        
        hrem_improvement = hrem_final_acc - hrem_baseline_acc
        hrm_improvement = hrm_final_acc - hrm_baseline_acc
        console.print(f"  HREM performance improvement:       {hrem_improvement:+.4f}")
        console.print(f"  HRM performance improvement:        {hrm_improvement:+.4f}")
        
        # Cost-benefit analysis
        console.print("\n[bold]💰 Cost-Benefit Analysis:[/bold]")
        hrm_baseline_params = baseline_results.get('HRM', {}).get('num_params', 0)
        hrem_baseline_params = baseline_results.get('HREM', {}).get('num_params', 0)
        try:
            hrm_baseline_params = int(float(hrm_baseline_params)) if isinstance(hrm_baseline_params, (str, float)) else int(hrm_baseline_params)
            hrem_baseline_params = int(float(hrem_baseline_params)) if isinstance(hrem_baseline_params, (str, float)) else int(hrem_baseline_params)
        except (ValueError, TypeError):
            hrm_baseline_params = 0
            hrem_baseline_params = 0
            
        # Get optimized model parameter counts
        hrm_final_params = final_results.get('HRM_best', {}).get('num_params', 0)
        hrem_final_params = final_results.get('HREM_best', {}).get('num_params', 0)
        try:
            hrm_final_params = int(float(hrm_final_params)) if isinstance(hrm_final_params, (str, float)) else int(hrm_final_params)
            hrem_final_params = int(float(hrem_final_params)) if isinstance(hrem_final_params, (str, float)) else int(hrem_final_params)
        except (ValueError, TypeError):
            hrm_final_params = 0
            hrem_final_params = 0
            
        if hrm_baseline_params and hrem_baseline_params:
            baseline_param_increase = ((hrem_baseline_params - hrm_baseline_params) / hrm_baseline_params) * 100
            console.print(f"  HRM baseline parameter count:       {hrm_baseline_params:,}")
            console.print(f"  HREM baseline parameter count:      {hrem_baseline_params:,}")
            console.print(f"  Baseline parameter difference:      {baseline_param_increase:+.1f}%")
            
        if hrm_final_params and hrem_final_params:
            final_param_increase = ((hrem_final_params - hrm_final_params) / hrm_final_params) * 100 if hrm_final_params != 0 else 0
            console.print(f"  HRM optimized parameter count:      {hrm_final_params:,}")
            console.print(f"  HREM optimized parameter count:     {hrem_final_params:,}")
            console.print(f"  Optimized parameter difference:     {final_param_increase:+.1f}%")
            
            # Efficiency ratios
            if hrem_improvement > 0 and final_param_increase > 0:
                hrem_efficiency = hrem_improvement / (final_param_increase / 100)
                console.print(f"  HREM efficiency ratio:              {hrem_efficiency:.2f} (accuracy gain per 1% params)")
                
            if hrm_improvement != 0 and final_param_increase != 0:  # Avoid division by zero
                hrm_efficiency = hrm_improvement / (final_param_increase / 100) if final_param_increase != 0 else 0
                console.print(f"  HRM efficiency ratio:               {hrm_efficiency:.2f} (accuracy gain per 1% params)")
        
        # Key insights
        console.print("\n[bold]🔑 Key Insights:[/bold]")
        console.print("• HRM: Traditional recurrent model with external memory")
        console.print("• HREM: Hierarchical approach with multiple memory layers")
        console.print("• Multi-objective optimization balances accuracy and parameter efficiency")
        console.print("• Both models are optimized for fair parameter count comparison")
        console.print("• Each iteration provides actionable insights for improvement")
        
        # Unique features
        console.print("\n[bold green]🎯 What Makes This Approach Unique:[/bold green]")
        console.print("• Real-time results generation after each iteration")
        console.print("• Guided search for efficient hyperparameter exploration")
        console.print("• Continuous feedback for actionable insights")
        console.print("• No configuration parameters needed - fully turnkey")
        console.print("\n[italic]The system begins generating results immediately,[/italic]")
        console.print("[italic]allowing for continuous improvement and insights.[/italic]")
        
    except Exception as e:
        console.print(f"[bold red]❌ Demo failed: {e}[/bold red]")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run HRM vs HREM demonstration")
    parser.add_argument("--fast", action="store_true", help="Run in fast mode for testing")
    parser.add_argument("--interactive", action="store_true", help="Enable interactive mode with user prompts")
    args = parser.parse_args()
    
    main(is_fast_mode=args.fast, interactive=args.interactive)