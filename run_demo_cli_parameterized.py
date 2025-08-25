#!/usr/bin/env python3
"""
Parameterized command-line version of the TUI's Demo that shows each iteration's results in the console.
Complete end-to-end demonstration of HRM vs HREM with real-time results generation.
Parameterized by challenge and set of algorithms to optimize.
"""

import time
import sys
import os
import argparse
import warnings
import logging
from typing import Dict, Any, Tuple, List, Optional
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich import box
from rich.prompt import Prompt

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

# Import the challenge system
from challenges import (
    Challenge,
    ChallengeDifficulty,
    get_all_challenges_sorted,
    get_challenge_by_name
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

class DemoConfiguration:
    """Configuration class for the demo parameters."""
    
    def __init__(self, is_fast_mode: bool = False, interactive: bool = False):
        self.is_fast_mode = is_fast_mode
        self.interactive = interactive
        self.config_settings = self._get_config_settings()
        
    def _get_config_settings(self) -> Dict[str, Any]:
        """Get configuration settings based on mode."""
        if self.is_fast_mode:
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
                "opt_trials": 10,
                "final_epochs": 400,
                "final_eval_interval": 50
            }

class ChallengeSelector:
    """Handles challenge selection and display."""
    
    @staticmethod
    def display_challenge_menu() -> Challenge:
        """Display a menu for selecting a challenge and return the selected challenge."""
        console.clear()
        console.print(Panel("[bold blue]🎯 HRM vs HREM Challenge Selection[/bold blue]", expand=False))
        
        challenges = get_all_challenges_sorted()
        
        # Group challenges by difficulty
        difficulty_order = {
            ChallengeDifficulty.BEGINNER: "🌱 Beginner",
            ChallengeDifficulty.INTERMEDIATE: "🌿 Intermediate",
            ChallengeDifficulty.ADVANCED: "🔥 Advanced",
            ChallengeDifficulty.RESEARCH: "🔬 Research"
        }
        
        # Display challenges grouped by difficulty
        for difficulty in [ChallengeDifficulty.BEGINNER, ChallengeDifficulty.INTERMEDIATE, 
                          ChallengeDifficulty.ADVANCED, ChallengeDifficulty.RESEARCH]:
            difficulty_challenges = [c for c in challenges if c.difficulty == difficulty]
            if difficulty_challenges:
                console.print(f"\n[bold]{difficulty_order[difficulty]} Challenges:[/bold]")
                for i, challenge in enumerate(difficulty_challenges, 1):
                    # Find the global index
                    global_index = challenges.index(challenge) + 1
                    console.print(f"  {global_index:2d}. [cyan]{challenge.name}[/cyan]")
                    console.print(f"      [dim]{challenge.description}[/dim]")
                    console.print(f"      🖥️  {challenge.recommended_hardware} | ⏱️  {challenge.expected_duration}")
        
        # Get user selection
        while True:
            try:
                choice = Prompt.ask("\n[bold green]Select a challenge[/bold green] (enter number or name)")
                
                # Try to parse as number first
                if choice.isdigit():
                    index = int(choice) - 1
                    if 0 <= index < len(challenges):
                        return challenges[index]
                
                # Try to match by name
                for challenge in challenges:
                    if challenge.name.lower() == choice.lower():
                        return challenge
                        
                console.print("[red]Invalid selection. Please try again.[/red]")
            except KeyboardInterrupt:
                console.print("\n[yellow]Exiting...[/yellow]")
                sys.exit(0)

class ResultsDisplay:
    """Handles displaying results in various formats."""
    
    @staticmethod
    def _create_results_table(title: str, model_names: List[str], model_styles: Dict[str, str]) -> Table:
        """Create a standardized results table with proper styling."""
        table = Table(title=title, show_header=True, header_style="bold magenta", box=box.ROUNDED)
        table.add_column("Metric", style="cyan")
        
        # Add columns for each model
        for model_name in model_names:
            style = model_styles.get(model_name, "bold white")
            table.add_column(model_name, justify="right", style=style)
        
        return table
    
    @staticmethod
    def _format_metric_value(val, key):
        """Format metric values for display."""
        if isinstance(val, (int, float)):
            if key == 'num_params':
                return f"{val:,}"  # Add commas for large numbers
            else:
                return f"{val:.4f}"
        return str(val)
    
    @staticmethod
    def display_model_detailed_stats(title: str, results: Dict[str, Any], model_names: list = None):
        """Display detailed statistics for models including parameter counts and performance metrics."""
        if not model_names:
            # Determine which models are present in the results
            model_names = list(results.keys())
        
        if not model_names:
            return
            
        # Create table with consistent styling
        model_styles = {"HRM": "bold blue", "HREM": "bold green", "HREM_best": "bold bright_green", "HRM_best": "bold blue"}
        table = ResultsDisplay._create_results_table(title, model_names, model_styles)
        
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
                        row_values.append(ResultsDisplay._format_metric_value(val, key))
                    else:
                        row_values.append('N/A')
                
                table.add_row(display_name, *row_values)
                
            console.print(table)
    
    @staticmethod
    def display_final_comparison(title: str, final_results: Dict[str, Any]):
        """Display a clear comparison of final results for all models."""
        console.print(Panel(f"[bold]{title}[/bold]", expand=False))
        
        # Get all model names from final results
        model_names = list(final_results.keys())
        
        if not model_names:
            console.print("[dim]No models to compare[/dim]")
            return
        
        # Create comparison table
        model_styles = {"HRM": "blue", "HREM": "green", "EnhancedHREM": "yellow", "HREM_best": "bright_green", "HRM_best": "blue"}
        table = ResultsDisplay._create_results_table("", sorted(model_names), model_styles)
        
        # Add columns for each model with custom display names
        table.columns[0].header = "Metric"  # Reset first column header
        
        # Update column headers with styled names
        for i, model_name in enumerate(sorted(model_names), 1):  # Start from 1 because first column is Metric
            base_style = model_styles.get(model_name, "white")
            # Remove '_best' suffix for cleaner display
            display_name = model_name.replace('_best', '') + ' (Optimized)'
            table.columns[i]._header = display_name
            table.columns[i].style = base_style
            table.columns[i].justify = "right"
        
        # Key metrics for comparison
        metrics_info = [
            ('all/accuracy', 'Accuracy'),
            ('all/lm_loss', 'Loss'),
            ('all/steps', 'Steps'),
            ('num_params', 'Parameters')
        ]
        
        for key, display_name in metrics_info:
            row_values = []
            for model_name in sorted(model_names):
                val = final_results.get(model_name, {}).get(key, 'N/A')
                row_values.append(ResultsDisplay._format_value(val, key))
            
            table.add_row(display_name, *row_values)
        
        console.print(table)
    
    @staticmethod
    def _format_value(val, key):
        """Helper method to format values for display."""
        if isinstance(val, (int, float)):
            if key == 'num_params':
                return f"{val:,}"
            else:
                return f"{val:.4f}"
        return str(val)
    
    @staticmethod
    def display_current_leader(results: Dict[str, Any]):
        """Display the current leader with a colorful panel based on accuracy."""
        if not results:
            return
            
        # Find the model with the highest accuracy
        best_model = None
        best_accuracy = -1
        for model_name, metrics in results.items():
            accuracy = metrics.get('all/accuracy', 0)
            try:
                accuracy = float(accuracy) if isinstance(accuracy, str) else accuracy
            except (ValueError, TypeError):
                continue
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model_name
        
        if best_model:
            console.print(f"\n[bold green]👑 Current Leader: {best_model}[/bold green]")
            console.print(f"[dim]Accuracy: {best_accuracy:.4f}[/dim]")
    
    @staticmethod
    def display_final_leader(final_results: Dict[str, Any]):
        """Display the final leader based on accuracy."""
        if not final_results:
            return
            
        # Find the model with the highest accuracy
        best_model_final = None
        best_accuracy_final = -1
        for model_name, metrics in final_results.items():
            accuracy = metrics.get('all/accuracy', 0)
            try:
                accuracy = float(accuracy) if isinstance(accuracy, str) else accuracy
            except (ValueError, TypeError):
                continue
            if accuracy > best_accuracy_final:
                best_accuracy_final = accuracy
                best_model_final = model_name
        
        if best_model_final:
            console.print(f"\n[bold green]🏆 Final Leader: {best_model_final}[/bold green]")
            console.print(f"[dim]Accuracy: {best_accuracy_final:.4f}[/dim]")
    
    # Removed display_performance_improvements method (no longer needed)
    # Removed display_performance_summary method (no longer needed)
    
    # Removed cost_benefit_analysis method
    
    @staticmethod
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
    
    @staticmethod
    def display_iteration_header(title: str, description: str = ""):
        """Display a header for each iteration with a clean screen."""
        console.clear()
        console.print(Panel(f"[bold blue]{title}[/bold blue]", expand=False))
        if description:
            console.print(f"[dim]{description}[/dim]")
        console.print()

class ExperimentRunner:
    """Handles running the different phases of the experiment."""
    
    def __init__(self, config: DemoConfiguration):
        self.config = config
        self.logger = DemoLogger()
    
    def _handle_dataset_error(self, e: Exception, data_config: DataConfig):
        """Handle dataset-related errors more gracefully."""
        if "No such file or directory" in str(e) and "raw-data" in str(e):
            console.print(f"[bold red]❌ Dataset not found![/bold red]")
            console.print(f"[yellow]The {data_config.dataset} dataset requires raw data files that are not included in this repository.[/yellow]")
            console.print("[dim]Please download the required dataset files or try a different challenge.[/dim]")
            console.print("[dim]For ARC challenges, see the README for dataset preparation instructions.[/dim]")
            raise SystemExit(1)
        else:
            # Re-raise other exceptions
            raise e
    
    # Removed run_baseline_evaluation method (no longer needed)
    
    def run_hyperparameter_optimization_for_model(self, model_config: ModelConfig, study_name: str, data_config: DataConfig) -> Dict[str, Any]:
        """Run hyperparameter optimization for a single model."""
        config_settings = self.config.config_settings
        
        # Create optimization config based on model type
        if "hrem" in model_config.algorithm_class.lower():
            # HREM optimization config
            opt_config = OptimizationConfig(
                n_trials=config_settings["opt_trials"],
                n_jobs=1,
                storage="sqlite:///experiments/optuna_demo_cli.db",
                n_final_runs=1,
                model_to_optimize=ModelConfig(
                    name=f"{model_config.name}_best",
                    algorithm_class=model_config.algorithm_class,
                    base_arch_config=model_config.base_arch_config
                )
            )
        else:
            # HRM or other model optimization config
            opt_config = OptimizationConfig(
                n_trials=config_settings["opt_trials"],
                n_jobs=1,
                storage="sqlite:///experiments/optuna_demo_cli.db",
                n_final_runs=1,
                model_to_optimize=ModelConfig(
                    name=f"{model_config.name}_best",
                    algorithm_class=model_config.algorithm_class,
                    base_arch_config=model_config.base_arch_config
                ),
                # Define search space for HRM parameters if needed
                search_space={
                    "path": "config/hrm_search_space.yaml"
                }
            )
        
        config = ExperimentConfig(
            mode="optimize",
            run_config=RunConfig(
                smoke_test=True,
                study_name=f"{study_name}_{model_config.name.lower()}",
                logger_callback=self.logger.log
            ),
            data_config=data_config,
            training_config=TrainingConfig(
                epochs=config_settings["opt_epochs"], 
                eval_interval=config_settings["opt_eval_interval"]
            ),
            optimization_config=opt_config
        )
        
        start_time = time.time()
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            progress.add_task(description=f"Running {model_config.name} hyperparameter optimization...", total=None)
            try:
                result = run_optimization(config)
            except Exception as e:
                self._handle_dataset_error(e, data_config)
        
        elapsed_time = time.time() - start_time
        console.print(f"[dim]⏱️  {model_config.name} optimization completed in {elapsed_time:.1f} seconds[/dim]")
        
        return result
    
    def run_hyperparameter_optimization(self, study_name: str, data_config: DataConfig, model_configs: List[ModelConfig] = None) -> Dict[str, Any]:
        """Run hyperparameter optimization for all provided models."""
        ResultsDisplay.display_iteration_header("🔍 Step 2: Guided Hyperparameter Optimization", 
                               f"Optimizing hyperparameters for all models on {data_config.dataset} dataset with real-time performance feedback...")
        console.print("[dim]💡 Key advantage: Results are generated after each iteration![/dim]")
        console.print()
        
        # If no specific models provided, use default HRM and HREM
        if model_configs is None:
            model_configs = [
                ModelConfig(name="HRM", algorithm_class="hrm_system.algorithms.hrm.HRMAlgorithm", base_arch_config="hrm_v1"),
                ModelConfig(name="HREM", algorithm_class="hrm_system.algorithms.hrem.HREMAlgorithm", base_arch_config="hrem_v1")
            ]
        
        # Run optimization for each model
        optimization_results = {}
        for model_config in model_configs:
            console.print(f"[bold blue]Running {model_config.name} hyperparameter optimization...[/bold blue]")
            result = self.run_hyperparameter_optimization_for_model(model_config, study_name, data_config)
            optimization_results[model_config.name] = result
        
        return optimization_results
    
    def run_final_evaluation(self, optimized_results: Dict[str, Any], baseline_model_configs: List[ModelConfig], study_name: str, data_config: DataConfig) -> Dict[str, Any]:
        """Run final evaluation with all optimized models."""
        ResultsDisplay.display_iteration_header("🏆 Step 3: Final Performance Comparison", 
                               f"Running final comparison with optimized parameters on {data_config.dataset} dataset...")
        
        config_settings = self.config.config_settings
        
        # Create model configurations with optimized parameters
        final_model_configs = []
        
        # Add optimized versions of all models that were optimized
        for model_name, opt_result in optimized_results.items():
            if opt_result and "best_params" in opt_result:
                # Find the corresponding baseline model config
                baseline_config = None
                for config in baseline_model_configs:
                    if config.name == model_name:
                        baseline_config = config
                        break
                
                if baseline_config:
                    if "hrem" in baseline_config.algorithm_class.lower():
                        # For HREM models, pass parameters as hrem_params
                        hrem_params = HREMParams(**opt_result["best_params"])
                        optimized_config = ModelConfig(
                            name=f"{model_name}_best",
                            algorithm_class=baseline_config.algorithm_class,
                            base_arch_config=baseline_config.base_arch_config,
                            hrem_params=hrem_params
                        )
                    else:
                        # For other models, pass parameters as arch_overrides
                        optimized_config = ModelConfig(
                            name=f"{model_name}_best",
                            algorithm_class=baseline_config.algorithm_class,
                            base_arch_config=baseline_config.base_arch_config,
                            arch_overrides=opt_result["best_params"] or {}
                        )
                    final_model_configs.append(optimized_config)
        
        # Also add the baseline models for comparison (like EnhancedHREM)
        for config in baseline_model_configs:
            # Only add baseline models that weren't optimized
            if config.name not in optimized_results:
                final_model_configs.append(config)
        
        if not final_model_configs:
            console.print("[bold yellow]No models to evaluate in final comparison.[/bold yellow]")
            return {}
        
        # Create evaluation config with all final models
        eval_config_dict = {
            "n_runs": 1
        }
        
        # Add models to evaluation config (up to 5 models supported directly)
        model_keys = ["model_a", "model_b", "model_c", "model_d", "model_e"]
        for i, model_config in enumerate(final_model_configs):
            if i < len(model_keys):
                eval_config_dict[model_keys[i]] = model_config
            else:
                break  # Only support up to 5 models directly
        
        config = ExperimentConfig(
            mode="evaluate",
            run_config=RunConfig(
                smoke_test=True,
                study_name=study_name,
                logger_callback=self.logger.log
            ),
            data_config=data_config,
            training_config=TrainingConfig(
                epochs=config_settings["final_epochs"], 
                eval_interval=config_settings["final_eval_interval"]
            ),
            evaluation_config=EvaluationConfig(**eval_config_dict)
        )
        
        start_time = time.time()
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            progress.add_task(description="Running final evaluation...", total=None)
            try:
                results = run_evaluation(config)
            except Exception as e:
                self._handle_dataset_error(e, data_config)
                
        elapsed_time = time.time() - start_time
        console.print(f"[dim]⏱️  Final evaluation completed in {elapsed_time:.1f} seconds[/dim]")
            
        return results.get("results", {})

def wait_for_user(interactive: bool = False):
    """Wait for user input if interactive mode is enabled."""
    if interactive:
        try:
            input("\n[bold blue]Press Enter to continue...[/bold blue]")
        except EOFError:
            pass  # Continue if input is not available

def get_model_configs(model_names: List[str]) -> List[ModelConfig]:
    """Get model configurations for the specified model names."""
    model_config_map = {
        "HRM": ModelConfig(name="HRM", algorithm_class="hrm_system.algorithms.hrm.HRMAlgorithm", base_arch_config="hrm_v1"),
        "HREM": ModelConfig(name="HREM", algorithm_class="hrm_system.algorithms.hrem.HREMAlgorithm", base_arch_config="hrem_v1"),
        "EnhancedHREM": ModelConfig(name="EnhancedHREM", algorithm_class="hrm_system.algorithms.enhanced_hrem.EnhancedHREMAlgorithm", base_arch_config="enhanced_hrem_v1")
    }
    
    configs = []
    for name in model_names:
        if name in model_config_map:
            configs.append(model_config_map[name])
        else:
            console.print(f"[bold yellow]Warning: Unknown model '{name}', skipping...[/bold yellow]")
    
    return configs

def _display_optimization_results(model_name: str, opt_result: Dict[str, Any]):
    """Display optimization results for a single model."""
    if opt_result and "best_params" in opt_result:
        console.print(f"\n[bold blue]{model_name} Optimization Results:[/bold blue]")
        # Display parameters in a table
        params_table = Table(show_header=True, header_style="bold cyan", box=box.ROUNDED)
        params_table.add_column("Parameter", style="dim")
        params_table.add_column("Value", justify="right")
        for key, value in opt_result["best_params"].items():
            params_table.add_row(key, str(value))
        console.print(params_table)
    else:
        console.print(f"\n[bold blue]{model_name} Optimization Results:[/bold blue]")
        console.print("  No optimization parameters found")

def main(is_fast_mode: bool = False, interactive: bool = False, challenge_key: str = None, model_names: List[str] = None):
    """Run the HRM vs HREM demonstration in the console, parameterized by challenge and models."""
    try:
        # Create configuration
        demo_config = DemoConfiguration(is_fast_mode, interactive)
        
        # Get challenge - either from command line or menu
        if challenge_key:
            selected_challenge = get_challenge_by_name(challenge_key)
            if not selected_challenge:
                # Try to find by name field or partial match
                all_challenges = get_all_challenges_sorted()
                for challenge in all_challenges:
                    if (challenge_key.lower() == challenge.name.lower() or 
                        challenge_key.lower() in challenge.name.lower() or 
                        challenge.name.lower() in challenge_key.lower()):
                        selected_challenge = challenge
                        break
                
                if not selected_challenge:
                    console.print(f"[bold red]❌ Challenge '{challenge_key}' not found![/bold red]")
                    console.print("[dim]Available challenges:[/dim]")
                    for challenge in all_challenges:
                        console.print(f"[dim]  • {challenge.name}[/dim]")
                    sys.exit(1)
        else:
            # Display challenge selection menu
            selected_challenge = ChallengeSelector.display_challenge_menu()
        
        # Get model configurations
        if not model_names:
            model_names = ["HRM", "HREM"]  # Default models
        
        model_configs = get_model_configs(model_names)
        if not model_configs:
            console.print("[bold red]❌ No valid models specified![/bold red]")
            sys.exit(1)
        
        # Display challenge details
        console.print(f"\n[bold]Selected Challenge:[/bold] [cyan]{selected_challenge.name}[/cyan]")
        console.print(f"[dim]{selected_challenge.description}[/dim]")
        console.print(f"💻 {selected_challenge.recommended_hardware} | ⏱️  {selected_challenge.expected_duration}")
        console.print(f"📈 Difficulty: {selected_challenge.difficulty.value.capitalize()}")
        
        # Display models to be compared
        console.print(f"\n[bold]Models to be compared:[/bold] [cyan]{', '.join([config.name for config in model_configs])}[/cyan]")
        
        if interactive:
            try:
                input("\n[bold green]Press Enter to start the demonstration...[/bold green]")
            except EOFError:
                pass  # Continue if input is not available
        
        # Header
        console.clear()
        console.print(Panel(f"[bold blue]🚀 HRM vs HREM Demonstration: {selected_challenge.name}[/bold blue]\n[italic]Complete end-to-end system showcasing real-time results generation[/italic]", expand=False))
        
        # Introduction
        console.print(f"\n[bold]Challenge:[/bold] {selected_challenge.name}")
        console.print(f"[dim]{selected_challenge.description}[/dim]")
        console.print("\n[bold]This demonstration showcases:[/bold]")
        console.print("• 🔍 Hyperparameter optimization with real-time feedback")
        console.print("• 🏆 Best results and their parameters")
        
        # Approach explanation
        console.print("\n[bold blue]🧠 Approach Explanation[/bold blue]")
        console.print("The HRM System uses a novel approach to hyperparameter optimization:")
        console.print("• Uses guided search to explore hyperparameter space efficiently")
        console.print("• Generates actionable results after each iteration")
        console.print("• Continuously improves based on real-time feedback")
        
        # Create experiment runner
        runner = ExperimentRunner(demo_config)
        
        # Step 1: Hyperparameter optimization
        # Optimize all specified models
        optimization_results = runner.run_hyperparameter_optimization("cli_demo_optimization", selected_challenge.data_config, model_configs)
        
        # Display optimization results
        for model_name, opt_result in optimization_results.items():
            _display_optimization_results(model_name, opt_result)
        
        if interactive:
            try:
                input("\n[bold blue]Press Enter to see final evaluation...[/bold blue]")
            except EOFError:
                pass  # Continue if input is not available
        
        # Step 2: Final evaluation with best parameters
        final_results = runner.run_final_evaluation(optimization_results, model_configs, "cli_demo_final", selected_challenge.data_config)
        ResultsDisplay.display_model_detailed_stats("🏆 Best Results", final_results)
        
        # Show final leader
        ResultsDisplay.display_final_leader(final_results)
        
        # Summary
        ResultsDisplay.display_iteration_header("✅ Demonstration Completed Successfully!")
        
        # Display final comparison
        ResultsDisplay.display_final_comparison("📊 Best Performance Comparison", final_results)
        
        # Key insights
        console.print("\n[bold]🔑 Key Insights:[/bold]")
        console.print("• HRM: Traditional recurrent model with external memory")
        console.print("• HREM: Hierarchical approach with multiple memory layers")
        console.print("• Multi-objective optimization balances accuracy and parameter efficiency")
        console.print("• All models are optimized for fair parameter count comparison")
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
    parser.add_argument("--challenge-key", type=str, help="Specify challenge key directly (skips menu)")
    parser.add_argument("--models", nargs="+", default=["HRM", "HREM"], 
                        help="Specify models to compare (default: HRM HREM)")
    args = parser.parse_args()
    
    main(is_fast_mode=args.fast, interactive=args.interactive, challenge_key=args.challenge_key, model_names=args.models)