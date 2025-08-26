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

from hrm_system.config import (
    ExperimentConfig,
    RunConfig,
    DataConfig,
    TrainingConfig,
    EvaluationConfig,
    OptimizationConfig,
    ModelConfig,
    HREMParams,
)
from hrm_system.runner import run_single_model
from hrm_system.evaluation import run_evaluation
from hrm_system.optimization import run_optimization

# Import the challenge system
from challenges import (
    Challenge,
    ChallengeDifficulty,
    get_all_challenges_sorted,
    get_challenge_by_name
)

# Import demo utilities
from demo_utils import (
    DemoLogger,
    ResultsDisplay,
    display_optimization_results
)

# Import metrics collector
from demo_metrics import MetricsCollector

# Import configuration
from demo_config import AlgorithmConfigFactory, DemoMode, get_demo_config

# Import model registry
from demo_models import get_model_configs, list_available_models, get_all_model_configs

# Import validation
from demo_validation import ConfigValidator

# Import parameters
from demo_parameters import (
    CHALLENGE_SELECTOR_PARAMS,
    EXPERIMENT_RUNNER_PARAMS,
    DEMO_DISPLAY_PARAMS,
    STORAGE_PATHS,
    DEFAULT_MODEL_NAMES
)

console = Console()

class ChallengeSelector:
    """Handles challenge selection and display."""
    
    @staticmethod
    def _display_challenge_list(challenges: List[Challenge]):
        """Display the list of challenges grouped by difficulty."""
        difficulty_display = CHALLENGE_SELECTOR_PARAMS["difficulty_display"]
        hardware_icon = CHALLENGE_SELECTOR_PARAMS["hardware_icon"]
        duration_icon = CHALLENGE_SELECTOR_PARAMS["duration_icon"]
        
        console.clear()
        console.print(Panel(CHALLENGE_SELECTOR_PARAMS["title"], expand=False))
        
        # Display challenges grouped by difficulty
        for difficulty in [ChallengeDifficulty.BEGINNER, ChallengeDifficulty.INTERMEDIATE, 
                          ChallengeDifficulty.ADVANCED, ChallengeDifficulty.RESEARCH]:
            difficulty_challenges = [c for c in challenges if c.difficulty == difficulty]
            if difficulty_challenges:
                console.print(f"\n[bold]{difficulty_display[difficulty.value.upper()]} Challenges:[/bold]")
                for i, challenge in enumerate(difficulty_challenges, 1):
                    # Find the global index
                    global_index = challenges.index(challenge) + 1
                    console.print(f"  {global_index:2d}. [cyan]{challenge.name}[/cyan]")
                    console.print(f"      [dim]{challenge.description}[/dim]")
                    console.print(f"      {hardware_icon}  {challenge.recommended_hardware} | {duration_icon}  {challenge.expected_duration}")
    
    @staticmethod
    def _get_user_selection(challenges: List[Challenge]) -> Challenge:
        """Get user selection from the challenge list."""
        while True:
            try:
                choice = Prompt.ask(CHALLENGE_SELECTOR_PARAMS["prompt"])
                
                # Try to parse as number first
                if choice.isdigit():
                    index = int(choice) - 1
                    if 0 <= index < len(challenges):
                        return challenges[index]
                
                # Try to match by name
                for challenge in challenges:
                    if challenge.name.lower() == choice.lower():
                        return challenge
                        
                console.print(CHALLENGE_SELECTOR_PARAMS["invalid_selection"])
            except KeyboardInterrupt:
                console.print(CHALLENGE_SELECTOR_PARAMS["exit_message"])
                sys.exit(0)
    
    @staticmethod
    def display_challenge_menu() -> Challenge:
        """Display a menu for selecting a challenge and return the selected challenge."""
        challenges = get_all_challenges_sorted()
        ChallengeSelector._display_challenge_list(challenges)
        return ChallengeSelector._get_user_selection(challenges)

class ExperimentRunner:
    """Handles running the different phases of the experiment."""
    
    def __init__(self, demo_config):
        self.config = demo_config
        self.logger = DemoLogger()
        self.metrics_collector = MetricsCollector()
    
    def _handle_dataset_error(self, e: Exception, data_config: DataConfig):
        """Handle dataset-related errors more gracefully."""
        if "No such file or directory" in str(e) and "raw-data" in str(e):
            console.print(EXPERIMENT_RUNNER_PARAMS["dataset_not_found_title"])
            console.print(EXPERIMENT_RUNNER_PARAMS["dataset_not_found_message"].format(dataset=data_config.dataset))
            console.print(EXPERIMENT_RUNNER_PARAMS["dataset_not_found_instructions"])
            console.print(EXPERIMENT_RUNNER_PARAMS["dataset_not_found_arc_instructions"])
            raise SystemExit(1)
        else:
            # Re-raise other exceptions
            raise e
    
    def run_hyperparameter_optimization_for_model(self, model_config: ModelConfig, study_name: str, data_config: DataConfig, 
                                            storage_path: str = None) -> Dict[str, Any]:
        """Run hyperparameter optimization for a single model."""
        config_settings = self.config.experiment_settings
        
        # Record timing
        start_time = self.metrics_collector.start_timer()
        
        # Create optimization config using the factory
        opt_config_dict = AlgorithmConfigFactory.create_optimization_config(
            model_config.algorithm_class,
            model_config.name,
            config_settings.opt_trials,
            storage_path
        )
        
        # Convert dict to OptimizationConfig object
        opt_config = OptimizationConfig(**opt_config_dict)
        
        config = ExperimentConfig(
            mode="optimize",
            run_config=RunConfig(
                smoke_test=True,
                study_name=f"{study_name}_{model_config.name.lower()}",
                logger_callback=self.logger.log
            ),
            data_config=data_config,
            training_config=TrainingConfig(
                epochs=config_settings.opt_epochs, 
                eval_interval=config_settings.opt_eval_interval
            ),
            optimization_config=opt_config
        )
        
        spinner_description = EXPERIMENT_RUNNER_PARAMS["optimization_spinner"].format(model_name=model_config.name)
        from demo_parameters import PROGRESS_SETTINGS
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=PROGRESS_SETTINGS["transient"],
        ) as progress:
            progress.add_task(description=spinner_description, total=None)
            try:
                result = run_optimization(config)
            except Exception as e:
                self._handle_dataset_error(e, data_config)
        
        elapsed_time = self.metrics_collector.end_timer(start_time)
        self.metrics_collector.record_timing(f"optimization_{model_config.name}", elapsed_time)
        completion_message = EXPERIMENT_RUNNER_PARAMS["optimization_completion"].format(model_name=model_config.name, elapsed_time=elapsed_time)
        console.print(f"[dim]{completion_message}[/dim]")
        
        return result
    
    def run_hyperparameter_optimization(self, study_name: str, data_config: DataConfig, model_configs: List[ModelConfig],
                                  storage_path: str = None) -> Dict[str, Any]:
        """Run hyperparameter optimization for all provided models."""
        ResultsDisplay.display_iteration_header(DEMO_DISPLAY_PARAMS["optimization_header"], 
                               DEMO_DISPLAY_PARAMS["optimization_description"].format(dataset=data_config.dataset))
        console.print(DEMO_DISPLAY_PARAMS["optimization_advantage"])
        console.print()
        
        # Record timing for overall optimization
        start_time = self.metrics_collector.start_timer()
        
        # Run optimization for each model
        optimization_results = {}
        for model_config in model_configs:
            result = self.run_hyperparameter_optimization_for_model(model_config, study_name, data_config, storage_path=storage_path)
            optimization_results[model_config.name] = result
        
        elapsed_time = self.metrics_collector.end_timer(start_time)
        self.metrics_collector.record_timing("optimization_total", elapsed_time)
        
        return optimization_results
    
    def run_final_evaluation(self, optimized_results: Dict[str, Any], baseline_model_configs: List[ModelConfig], study_name: str, data_config: DataConfig) -> Dict[str, Any]:
        """Run final evaluation with all optimized models."""
        ResultsDisplay.display_iteration_header(DEMO_DISPLAY_PARAMS["evaluation_header"], 
                               DEMO_DISPLAY_PARAMS["evaluation_description"].format(dataset=data_config.dataset))
        
        config_settings = self.config.experiment_settings
        
        # Record timing
        start_time = self.metrics_collector.start_timer()
        
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
            console.print(EXPERIMENT_RUNNER_PARAMS["no_models_to_evaluate"])
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
        
        # Explicitly set unused model slots to None to prevent defaults from being used
        for i in range(len(final_model_configs), len(model_keys)):
            eval_config_dict[model_keys[i]] = None
        
        config = ExperimentConfig(
            mode="evaluate",
            run_config=RunConfig(
                smoke_test=True,
                study_name=study_name,
                logger_callback=self.logger.log
            ),
            data_config=data_config,
            training_config=TrainingConfig(
                epochs=config_settings.final_epochs, 
                eval_interval=config_settings.final_eval_interval
            ),
            evaluation_config=EvaluationConfig(**eval_config_dict)
        )
        
        from demo_parameters import PROGRESS_SETTINGS
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=PROGRESS_SETTINGS["transient"],
        ) as progress:
            progress.add_task(description=EXPERIMENT_RUNNER_PARAMS["evaluation_spinner"], total=None)
            try:
                results = run_evaluation(config)
            except Exception as e:
                self._handle_dataset_error(e, data_config)
                
        elapsed_time = self.metrics_collector.end_timer(start_time)
        self.metrics_collector.record_timing("evaluation_total", elapsed_time)
        completion_message = EXPERIMENT_RUNNER_PARAMS["evaluation_completion"].format(elapsed_time=elapsed_time)
        console.print(f"[dim]{completion_message}[/dim]")
            
        return results.get("results", {})
    
    def display_timing_summary(self):
        """Display a summary of all recorded timings."""
        self.metrics_collector.display_timing_summary()

def main(is_fast_mode: bool = False, interactive: bool = False, challenge_key: str = None, model_names: List[str] = None,
         storage_path: str = None):
    """Run the HRM vs HREM demonstration in the console, parameterized by challenge and models."""
    try:
        # Create configuration
        mode = DemoMode.FAST if is_fast_mode else DemoMode.FULL
        demo_config = get_demo_config(mode, interactive)
        
        # Get challenge - either from command line or menu
        if challenge_key:
            selected_challenge = get_challenge_by_name(challenge_key)
            if not selected_challenge:
                # Try to find by name field or partial match
                all_challenges = get_all_challenges_sorted()
                challenge_names = [c.name for c in all_challenges]
                if not ConfigValidator.validate_challenge_key(challenge_key, challenge_names):
                    console.print(DEMO_DISPLAY_PARAMS["challenge_not_found"].format(challenge_key=challenge_key))
                    console.print(DEMO_DISPLAY_PARAMS["available_challenges"])
                    for challenge in all_challenges:
                        console.print(DEMO_DISPLAY_PARAMS["challenge_list_item"].format(challenge_name=challenge.name))
                    sys.exit(1)
                
                # Try to find the challenge
                for challenge in all_challenges:
                    if (challenge_key.lower() == challenge.name.lower() or 
                        challenge_key.lower() in challenge.name.lower() or 
                        challenge.name.lower() in challenge_key.lower()):
                        selected_challenge = challenge
                        break
                
                if not selected_challenge:
                    console.print(DEMO_DISPLAY_PARAMS["challenge_not_found"].format(challenge_key=challenge_key))
                    console.print(DEMO_DISPLAY_PARAMS["available_challenges"])
                    for challenge in all_challenges:
                        console.print(DEMO_DISPLAY_PARAMS["challenge_list_item"].format(challenge_name=challenge.name))
                    sys.exit(1)
        else:
            # Display challenge selection menu
            selected_challenge = ChallengeSelector.display_challenge_menu()
        
        # Get model configurations
        # Use DEFAULT_MODEL_NAMES as the single source of truth for default models
        if not model_names:
            from demo_parameters import DEFAULT_MODEL_NAMES
            model_names = DEFAULT_MODEL_NAMES
        
        # Validate model names
        available_models = list_available_models()
        model_names = ConfigValidator.validate_model_names(model_names, available_models)
        
        if not model_names:
            console.print(DEMO_DISPLAY_PARAMS["no_valid_models"])
            sys.exit(1)
        
        model_configs = get_model_configs(model_names)
        if not model_configs or not ConfigValidator.validate_model_configs(model_configs):
            console.print(DEMO_DISPLAY_PARAMS["invalid_model_configs"])
            sys.exit(1)
        
        # Display challenge details
        console.print(f"\n[bold]Selected Challenge:[/bold] [cyan]{selected_challenge.name}[/cyan]")
        console.print(f"[dim]{selected_challenge.description}[/dim]")
        console.print(DEMO_DISPLAY_PARAMS["challenge_details_format"].format(
            hardware=selected_challenge.recommended_hardware,
            duration=selected_challenge.expected_duration))
        console.print(DEMO_DISPLAY_PARAMS["difficulty_label"].format(difficulty=selected_challenge.difficulty.value.capitalize()))
        
        # Display models to be compared
        console.print(DEMO_DISPLAY_PARAMS["models_comparison_format"].format(
            models=', '.join([config.name for config in model_configs])))
        
        if interactive:
            try:
                input(DEMO_DISPLAY_PARAMS["start_demo_prompt"])
            except EOFError:
                pass  # Continue if input is not available
        
        # Header
        console.clear()
        console.print(Panel(f"[bold blue]{DEMO_DISPLAY_PARAMS['demo_title'].format(challenge_name=selected_challenge.name)}[/bold blue]\n[italic]{DEMO_DISPLAY_PARAMS['demo_subtitle']}[/italic]", expand=False))
        
        # Introduction
        console.print(DEMO_DISPLAY_PARAMS["intro_title"].format(challenge_name=selected_challenge.name))
        console.print(DEMO_DISPLAY_PARAMS["intro_challenge_label"].format(description=selected_challenge.description))
        console.print(DEMO_DISPLAY_PARAMS["intro_showcase_title"])
        for item in DEMO_DISPLAY_PARAMS["intro_showcase_items"]:
            console.print(item)
        
        # Approach explanation
        console.print(DEMO_DISPLAY_PARAMS["approach_title"])
        console.print(DEMO_DISPLAY_PARAMS["approach_explanation"])
        for bullet in DEMO_DISPLAY_PARAMS["approach_bullets"]:
            console.print(bullet)
        
        # Create experiment runner
        runner = ExperimentRunner(demo_config)
        
        # Use default storage path if none provided
        if not storage_path:
            storage_path = STORAGE_PATHS["default_optimization_db"]
        
        # Step 1: Hyperparameter optimization
        # Optimize all specified models
        optimization_results = runner.run_hyperparameter_optimization("cli_demo_optimization", selected_challenge.data_config, model_configs, storage_path=storage_path)
        
        # Display optimization results
        for model_name, opt_result in optimization_results.items():
            display_optimization_results(model_name, opt_result)
        
        if interactive:
            try:
                input(DEMO_DISPLAY_PARAMS["press_enter_evaluation"])
            except EOFError:
                pass  # Continue if input is not available
        
        # Step 2: Final evaluation with best parameters
        final_results = runner.run_final_evaluation(optimization_results, model_configs, "cli_demo_final", selected_challenge.data_config)
        ResultsDisplay.display_model_detailed_stats(DEMO_DISPLAY_PARAMS["best_results_title"], final_results)
        
        # Show final leader
        ResultsDisplay.display_final_leader(final_results)
        
        # Display timing summary
        runner.display_timing_summary()
        
        # Summary
        ResultsDisplay.display_iteration_header(DEMO_DISPLAY_PARAMS["demo_completed_title"])
        
        # Display final comparison
        ResultsDisplay.display_final_comparison(DEMO_DISPLAY_PARAMS["final_comparison_title"], final_results)
        
        # Key insights
        console.print(DEMO_DISPLAY_PARAMS["key_insights_title"])
        for item in DEMO_DISPLAY_PARAMS["key_insights_items"]:
            console.print(item)
        
        # Unique features
        console.print(DEMO_DISPLAY_PARAMS["unique_features_title"])
        for item in DEMO_DISPLAY_PARAMS["unique_features_items"]:
            console.print(item)
        console.print(DEMO_DISPLAY_PARAMS["continuous_improvement_1"])
        console.print(DEMO_DISPLAY_PARAMS["continuous_improvement_2"])
        
    except Exception as e:
        console.print(DEMO_DISPLAY_PARAMS["demo_failed_message"].format(error=e))
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run HRM vs HREM demonstration")
    parser.add_argument("--fast", action="store_true", help="Run in fast mode for testing")
    parser.add_argument("--interactive", action="store_true", help="Enable interactive mode with user prompts")
    parser.add_argument("--challenge-key", type=str, help="Specify challenge key directly (skips menu)")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODEL_NAMES, 
                        help=f"Specify models to compare (default: {' '.join(DEFAULT_MODEL_NAMES)})")
    parser.add_argument("--storage-path", type=str, help="Custom storage path for optimization database")
    
    args = parser.parse_args()
    
    # Pass parameters to main function
    main(is_fast_mode=args.fast, interactive=args.interactive, 
         challenge_key=args.challenge_key, model_names=args.models,
         storage_path=args.storage_path)
