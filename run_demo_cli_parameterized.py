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
import optuna
import subprocess
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich import box
from rich.prompt import Prompt
from rich.live import Live

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
from demo_config import PatienceLevel, get_demo_config

# Import model registry
from demo_models import get_model_configs, list_available_models, get_all_model_configs

# Import validation
from demo_validation import ConfigValidator

# Import resource utils
from demo_resource_utils import get_resource_profile

# No longer need to import from a separate parameters file
# from demo_parameters import (...)

console = Console()

class ChallengeSelector:
    """Handles challenge selection and display, using a config object."""
    
    def __init__(self, ui_config: Dict[str, Any]):
        self.ui_config = ui_config["challenge_selector"]

    def _display_challenge_list(self, challenges: List[Challenge]):
        """Display the list of challenges grouped by difficulty."""
        difficulty_display = self.ui_config["difficulty_display"]
        hardware_icon = self.ui_config["hardware_icon"]
        duration_icon = self.ui_config["duration_icon"]
        
        console.clear()
        console.print(Panel(self.ui_config["title"], expand=False))
        
        # Display challenges grouped by difficulty
        for difficulty in [ChallengeDifficulty.BEGINNER, ChallengeDifficulty.INTERMEDIATE, 
                          ChallengeDifficulty.ADVANCED, ChallengeDifficulty.RESEARCH]:
            difficulty_challenges = [c for c in challenges if c.difficulty == difficulty]
            if difficulty_challenges:
                console.print(f"\n[bold]{difficulty_display[difficulty.value.upper()]} Challenges:[/bold]")
                for i, challenge in enumerate(difficulty_challenges, 1):
                    global_index = challenges.index(challenge) + 1
                    console.print(f"  {global_index:2d}. [cyan]{challenge.name}[/cyan]")
                    console.print(f"      [dim]{challenge.description}[/dim]")
                    console.print(f"      {hardware_icon}  {challenge.recommended_hardware} | {duration_icon}  {challenge.expected_duration}")
    
    def _get_user_selection(self, challenges: List[Challenge]) -> Challenge:
        """Get user selection from the challenge list."""
        while True:
            try:
                choice = Prompt.ask(self.ui_config["prompt"])
                
                if choice.isdigit():
                    index = int(choice) - 1
                    if 0 <= index < len(challenges):
                        return challenges[index]
                
                for challenge in challenges:
                    if challenge.name.lower() == choice.lower():
                        return challenge
                        
                console.print(self.ui_config["invalid_selection"])
            except KeyboardInterrupt:
                console.print(self.ui_config["exit_message"])
                sys.exit(0)
    
    def display_challenge_menu(self) -> Challenge:
        """Display a menu for selecting a challenge and return the selected challenge."""
        challenges = get_all_challenges_sorted()
        self._display_challenge_list(challenges)
        return self._get_user_selection(challenges)

from adaptive_demo_runner import AdaptiveDemoRunner

def main(patience: PatienceLevel = PatienceLevel.MEDIUM, interactive: bool = False, challenge_key: str = None, model_names: List[str] = None,
         storage_path: str = None):
    """Run the HRM vs HREM demonstration in the console, parameterized by challenge and models."""
    try:
        demo_config = get_demo_config(patience, interactive)
        ui_config = demo_config.ui
        display_ui = ui_config["main_display"]

        # Instantiate the results displayer
        results_displayer = ResultsDisplay(ui_config)
        
        challenge_selector = ChallengeSelector(ui_config)
        if challenge_key:
            selected_challenge = get_challenge_by_name(challenge_key)
            if not selected_challenge:
                all_challenges = get_all_challenges_sorted()
                challenge_names = [c.name for c in all_challenges]
                if not ConfigValidator.validate_challenge_key(challenge_key, challenge_names):
                    console.print(display_ui["challenge_not_found"].format(challenge_key=challenge_key))
                    console.print(display_ui["available_challenges"])
                    for challenge in all_challenges:
                        console.print(display_ui["challenge_list_item"].format(challenge_name=challenge.name))
                    sys.exit(1)
                
                selected_challenge = next((c for c in all_challenges if challenge_key.lower() in c.name.lower() or c.name.lower() in challenge_key.lower()), None)
                
                if not selected_challenge:
                    console.print(display_ui["challenge_not_found"].format(challenge_key=challenge_key))
                    console.print(display_ui["available_challenges"])
                    for challenge in all_challenges:
                        console.print(display_ui["challenge_list_item"].format(challenge_name=challenge.name))
                    sys.exit(1)
        else:
            selected_challenge = challenge_selector.display_challenge_menu()
        
        if not model_names:
            model_names = demo_config.models.keys()
        
        available_models = list_available_models()
        model_names = ConfigValidator.validate_model_names(model_names, available_models)
        
        if not model_names:
            console.print(display_ui["no_valid_models"])
            sys.exit(1)
        
        model_configs = get_model_configs(model_names)
        if not model_configs or not ConfigValidator.validate_model_configs(model_configs):
            console.print(display_ui["invalid_model_configs"])
            sys.exit(1)
        
        console.print(f"\n[bold]Selected Challenge:[/bold] [cyan]{selected_challenge.name}[/cyan]")
        console.print(f"[dim]{selected_challenge.description}[/dim]")
        console.print(display_ui["challenge_details_format"].format(hardware=selected_challenge.recommended_hardware, duration=selected_challenge.expected_duration))
        console.print(display_ui["difficulty_label"].format(difficulty=selected_challenge.difficulty.value.capitalize()))
        
        console.print(display_ui["models_comparison_format"].format(models=', '.join([config.name for config in model_configs])))
        
        if interactive:
            try:
                input(display_ui["start_demo_prompt"])
            except EOFError:
                pass
        
        console.clear()
        console.print(Panel(f"[bold blue]{display_ui['demo_title'].format(challenge_name=selected_challenge.name)}[/bold blue]\n[italic]{display_ui['demo_subtitle']}[/italic]", expand=False))
        
        console.print(display_ui["intro_title"].format(challenge_name=selected_challenge.name))
        console.print(display_ui["intro_challenge_label"].format(description=selected_challenge.description))
        console.print(display_ui["intro_showcase_title"])
        for item in display_ui["intro_showcase_items"]:
            console.print(item)
        
        console.print(display_ui["approach_title"])
        console.print(display_ui["approach_explanation"])
        for bullet in display_ui["approach_bullets"]:
            console.print(bullet)
        
        runner = AdaptiveDemoRunner(demo_config, results_displayer)
        
        if not storage_path:
            storage_path = demo_config.paths["default_optimization_db"]

        # Ensure the directory for the database exists
        db_path = storage_path.split("///")[1]
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Determine resource profile and get n_jobs
        resource_profile = get_resource_profile()
        n_jobs = demo_config.resource_settings[resource_profile]["n_jobs"]
        console.print(f"Detected resource profile: [bold cyan]{resource_profile}[/bold cyan] (using {n_jobs} parallel jobs for optimization)")

        # *** New workflow starts here ***
        baseline_results = runner.run_baseline_evaluation(model_configs, selected_challenge.data_config)
        results_displayer.display_model_detailed_stats(display_ui["baseline_results_title"], baseline_results)

        optimization_results = {}
        if interactive:
            proceed = Prompt.ask(display_ui["ask_for_optimization"], choices=["y", "n"], default="y")
            if proceed == "y":
                optimization_results = runner.run_hyperparameter_optimization("cli_demo_optimization", selected_challenge.data_config, model_configs, storage_path=storage_path, n_jobs=n_jobs)
                for model_name, opt_result in optimization_results.items():
                    display_optimization_results(model_name, opt_result)
        else: # Non-interactive mode runs optimization by default
            optimization_results = runner.run_hyperparameter_optimization("cli_demo_optimization", selected_challenge.data_config, model_configs, storage_path=storage_path, n_jobs=n_jobs)
            for model_name, opt_result in optimization_results.items():
                display_optimization_results(model_name, opt_result)

        if interactive and not optimization_results:
             # If user skipped optimization, there's no final evaluation to run
            console.print(display_ui["skipping_final_eval"])
        else:
            if interactive:
                try:
                    input(display_ui["press_enter_evaluation"])
                except EOFError:
                    pass

            final_results = runner.run_final_evaluation(optimization_results, model_configs, "cli_demo_final", selected_challenge.data_config)
            results_displayer.display_model_detailed_stats(display_ui["best_results_title"], final_results)
            results_displayer.display_final_leader(final_results)
            results_displayer.display_final_comparison(display_ui["final_comparison_title"], final_results)
        
        runner.display_timing_summary()
        results_displayer.display_iteration_header(display_ui["demo_completed_title"])
        
        console.print(display_ui["key_insights_title"])
        for item in display_ui["key_insights_items"]:
            console.print(item)
        
        console.print(display_ui["unique_features_title"])
        for item in display_ui["unique_features_items"]:
            console.print(item)
        console.print(display_ui["continuous_improvement_1"])
        console.print(display_ui["continuous_improvement_2"])
        
    except Exception as e:
        console.print(ui_config["main_display"]["demo_failed_message"].format(error=e))
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run HRM vs HREM demonstration")
    # Load default models from the new config structure to display in help message
    try:
        from demo_config import DEMO_CONFIG
        default_models = list(DEMO_CONFIG.models.keys())
    except (ImportError, KeyError):
        # Fallback if config isn't available yet or is malformed
        default_models = ["HRM", "HREM"]

    parser.add_argument("--patience", type=str, default="medium", choices=["low", "medium", "high"], help="Set the patience level for the demo")
    parser.add_argument("--interactive", action="store_true", help="Enable interactive mode with user prompts")
    parser.add_argument("--challenge-key", type=str, help="Specify challenge key directly (skips menu)")
    parser.add_argument("--models", nargs="+", default=default_models,
                        help=f"Specify models to compare (default: {' '.join(default_models)})")
    parser.add_argument("--storage-path", type=str, help="Custom storage path for optimization database")

    args = parser.parse_args()

    main(patience=PatienceLevel(args.patience), interactive=args.interactive,
         challenge_key=args.challenge_key, model_names=args.models,
         storage_path=args.storage_path)
