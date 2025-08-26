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
from demo_config import AlgorithmConfigFactory, DemoMode, get_demo_config

# Import model registry
from demo_models import get_model_configs, list_available_models, get_all_model_configs

# Import validation
from demo_validation import ConfigValidator

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

class DemoRunner:
    """Handles running the different phases of the demo."""
    
    def __init__(self, demo_config, results_displayer):
        self.config = demo_config
        self.logger = DemoLogger()
        self.metrics_collector = MetricsCollector()
        self.ui_config = demo_config.ui
        self.results_displayer = results_displayer

    def _get_best_trial_info(self, study: optuna.Study) -> Optional[Dict[str, Any]]:
        """Safely retrieves information about the best trial from a study."""
        try:
            best_trial = study.best_trial
            return {
                "number": best_trial.number,
                "params": best_trial.params,
                "value": best_trial.value,
            }
        except ValueError:
            return None

    def _handle_dataset_error(self, e: Exception, data_config: DataConfig):
        """Handle dataset-related errors more gracefully."""
        runner_ui = self.ui_config["experiment_runner"]
        if "No such file or directory" in str(e) and "raw-data" in str(e):
            console.print(runner_ui["dataset_not_found_title"])
            console.print(runner_ui["dataset_not_found_message"].format(dataset=data_config.dataset))
            console.print(runner_ui["dataset_not_found_instructions"])
            console.print(runner_ui["dataset_not_found_arc_instructions"])
            raise SystemExit(1)
        else:
            raise e

    def run_hyperparameter_optimization(self, study_name: str, data_config: DataConfig, model_configs: List[ModelConfig],
                                  storage_path: str = None) -> Dict[str, Any]:
        """Run hyperparameter optimization with a live-updating display."""
        display_ui = self.ui_config["main_display"]
        self.results_displayer.display_iteration_header(display_ui["optimization_header"],
                                              display_ui["optimization_description"].format(dataset=data_config.dataset))
        console.print(display_ui["optimization_advantage"])
        
        start_time = self.metrics_collector.start_timer()

        # This is a nested function now, captures necessary context
        def _run_trial(trial: optuna.trial.Trial, model_config: ModelConfig) -> float:
            """Execute a single trial for a given model."""
            config_settings = self.config.settings
            params = {}
            model_search_space = self.config.models[model_config.name].get("search_space", {})
            param_section_key = next(iter(model_search_space), None)
            if not param_section_key:
                return float('inf')

            param_definitions = model_search_space[param_section_key]
            for name, definition in param_definitions.items():
                param_type = definition['type']
                if self.config.mode == DemoMode.FAST:
                    if 'smoke_choices' in definition:
                        params[name] = trial.suggest_categorical(name, definition['smoke_choices'])
                    elif 'smoke_low' in definition and 'smoke_high' in definition:
                        params[name] = trial.suggest_int(name, definition['smoke_low'], definition['smoke_high'])
                    else:
                        params[name] = trial.suggest_categorical(name, definition['choices']) if param_type == 'categorical' else trial.suggest_int(name, definition['low'], definition['high'])
                else:
                    if param_type == "categorical":
                        params[name] = trial.suggest_categorical(name, definition['choices'])
                    elif param_type == "int":
                        params[name] = trial.suggest_int(name, definition['low'], definition['high'])

            trial_model_config = model_config.model_copy(deep=True)
            if "hrem" in trial_model_config.algorithm_class.lower():
                trial_model_config.hrem_params = HREMParams(**params)
            else:
                trial_model_config.arch_overrides = params

            try:
                metrics = run_single_model(
                    run_config=RunConfig(smoke_test=(self.config.mode == DemoMode.FAST)),
                    data_config=data_config, model_config=trial_model_config,
                    training_config=TrainingConfig(epochs=config_settings["opt_epochs"], eval_interval=config_settings["opt_eval_interval"]),
                    run_identifier=f"trial_{trial.number}"
                )
                loss = metrics.get('all/lm_loss', float('inf'))
                return float(loss) if loss is not None else float('inf')
            except (subprocess.CalledProcessError, FileNotFoundError, RuntimeError) as e:
                raise optuna.TrialPruned()

        studies = {}
        models_to_optimize = [mc for mc in model_configs if self.config.models[mc.name].get("search_space")]
        for mc in models_to_optimize:
            studies[mc.name] = optuna.create_study(study_name=f"{study_name}_{mc.name}", storage=storage_path, direction="minimize", load_if_exists=True)

        if not models_to_optimize:
            console.print("[yellow]No models with defined search spaces to optimize.[/yellow]")
            return {}

        table = Table(title="Live Optimization Progress", box=box.HORIZONTALS)
        table.add_column("Model", style="cyan", no_wrap=True)
        table.add_column("Best Value", style="magenta")
        table.add_column("Best Params", style="green")
        model_rows = {mc.name: i for i, mc in enumerate(models_to_optimize)}
        for mc in models_to_optimize:
            table.add_row(mc.name, "N/A", "N/A")

        n_trials = self.config.settings["opt_trials"]
        with Live(table, console=console, screen=False, refresh_per_second=4) as live:
            for trial_num in range(n_trials):
                for model_config in models_to_optimize:
                    study = studies[model_config.name]
                    trial = study.ask()
                    try:
                        value = _run_trial(trial, model_config)
                        study.tell(trial, value)
                    except optuna.TrialPruned:
                        study.tell(trial, state=optuna.trial.TrialState.PRUNED)

                    best_trial_info = self._get_best_trial_info(study)
                    if best_trial_info:
                        best_value = f"{best_trial_info['value']:.4f}"
                        best_params_str = ", ".join(f"{k}={v}" for k, v in best_trial_info['params'].items())
                        table.rows[model_rows[model_config.name]]._cells = [model_config.name, best_value, best_params_str]
                    live.update(table)

        optimization_results = {}
        for mc in model_configs:
            if mc.name in studies:
                study = studies[mc.name]
                best_trial_info = self._get_best_trial_info(study)
                if best_trial_info:
                    optimization_results[mc.name] = {"best_trial": best_trial_info["number"], "best_params": best_trial_info["params"], "best_value": best_trial_info["value"]}
                else:
                    optimization_results[mc.name] = {}
            else:
                optimization_results[mc.name] = {}

        elapsed_time = self.metrics_collector.end_timer(start_time)
        self.metrics_collector.record_timing("optimization_total", elapsed_time)
        
        return optimization_results

    def _get_default_params(self, model_name: str) -> Dict[str, Any]:
        """Extracts default parameters for a model from its search space."""
        params = {}
        model_search_space = self.config.models[model_name].get("search_space", {})
        param_section_key = next(iter(model_search_space), None)
        if not param_section_key:
            return {} # No search space, so no default params to extract

        param_definitions = model_search_space[param_section_key]
        for name, definition in param_definitions.items():
            if self.config.mode == DemoMode.FAST and 'smoke_choices' in definition:
                params[name] = definition['smoke_choices'][0]
            elif self.config.mode == DemoMode.FAST and 'smoke_low' in definition:
                params[name] = definition['smoke_low']
            elif 'choices' in definition:
                params[name] = definition['choices'][0] # Take the first choice as default
            elif 'low' in definition:
                params[name] = definition['low'] # Take the lower bound as default
        return params

    def run_baseline_evaluation(self, model_configs: List[ModelConfig], data_config: DataConfig) -> Dict[str, Any]:
        """Run a baseline evaluation for each model with default parameters."""
        display_ui = self.ui_config["main_display"]
        runner_ui = self.ui_config["experiment_runner"]

        self.results_displayer.display_iteration_header(display_ui["baseline_header"],
                                              display_ui["baseline_description"].format(dataset=data_config.dataset))

        baseline_results = {}
        config_settings = self.config.settings

        for model_config in model_configs:
            start_time = self.metrics_collector.start_timer()

            # Use default parameters for the baseline run
            params = self._get_default_params(model_config.name)

            # Create a temporary model config for this baseline run
            baseline_model_config = model_config.model_copy(deep=True)
            if "hrem" in baseline_model_config.algorithm_class.lower() and params:
                baseline_model_config.hrem_params = HREMParams(**params)
            elif params:
                baseline_model_config.arch_overrides = params

            spinner_description = runner_ui["baseline_spinner"].format(model_name=model_config.name)
            with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True) as progress:
                progress.add_task(description=spinner_description, total=None)
                try:
                    metrics = run_single_model(
                        run_config=RunConfig(smoke_test=(self.config.mode == DemoMode.FAST)),
                        data_config=data_config,
                        model_config=baseline_model_config,
                        training_config=TrainingConfig(
                            epochs=config_settings["final_epochs"], # Using final epochs for a solid baseline
                            eval_interval=config_settings["final_eval_interval"]
                        ),
                        run_identifier=f"baseline_{model_config.name}"
                    )
                    baseline_results[model_config.name] = metrics
                except Exception as e:
                    self._handle_dataset_error(e, data_config)

            elapsed_time = self.metrics_collector.end_timer(start_time)
            self.metrics_collector.record_timing(f"baseline_{model_config.name}", elapsed_time)
            completion_message = runner_ui["baseline_completion"].format(model_name=model_config.name, elapsed_time=elapsed_time)
            console.print(f"[dim]{completion_message}[/dim]")

        return baseline_results

    def run_final_evaluation(self, optimized_results: Dict[str, Any], baseline_model_configs: List[ModelConfig], study_name: str, data_config: DataConfig) -> Dict[str, Any]:
        """Run final evaluation with all optimized models."""
        display_ui = self.ui_config["main_display"]
        runner_ui = self.ui_config["experiment_runner"]
        progress_settings = self.ui_config["progress_settings"]
        
        self.results_displayer.display_iteration_header(display_ui["evaluation_header"],
                               display_ui["evaluation_description"].format(dataset=data_config.dataset))
        
        config_settings = self.config.settings
        start_time = self.metrics_collector.start_timer()
        
        final_model_configs = []
        # Add optimized models
        for model_name, opt_result in optimized_results.items():
            if opt_result and "best_params" in opt_result:
                baseline_config = next((c for c in baseline_model_configs if c.name == model_name), None)
                if baseline_config:
                    if "hrem" in baseline_config.algorithm_class.lower():
                        hrem_params = HREMParams(**opt_result["best_params"])
                        optimized_config = ModelConfig(name=f"{model_name}_best", algorithm_class=baseline_config.algorithm_class,
                                                   base_arch_config=baseline_config.base_arch_config, hrem_params=hrem_params)
                    else:
                        optimized_config = ModelConfig(name=f"{model_name}_best", algorithm_class=baseline_config.algorithm_class,
                                                   base_arch_config=baseline_config.base_arch_config, arch_overrides=opt_result["best_params"] or {})
                    final_model_configs.append(optimized_config)

        # Add baseline models that were not optimized
        for config in baseline_model_configs:
            if config.name not in optimized_results or not optimized_results[config.name]:
                final_model_configs.append(config)

        if not final_model_configs:
            console.print(runner_ui["no_models_to_evaluate"])
            return {}
        
        eval_config_dict = {"n_runs": 1}
        model_keys = ["model_a", "model_b", "model_c", "model_d", "model_e"]
        for i, model_config in enumerate(final_model_configs):
            if i < len(model_keys):
                eval_config_dict[model_keys[i]] = model_config
        for i in range(len(final_model_configs), len(model_keys)):
            eval_config_dict[model_keys[i]] = None
        
        config = ExperimentConfig(
            mode="evaluate",
            run_config=RunConfig(smoke_test=True, study_name=study_name, logger_callback=self.logger.log),
            data_config=data_config,
            training_config=TrainingConfig(epochs=config_settings["final_epochs"], eval_interval=config_settings["final_eval_interval"]),
            evaluation_config=EvaluationConfig(**eval_config_dict)
        )
        
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=progress_settings["transient"]) as progress:
            progress.add_task(description=runner_ui["evaluation_spinner"], total=None)
            try:
                results = run_evaluation(config)
            except Exception as e:
                self._handle_dataset_error(e, data_config)
                
        elapsed_time = self.metrics_collector.end_timer(start_time)
        self.metrics_collector.record_timing("evaluation_total", elapsed_time)
        completion_message = runner_ui["evaluation_completion"].format(elapsed_time=elapsed_time)
        console.print(f"[dim]{completion_message}[/dim]")
            
        return results.get("results", {})
    
    def display_timing_summary(self):
        """Display a summary of all recorded timings."""
        self.metrics_collector.display_timing_summary()

def main(is_fast_mode: bool = False, interactive: bool = False, challenge_key: str = None, model_names: List[str] = None,
         storage_path: str = None):
    """Run the HRM vs HREM demonstration in the console, parameterized by challenge and models."""
    try:
        mode = DemoMode.FAST if is_fast_mode else DemoMode.FULL
        demo_config = get_demo_config(mode, interactive)
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
        
        runner = DemoRunner(demo_config, results_displayer)
        
        if not storage_path:
            storage_path = demo_config.paths["default_optimization_db"]

        # Ensure the directory for the database exists
        db_path = storage_path.split("///")[1]
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # *** New workflow starts here ***
        baseline_results = runner.run_baseline_evaluation(model_configs, selected_challenge.data_config)
        results_displayer.display_model_detailed_stats(display_ui["baseline_results_title"], baseline_results)

        optimization_results = {}
        if interactive:
            proceed = Prompt.ask(display_ui["ask_for_optimization"], choices=["y", "n"], default="y")
            if proceed == "y":
                optimization_results = runner.run_hyperparameter_optimization("cli_demo_optimization", selected_challenge.data_config, model_configs, storage_path=storage_path)
                for model_name, opt_result in optimization_results.items():
                    display_optimization_results(model_name, opt_result)
        else: # Non-interactive mode runs optimization by default
            optimization_results = runner.run_hyperparameter_optimization("cli_demo_optimization", selected_challenge.data_config, model_configs, storage_path=storage_path)
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

    parser.add_argument("--fast", action="store_true", help="Run in fast mode for testing")
    parser.add_argument("--interactive", action="store_true", help="Enable interactive mode with user prompts")
    parser.add_argument("--challenge-key", type=str, help="Specify challenge key directly (skips menu)")
    parser.add_argument("--models", nargs="+", default=default_models,
                        help=f"Specify models to compare (default: {' '.join(default_models)})")
    parser.add_argument("--storage-path", type=str, help="Custom storage path for optimization database")
    
    args = parser.parse_args()
    
    main(is_fast_mode=args.fast, interactive=args.interactive, 
         challenge_key=args.challenge_key, model_names=args.models,
         storage_path=args.storage_path)
