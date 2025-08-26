"""Base demo runner for the HRM/HREM demo system."""

import time
import subprocess
from typing import Dict, Any, List, Optional
import optuna
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import box
from rich.live import Live

from demo_utils import DemoLogger, ResultsDisplay
from demo_metrics import MetricsCollector
from demo_config import PatienceLevel
from hrm_system.config import (
    ExperimentConfig,
    RunConfig,
    DataConfig,
    TrainingConfig,
    EvaluationConfig,
    ModelConfig,
    HREMParams,
)
from hrm_system.runner import run_single_model
from hrm_system.evaluation import run_evaluation

console = Console()

class DemoRunner:
    """Handles running the different phases of the demo."""

    def __init__(self, demo_config, results_displayer):
        self.config = demo_config
        self.logger = DemoLogger()
        self.metrics_collector = MetricsCollector()
        self.ui_config = demo_config.ui
        self.results_displayer = results_displayer

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

    def _get_default_params(self, model_name: str) -> Dict[str, Any]:
        """Extracts default parameters for a model from its search space."""
        params = {}
        model_search_space = self.config.models[model_name].get("search_space", {})
        param_section_key = next(iter(model_search_space), None)
        if not param_section_key:
            return {} # No search space, so no default params to extract

        param_definitions = model_search_space[param_section_key]
        for name, definition in param_definitions.items():
            if self.config.patience == PatienceLevel.LOW and 'smoke_choices' in definition:
                params[name] = definition['smoke_choices'][0]
            elif self.config.patience == PatienceLevel.LOW and 'smoke_low' in definition:
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
                        run_config=RunConfig(smoke_test=(self.config.patience == PatienceLevel.LOW)),
                        data_config=data_config,
                        model_config=baseline_model_config,
                        training_config=TrainingConfig(
                            epochs=config_settings["baseline_epochs"],
                            eval_interval=config_settings["baseline_eval_interval"]
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
            run_config=RunConfig(smoke_test=(self.config.patience == PatienceLevel.LOW), study_name=study_name, logger_callback=self.logger.log),
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
