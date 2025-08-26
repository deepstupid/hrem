"""Base demo runner for the HRM/HREM demo system."""

import time
from typing import Dict, Any, List
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from demo_utils import DemoLogger, ResultsDisplay
from demo_metrics import MetricsCollector
from hrm_system.config import (
    ExperimentConfig,
    ModelConfig,
    HREMParams,
)
from hrm_system.runner import run_single_model
from hrm_system.evaluation import run_evaluation

console = Console()

class DemoRunner:
    """Handles running the different phases of the demo."""

    def __init__(self, config: ExperimentConfig, results_displayer: ResultsDisplay):
        self.config = config
        self.logger = DemoLogger()
        self.metrics_collector = MetricsCollector()
        self.results_displayer = results_displayer
        # A bit of a hack to get the UI config, assuming it's passed via the logger
        # In a real app, this would be handled more elegantly.
        self.ui_config = getattr(results_displayer, 'ui_config', {})

    def _handle_dataset_error(self, e: Exception):
        """Handle dataset-related errors more gracefully."""
        runner_ui = self.ui_config.get("experiment_runner", {})
        if "No such file or directory" in str(e) and "raw-data" in str(e):
            console.print(runner_ui.get("dataset_not_found_title", "[bold red]Dataset not found![/bold red]"))
            console.print(runner_ui.get("dataset_not_found_message", "Dataset not found").format(dataset=self.config.data_config.dataset))
            console.print(runner_ui.get("dataset_not_found_instructions", "Please download the required dataset files."))
            raise SystemExit(1)
        else:
            raise e

    def run_baseline_evaluation(self) -> Dict[str, Any]:
        """Run a baseline evaluation for each model with default parameters."""
        display_ui = self.ui_config.get("main_display", {})
        runner_ui = self.ui_config.get("experiment_runner", {})

        self.results_displayer.display_iteration_header(
            display_ui.get("baseline_header", "Baseline Evaluation"),
            display_ui.get("baseline_description", "Running baseline...").format(dataset=self.config.data_config.dataset)
        )

        baseline_results = {}
        # The models to evaluate are in the evaluation_config
        model_configs = self.config.evaluation_config.get_models()

        for model_config in model_configs:
            start_time = self.metrics_collector.start_timer()

            spinner_description = runner_ui.get("baseline_spinner", "Running...").format(model_name=model_config.name)
            with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True) as progress:
                progress.add_task(description=spinner_description, total=None)
                try:
                    metrics = run_single_model(
                        run_config=self.config.run_config,
                        data_config=self.config.data_config,
                        model_config=model_config,
                        training_config=self.config.training_config,
                        run_identifier=f"baseline_{model_config.name}"
                    )
                    baseline_results[model_config.name] = metrics
                except Exception as e:
                    self._handle_dataset_error(e)

            elapsed_time = self.metrics_collector.end_timer(start_time)
            self.metrics_collector.record_timing(f"baseline_{model_config.name}", elapsed_time)
            completion_message = runner_ui.get("baseline_completion", "{model_name} done in {elapsed_time:.1f}s").format(
                model_name=model_config.name, elapsed_time=elapsed_time
            )
            console.print(f"[dim]{completion_message}[/dim]")

        return baseline_results

    def run_final_evaluation(self, optimized_results: Dict[str, Any]) -> Dict[str, Any]:
        """Run final evaluation with all optimized models."""
        display_ui = self.ui_config.get("main_display", {})
        runner_ui = self.ui_config.get("experiment_runner", {})

        self.results_displayer.display_iteration_header(
            display_ui.get("evaluation_header", "Final Evaluation"),
            display_ui.get("evaluation_description", "Running final eval...").format(dataset=self.config.data_config.dataset)
        )

        start_time = self.metrics_collector.start_timer()

        # Create a new ExperimentConfig for the final evaluation
        final_eval_config = self.config.model_copy(deep=True)
        final_eval_config.evaluation_config.clear_models()

        final_model_configs = []
        baseline_models = self.config.evaluation_config.get_models()

        for model_name, opt_result in optimized_results.items():
            if opt_result and "best_params" in opt_result:
                baseline_config = next((c for c in baseline_models if c.name == model_name), None)
                if baseline_config:
                    if "hrem" in baseline_config.algorithm_class.lower():
                        optimized_config = baseline_config.model_copy(update={"name": f"{model_name}_best", "hrem_params": HREMParams(**opt_result["best_params"])})
                    else:
                        optimized_config = baseline_config.model_copy(update={"name": f"{model_name}_best", "arch_overrides": opt_result["best_params"] or {}})
                    final_model_configs.append(optimized_config)

        # Add baseline models that were not optimized
        for config in baseline_models:
            if config.name not in optimized_results or not optimized_results[config.name]:
                final_model_configs.append(config)

        if not final_model_configs:
            console.print(runner_ui.get("no_models_to_evaluate", "No models to evaluate."))
            return {}

        final_eval_config.evaluation_config.set_models(final_model_configs)

        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True) as progress:
            progress.add_task(description=runner_ui.get("evaluation_spinner", "Running..."), total=None)
            try:
                results = run_evaluation(final_eval_config)
            except Exception as e:
                self._handle_dataset_error(e)

        elapsed_time = self.metrics_collector.end_timer(start_time)
        self.metrics_collector.record_timing("evaluation_total", elapsed_time)
        completion_message = runner_ui.get("evaluation_completion", "Done in {elapsed_time:.1f}s").format(elapsed_time=elapsed_time)
        console.print(f"[dim]{completion_message}[/dim]")

        return results.get("results", {})

    def display_timing_summary(self):
        """Display a summary of all recorded timings."""
        self.metrics_collector.display_timing_summary()
