"""Adaptive demo runner for the HRM/HREM demo system."""

from typing import Dict, Any, List
import time
import optuna
from rich.console import Console

import subprocess
from rich.table import Table
from rich import box
from rich.live import Live
from base_demo_runner import DemoRunner
from hrm_system.config import DataConfig, ModelConfig, RunConfig, TrainingConfig, HREMParams
from hrm_system.runner import run_single_model
from demo_config import PatienceLevel

console = Console()

class AdaptiveDemoRunner(DemoRunner):
    """
    An enhanced demo runner that adaptively manages the experiment process.
    """
    def _get_best_trial_info(self, study: optuna.Study) -> Dict[str, Any]:
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

    def _run_trial(self, trial: optuna.trial.Trial, model_config: ModelConfig, data_config: DataConfig) -> float:
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
            if self.config.patience == PatienceLevel.LOW:
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
                run_config=RunConfig(smoke_test=(self.config.patience == PatienceLevel.LOW)),
                data_config=data_config, model_config=trial_model_config,
                training_config=TrainingConfig(epochs=config_settings["opt_epochs"], eval_interval=config_settings["opt_eval_interval"]),
                run_identifier=f"trial_{trial.number}"
            )
            loss = metrics.get('all/lm_loss', float('inf'))
            return float(loss) if loss is not None else float('inf')
        except (subprocess.CalledProcessError, FileNotFoundError, RuntimeError) as e:
            raise optuna.TrialPruned()

    def run_hyperparameter_optimization(self, study_name: str, data_config: DataConfig, model_configs: List[ModelConfig],
                                      storage_path: str = None, n_jobs: int = 1) -> Dict[str, Any]:
        """
        Run hyperparameter optimization with adaptive logic for time, resources, and performance.
        """
        display_ui = self.ui_config["main_display"]
        self.results_displayer.display_iteration_header(display_ui["optimization_header"],
                                              display_ui["optimization_description"].format(dataset=data_config.dataset))
        console.print("[bold blue]Using adaptive optimization strategy.[/bold blue]")

        # Adaptive settings
        settings = self.config.settings
        time_budget_seconds = settings["time_budget_seconds"]
        min_trials_per_model = settings["min_trials_per_model"]
        max_trials_per_model = settings["max_trials_per_model"]
        no_improvement_patience = settings["no_improvement_patience"]
        improvement_threshold = settings["improvement_threshold"]

        # Track state for each model
        model_states = {
            mc.name: {
                "study": optuna.create_study(study_name=f"{study_name}_{mc.name}", storage=storage_path, direction="minimize", load_if_exists=True),
                "best_value": float('inf'),
                "trials_done": 0,
                "no_improvement_count": 0,
                "active": True
            } for mc in model_configs if self.config.models[mc.name].get("search_space")
        }

        if not model_states:
            console.print("[yellow]No models with defined search spaces to optimize.[/yellow]")
            return {}

        # TODO: Implement parallel execution with n_jobs and a callback for the live display.
        # For now, we run sequentially.

        # Setup live display
        table = Table(title="Live Adaptive Optimization Progress", box=box.HORIZONTALS)
        table.add_column("Model", style="cyan", no_wrap=True)
        table.add_column("Trials", style="yellow")
        table.add_column("Best Value", style="magenta")
        table.add_column("Status", style="green")
        model_rows = {name: i for i, name in enumerate(model_states.keys())}
        for name in model_states.keys():
            table.add_row(name, "0", "N/A", "Initializing")

        start_time = time.time()

        with Live(table, console=console, screen=False, refresh_per_second=4) as live:
            def callback(study, trial):
                model_name = study.study_name.split('_')[-1]
                state = model_states[model_name]
                state["trials_done"] += 1

                # Update state and check for improvement
                if trial.value is not None:
                    if trial.value < state["best_value"] - improvement_threshold:
                        state["best_value"] = trial.value
                        state["no_improvement_count"] = 0
                        state["status"] = "Improving"
                    else:
                        state["no_improvement_count"] += 1
                        state["status"] = f"Stalled ({state['no_improvement_count']}/{no_improvement_patience})"

                # Update live display
                best_trial_info = self._get_best_trial_info(state["study"])
                best_value_str = f"{best_trial_info['value']:.4f}" if best_trial_info and best_trial_info['value'] is not None else "N/A"
                table.rows[model_rows[model_name]]._cells = [model_name, str(state["trials_done"]), best_value_str, state["status"]]
                live.update(table)

            # This is a blocking call, so we need to manage time budget with a timeout.
            # We will run optimization for each model separately to handle early stopping.
            for model_name, state in model_states.items():
                if not state["active"]:
                    continue

                mc = next(m for m in model_configs if m.name == model_name)

                # Use a partial to pass model_config and data_config to the objective
                objective = lambda trial: self._run_trial(trial, mc, data_config)

                # Calculate remaining time and trials
                remaining_time = time_budget_seconds - (time.time() - start_time)
                if remaining_time <= 0:
                    console.print("[bold yellow]Time budget exceeded. Stopping optimization.[/bold yellow]")
                    break

                # We can't easily combine a trial limit and a timeout with a simple round-robin.
                # A better approach is to run optimize for each model for a number of trials,
                # and check the time budget between models.
                # This is a simplification of the adaptive logic, but it allows for parallel execution.

                trials_to_run = max_trials_per_model - state["trials_done"]

                try:
                    state["study"].optimize(objective, n_trials=trials_to_run, timeout=remaining_time, n_jobs=n_jobs, callbacks=[callback])
                except Exception as e:
                    # Catch exceptions from optimize, e.g., if the timeout is hit.
                    console.print(f"[red]Error during optimization for {model_name}: {e}[/red]")

                # After optimizing, check for early stopping
                if state["no_improvement_count"] >= no_improvement_patience:
                    state["active"] = False
                    state["status"] = "Stopped (no improvement)"

                if state["trials_done"] >= max_trials_per_model:
                    state["active"] = False
                    state["status"] = "Stopped (max trials)"

                # Update final status on the table
                best_trial_info = self._get_best_trial_info(state["study"])
                best_value_str = f"{best_trial_info['value']:.4f}" if best_trial_info and best_trial_info['value'] is not None else "N/A"
                table.rows[model_rows[model_name]]._cells = [model_name, str(state["trials_done"]), best_value_str, state["status"]]
                live.update(table)

        # Collect final results
        optimization_results = {}
        for name, state in model_states.items():
            best_trial_info = self._get_best_trial_info(state["study"])
            if best_trial_info:
                optimization_results[name] = {"best_trial": best_trial_info["number"], "best_params": best_trial_info["params"], "best_value": best_trial_info["value"]}
            else:
                optimization_results[name] = {}

        elapsed_time = time.time() - start_time
        self.metrics_collector.record_timing("optimization_total", elapsed_time)

        return optimization_results
