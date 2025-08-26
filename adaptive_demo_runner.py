"""Adaptive demo runner for the HRM/HREM demo system."""

from typing import Dict, Any
import time
import optuna
from rich.console import Console
from rich.table import Table
from rich import box
from rich.live import Live

from base_demo_runner import DemoRunner
from hrm_system.config import HREMParams
from hrm_system.runner import run_single_model
from scientific_reporting import ScientificReporter

console = Console()

class AdaptiveDemoRunner(DemoRunner):
    """
    An enhanced demo runner that adaptively manages the experiment process.
    """

    def _get_best_trial_info(self, study: optuna.Study) -> Dict[str, Any]:
        """Safely retrieves information about the best trial from a study."""
        try:
            return {"number": study.best_trial.number, "params": study.best_trial.params, "value": study.best_trial.value}
        except ValueError:
            return None

    def _get_motivational_message(self, state: Dict[str, Any]) -> str:
        """Generates a motivational message based on the optimization state."""
        if state["status"] == "Improving": return "Great find! We're making progress. ✨"
        if "Stalled" in state["status"]: return "Keep searching! The best is yet to come. 🕵️"
        if state["trials_done"] == 0: return "Let's find the best model! 🚀"
        if state["trials_done"] % 5 == 0: return "Pushing the boundaries... 🌌"
        return "Analyzing the results... 📊"

    def _run_trial(self, trial: optuna.trial.Trial) -> float:
        """Execute a single trial for a given model."""
        opt_config = self.config.optimization_config
        model_config = opt_config.model_to_optimize
        search_space = opt_config.search_space or {}

        params = {}
        for param_name, definition in search_space.items():
            param_type = definition.get("type")
            if param_type == "categorical":
                params[param_name] = trial.suggest_categorical(param_name, definition["choices"])
            elif param_type == "int":
                params[param_name] = trial.suggest_int(param_name, definition["low"], definition["high"])

        trial_model_config = model_config.model_copy(deep=True)
        if "hrem" in trial_model_config.algorithm_class.lower():
            trial_model_config.hrem_params = HREMParams(**params)
        else:
            trial_model_config.arch_overrides = params

        try:
            metrics = run_single_model(
                run_config=self.config.run_config,
                data_config=self.config.data_config,
                model_config=trial_model_config,
                training_config=self.config.training_config,
                run_identifier=f"trial_{trial.number}"
            )
            return metrics.get('all/lm_loss', float('inf'))
        except Exception:
            raise optuna.TrialPruned()

    def run_hyperparameter_optimization(self, rich_callback: callable = None) -> Dict[str, Any]:
        """Run hyperparameter optimization with adaptive logic."""
        opt_config = self.config.optimization_config
        study_name = self.config.run_config.study_name
        model_to_optimize = opt_config.model_to_optimize.name

        display_ui = self.ui_config.get("main_display", {})
        self.results_displayer.display_iteration_header(
            display_ui.get("optimization_header", "Hyperparameter Optimization"),
            display_ui.get("optimization_description", "Optimizing...").format(dataset=self.config.data_config.dataset)
        )

        study = optuna.create_study(study_name=study_name, storage=opt_config.storage, direction="minimize", load_if_exists=True)

        table = Table(title="Live Optimization Progress", box=box.HORIZONTALS)
        table.add_column("Model", style="cyan")
        table.add_column("Trials", style="yellow")
        table.add_column("Best Value", style="magenta")
        table.add_column("Status", style="green")
        table.add_row(model_to_optimize, "0", "N/A", "Initializing")

        def live_update_callback(study, trial):
            state = {"trials_done": len(study.trials), "status": "Improving"}
            best_trial_info = self._get_best_trial_info(study)
            best_value_str = f"{best_trial_info['value']:.4f}" if best_trial_info else "N/A"
            table.rows[0]._cells = [model_to_optimize, str(state["trials_done"]), best_value_str, state["status"]]
            if rich_callback:
                rich_callback(table)

        with Live(table, console=console, screen=False, auto_refresh=False) as live:
            def live_refresh_callback(study, trial):
                live_update_callback(study, trial)
                live.refresh()

            callbacks = [live_refresh_callback] if not rich_callback else [live_update_callback]
            study.optimize(self._run_trial, n_trials=opt_config.n_trials, n_jobs=opt_config.n_jobs, callbacks=callbacks)

        best_trial_info = self._get_best_trial_info(study)
        optimization_results = {model_to_optimize: best_trial_info} if best_trial_info else {}

        if self.config.run_config.output_dir and study.trials:
            report_path = ScientificReporter.generate_optimization_report(
                study=study,
                output_dir=self.config.run_config.output_dir / model_to_optimize
            )
            console.print(f"Generated scientific report for {model_to_optimize}: [link=file://{report_path}]{report_path}[/link]")

        return optimization_results
