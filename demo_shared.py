"""Shared utilities for the HRM/HREM demo system."""

import optuna
import os
import glob
from typing import Dict, Any
from hrm_system.config import DataConfig
from rich.console import Console
from demo_model_runner import run_model_with_fallback, get_dataset_config

console = Console()

def run_trial(trial: optuna.trial.Trial, config) -> float:
    """Execute a single trial for a given model."""
    opt_config = config.optimization_config
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
        from hrm_system.config import HREMParams
        trial_model_config.hrem_params = HREMParams(**params)
    else:
        trial_model_config.arch_overrides = params

    try:
        metrics = run_model_with_fallback(
            model_config=trial_model_config,
            run_config=config.run_config,
            data_config=config.data_config,
            training_config=config.training_config,
            run_identifier=f"trial_{trial.number}"
        )
        return metrics.get('all/lm_loss', float('inf'))
    except Exception:
        raise optuna.TrialPruned()

def get_best_trial_info(study: optuna.Study) -> Dict[str, Any]:
    """Safely retrieves information about the best trial from a study."""
    try:
        return {"number": study.best_trial.number, "params": study.best_trial.params, "value": study.best_trial.value}
    except ValueError:
        return None

def clear_optuna_studies(study_name_prefix):
    """Clear any existing Optuna studies with the given prefix."""
    db_files = glob.glob(f"{os.path.abspath('experiments')}/optuna_*_{study_name_prefix}.db")
    for db_file in db_files:
        try:
            os.remove(db_file)
            console.print(f"[yellow]Cleared previous study: {db_file}[/yellow]")
        except Exception as e:
            console.print(f"[red]Failed to clear study {db_file}: {e}[/red]")