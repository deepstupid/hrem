"""Unified model execution functions for the HRM/HREM demo system."""

import time
import optuna
from rich.console import Console

from demo_config_manager import ConfigManager

console = Console()

def run_model_with_fallback(model_config, run_config, data_config, training_config, run_identifier):
    """Run a model with fallback to synthetic dataset if needed."""
    from hrm_system import run_single_model
    try:
        metrics = run_single_model(
            run_config=run_config,
            data_config=data_config,
            model_config=model_config,
            training_config=training_config,
            run_identifier=run_identifier
        )
        return metrics
    except Exception as e:
        if "not found" in str(e).lower() or "no such file" in str(e).lower():
            console.print(f"[yellow]⚠️  Dataset issue for {model_config.name}. Falling back to synthetic.[/yellow]")
            synthetic_data_config = ConfigManager.get_dataset_config("synthetic", run_config.smoke_test, data_config.num_aug)
            metrics = run_single_model(
                run_config=run_config,
                data_config=synthetic_data_config,
                model_config=model_config,
                training_config=training_config,
                run_identifier=run_identifier
            )
            return metrics
        else:
            raise

def run_trial_with_timing(trial: optuna.trial.Trial, config, timing_manager=None, operation_name=None):
    """Execute a single trial for a given model with optional timing."""
    from hrm_system.config import HREMParams
    start_time = time.time()

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
        trial_model_config.hrem_params = HREMParams(**params)
    else:
        trial_model_config.arch_overrides = params

    try:
        metrics = run_model_with_fallback(
            trial_model_config,
            config.run_config,
            config.data_config,
            config.training_config,
            f"trial_{trial.number}"
        )
        elapsed_time = time.time() - start_time
        if timing_manager and operation_name:
            timing_manager.record_timing(operation_name, elapsed_time)
        return metrics.get('all/lm_loss', float('inf'))
    except Exception:
        raise optuna.TrialPruned()
