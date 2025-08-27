"""Unified model running utilities for the HRM/HREM demo system."""

import optuna
from typing import Dict, Any, Tuple
from hrm_system import run_single_model
from hrm_system.config import DataConfig, HREMParams
from dataset_manager import dataset_manager
from rich.console import Console

console = Console()

def run_model_with_fallback(model_config, run_config, data_config, training_config, run_identifier):
    """Run a model with fallback to synthetic dataset if needed."""
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
        # If dataset-related error, try with synthetic dataset
        if "not found" in str(e).lower() or "no such file" in str(e).lower():
            console.print(f"[yellow]⚠️  Dataset issue for {model_config.name}. Falling back to synthetic.[/yellow]")
            try:
                synthetic_data_config = DataConfig(dataset="synthetic", num_aug=data_config.num_aug)
                metrics = run_single_model(
                    run_config=run_config,
                    data_config=synthetic_data_config,
                    model_config=model_config,
                    training_config=training_config,
                    run_identifier=run_identifier
                )
                return metrics
            except Exception as e2:
                console.print(f"[red]Error running {model_config.name} with synthetic dataset: {str(e2)}[/red]")
                raise
        else:
            # Non-dataset related error
            console.print(f"[red]Error running {model_config.name}: {str(e)}[/red]")
            raise

def run_trial_with_timing(trial: optuna.trial.Trial, config, timing_manager=None, operation_name=None) -> Tuple[float, float]:
    """Execute a single trial for a given model with optional timing."""
    import time
    
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
        metrics = run_single_model(
            run_config=config.run_config,
            data_config=config.data_config,
            model_config=trial_model_config,
            training_config=config.training_config,
            run_identifier=f"trial_{trial.number}"
        )
        
        elapsed_time = time.time() - start_time
        if timing_manager and operation_name:
            timing_manager.record_timing(operation_name, elapsed_time)
            
        return metrics.get('all/lm_loss', float('inf')), elapsed_time
    except Exception:
        elapsed_time = time.time() - start_time
        if timing_manager and operation_name:
            timing_manager.record_timing(operation_name, elapsed_time)
            
        raise optuna.TrialPruned()

def get_dataset_config(dataset: str, smoke_test: bool, num_aug: int = 0) -> DataConfig:
    """Get dataset configuration, with fallback to synthetic if needed."""
    try:
        dataset_path = dataset_manager.get_dataset_path(dataset, smoke_test)
        return DataConfig(dataset=dataset, dataset_path=dataset_path, num_aug=num_aug)
    except Exception as e:
        console.print(f"[red]Error accessing dataset '{dataset}': {str(e)}[/red]")
        if dataset != "synthetic":
            console.print("[yellow]Falling back to synthetic dataset...[/yellow]")
            try:
                dataset_path = dataset_manager.get_dataset_path("synthetic", smoke_test)
                return DataConfig(dataset="synthetic", dataset_path=dataset_path, num_aug=num_aug)
            except Exception as e2:
                console.print(f"[red]Failed to access synthetic dataset: {str(e2)}[/red]")
                raise
        else:
            raise