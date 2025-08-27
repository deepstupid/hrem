"""Shared utilities for the HRM/HREM demo system.

This module provides utility functions that are shared across multiple
demo components, including optimization result handling and cleanup utilities.
"""

import optuna
import os
import glob
from typing import Dict, Any, Optional
from hrm_system.config import DataConfig
from rich.console import Console
from demo_model_runner import run_model_with_fallback, get_dataset_config, run_trial_with_timing

console = Console()

def get_best_trial_info(study: optuna.Study) -> Optional[Dict[str, Any]]:
    """Safely retrieves information about the best trial from a study.
    
    This function safely extracts the best trial information from an Optuna study,
    handling cases where no valid trials exist.
    
    Args:
        study: Optuna study object
        
    Returns:
        Dictionary with best trial information or None if no valid trials
    """
    try:
        return {
            "number": study.best_trial.number, 
            "params": study.best_trial.params, 
            "value": study.best_trial.value
        }
    except ValueError:
        return None

def clear_optuna_studies(study_name_prefix: str):
    """Clear any existing Optuna studies with the given prefix.
    
    This function removes SQLite database files for Optuna studies
    that match the given prefix, helping to keep the experiments directory clean.
    
    Args:
        study_name_prefix: Prefix to match study names against
    """
    db_files = glob.glob(f"{os.path.abspath('experiments')}/optuna_*_{study_name_prefix}.db")
    for db_file in db_files:
        try:
            os.remove(db_file)
            console.print(f"[yellow]Cleared previous study: {db_file}[/yellow]")
        except Exception as e:
            console.print(f"[red]Failed to clear study {db_file}: {e}[/red]")