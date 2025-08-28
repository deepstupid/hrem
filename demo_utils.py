"""Shared utilities for the HRM/HREM demo system."""

import optuna
import os
import glob
from typing import Dict, Any, Optional
from rich.console import Console

console = Console()

def get_best_trial_info(study: optuna.Study) -> Optional[Dict[str, Any]]:
    """Safely retrieves information about the best trial from a study."""
    try:
        return {
            "number": study.best_trial.number,
            "params": study.best_trial.params,
            "value": study.best_trial.value
        }
    except ValueError:
        return None

def clear_optuna_studies(study_name_prefix: str):
    """Clear any existing Optuna studies with the given prefix."""
    db_files = glob.glob(f"{os.path.abspath('experiments')}/optuna_*_{study_name_prefix}.db")
    for db_file in db_files:
        try:
            os.remove(db_file)
            console.print(f"[yellow]Cleared previous study: {db_file}[/yellow]")
        except Exception as e:
            console.print(f"[red]Failed to clear study {db_file}: {e}[/red]")
