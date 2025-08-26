"""Configuration management for the HRM/HREM demo system."""

from typing import Dict, Any, List, Tuple
import yaml
from enum import Enum
from hrm_system.config import (
    DataConfig,
    TrainingConfig,
    OptimizationConfig,
    ModelConfig,
)
from demo_models import get_model_configs, get_model_search_space

class PatienceLevel(Enum):
    """Enumeration of demo patience levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

def load_ui__config() -> Dict[str, Any]:
    """Load the UI configuration from the demo YAML file."""
    try:
        with open("config/demo_config.yaml", 'r') as f:
            return yaml.safe_load(f).get("ui", {})
    except FileNotFoundError:
        return {}

def load_challenge_config() -> List[Dict[str, Any]]:
    """Load the challenge configurations from the challenge YAML file."""
    try:
        with open("config/challenge_config.yaml", 'r') as f:
            return yaml.safe_load(f).get("challenges", [])
    except FileNotFoundError:
        return []

def get_configs_for_challenge(
    challenge: Dict[str, Any],
    smoke_test: bool = False
) -> Tuple[DataConfig, TrainingConfig, OptimizationConfig, List[ModelConfig]]:
    """
    Get core Pydantic configuration objects for a given challenge.

    Args:
        challenge: A dictionary representing a single challenge.
        smoke_test: A flag to override settings for a quick smoke test.

    Returns:
        A tuple containing the DataConfig, TrainingConfig, OptimizationConfig,
        and a list of ModelConfig objects.
    """
    patience = PatienceLevel(challenge.get("patience_level", "medium"))
    if smoke_test:
        patience = PatienceLevel.LOW

    # --- Training Config based on Patience ---
    if patience == PatienceLevel.LOW:
        training_config = TrainingConfig(epochs=1, eval_interval=1)
    elif patience == PatienceLevel.MEDIUM:
        training_config = TrainingConfig(epochs=150, eval_interval=50)
    else: # HIGH
        training_config = TrainingConfig(epochs=300, eval_interval=50)

    # --- Optimization Config based on Patience ---
    if patience == PatienceLevel.LOW:
        optimization_config = OptimizationConfig(n_trials=2, n_jobs=1, n_final_runs=1)
    elif patience == PatienceLevel.MEDIUM:
        optimization_config = OptimizationConfig(n_trials=15, n_jobs=4, n_final_runs=3)
    else: # HIGH
        optimization_config = OptimizationConfig(n_trials=50, n_jobs=8, n_final_runs=5)

    # --- Data Config ---
    data_config = DataConfig(**challenge.get("dataset", {}))

    # --- Model Configs ---
    model_names = challenge.get("models", [])
    model_configs = get_model_configs(model_names)

    # Add search space to the model to be optimized if applicable
    if challenge.get("optimization") and model_configs:
        # For simplicity, we assume the last model in the list is the one to optimize
        model_to_optimize = model_configs[-1]
        search_space = get_model_search_space(model_to_optimize.name)
        if search_space:
            optimization_config.model_to_optimize = model_to_optimize
            optimization_config.search_space = search_space

    return data_config, training_config, optimization_config, model_configs