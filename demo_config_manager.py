"""Unified configuration management for the HRM/HREM demo system."""

from typing import Dict, Any, List, Tuple, Optional
import yaml
import os
from enum import Enum
from hrm_system.config import (
    DataConfig,
    TrainingConfig,
    OptimizationConfig,
    ModelConfig,
)
from demo_models import get_model_configs, get_model_search_space
from dataset_manager import dataset_manager
from demo_model_runner import get_dataset_config
from rich.console import Console

console = Console()

class PatienceLevel(Enum):
    """Enumeration of demo patience levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class ConfigManager:
    """Centralized configuration manager for the HRM/HREM demo system."""
    
    @staticmethod
    def load_ui_config() -> Dict[str, Any]:
        """Load the UI configuration from the demo YAML file."""
        try:
            with open("config/demo_config.yaml", 'r') as f:
                return yaml.safe_load(f).get("ui", {})
        except FileNotFoundError:
            return {}

    @staticmethod
    def load_challenge_config() -> List[Dict[str, Any]]:
        """Load the challenge configurations from the challenge YAML file."""
        try:
            with open("config/challenge_config.yaml", 'r') as f:
                base_challenges = yaml.safe_load(f).get("challenges", [])
        except FileNotFoundError:
            base_challenges = []
        
        # Add more synthetic challenges
        synthetic_challenges = [
            {
                "name": "Synthetic Reverse",
                "id": "synthetic_reverse",
                "description": "Test models on reversing sequences.",
                "difficulty": "BEGINNER",
                "hardware": "Low",
                "duration": "~1 min",
                "dataset": {
                    "dataset": "synthetic-reverse"
                },
                "models": ["HRM", "HREM"],
                "patience_level": "low",
                "optimization": True
            },
            {
                "name": "Synthetic Sort",
                "id": "synthetic_sort",
                "description": "Test models on sorting sequences.",
                "difficulty": "BEGINNER",
                "hardware": "Low",
                "duration": "~1 min",
                "dataset": {
                    "dataset": "synthetic-sort"
                },
                "models": ["HRM", "HREM"],
                "patience_level": "low",
                "optimization": True
            },
            {
                "name": "Synthetic Parity",
                "id": "synthetic_parity",
                "description": "Test models on parity computation.",
                "difficulty": "BEGINNER",
                "hardware": "Low",
                "duration": "~1 min",
                "dataset": {
                    "dataset": "synthetic-parity"
                },
                "models": ["HRM", "HREM"],
                "patience_level": "low",
                "optimization": True
            },
            {
                "name": "Synthetic Duplicate",
                "id": "synthetic_duplicate",
                "description": "Test models on duplicating sequences.",
                "difficulty": "BEGINNER",
                "hardware": "Low",
                "duration": "~1 min",
                "dataset": {
                    "dataset": "synthetic-duplicate"
                },
                "models": ["HRM", "HREM"],
                "patience_level": "low",
                "optimization": True
            }
        ]
        
        return base_challenges + synthetic_challenges

    @staticmethod
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
            training_config = TrainingConfig(epochs=50, eval_interval=25)
        else: # HIGH
            training_config = TrainingConfig(epochs=150, eval_interval=50)

        # --- Optimization Config based on Patience ---
        if patience == PatienceLevel.LOW:
            optimization_config = OptimizationConfig(n_trials=3, n_jobs=1, n_final_runs=1)
        elif patience == PatienceLevel.MEDIUM:
            optimization_config = OptimizationConfig(n_trials=10, n_jobs=2, n_final_runs=2)
        else: # HIGH
            optimization_config = OptimizationConfig(n_trials=20, n_jobs=4, n_final_runs=3)

        # --- Data Config ---
        dataset_config = challenge.get("dataset", {})
        dataset_name = dataset_config.get("dataset", "synthetic")
        
        # Use dataset manager to get the correct path
        try:
            data_config = get_dataset_config(dataset_name, smoke_test, dataset_config.get("num_aug", 0))
        except Exception as e:
            # Fallback to synthetic dataset if there's an error
            console.print(f"[yellow]Warning: {str(e)}. Falling back to synthetic dataset.[/yellow]")
            data_config = get_dataset_config("synthetic", smoke_test, dataset_config.get("num_aug", 0))
            dataset_config["dataset"] = "synthetic"

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

    @staticmethod
    def get_model_training_config(model_name: str) -> Optional[TrainingConfig]:
        """
        Get the training configuration for a specific model.
        
        Args:
            model_name: Name of the model to get training config for.
            
        Returns:
            TrainingConfig object or None if not found.
        """
        # This would typically load from a config file or database
        # For now, we'll return a default configuration
        return TrainingConfig()