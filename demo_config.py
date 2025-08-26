"""Configuration management for the HRM/HREM demo system."""

from typing import Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum
from demo_parameters import STORAGE_PATHS, SEARCH_SPACE_PATHS

class DemoMode(Enum):
    """Enumeration of demo modes."""
    FAST = "fast"
    FULL = "full"
    INTERACTIVE = "interactive"

@dataclass
class ExperimentSettings:
    """Settings for different phases of the experiment."""
    baseline_epochs: int
    baseline_eval_interval: int
    opt_epochs: int
    opt_eval_interval: int
    opt_trials: int
    final_epochs: int
    final_eval_interval: int

@dataclass
class DemoConfig:
    """Complete configuration for the demo system."""
    mode: DemoMode
    interactive: bool = False
    experiment_settings: ExperimentSettings = field(default_factory=lambda: ExperimentSettings(
        baseline_epochs=200,
        baseline_eval_interval=50,
        opt_epochs=300,
        opt_eval_interval=50,
        opt_trials=20,
        final_epochs=400,
        final_eval_interval=50
    ))
    
    def __post_init__(self):
        """Initialize based on mode."""
        if self.mode == DemoMode.FAST:
            self.experiment_settings = ExperimentSettings(
                baseline_epochs=50,
                baseline_eval_interval=25,
                opt_epochs=50,
                opt_eval_interval=25,
                opt_trials=3,
                final_epochs=50,
                final_eval_interval=25
            )

# Predefined configurations
DEMO_CONFIGS = {
    DemoMode.FAST: DemoConfig(
        mode=DemoMode.FAST,
        experiment_settings=ExperimentSettings(
            baseline_epochs=50,
            baseline_eval_interval=25,
            opt_epochs=50,
            opt_eval_interval=25,
            opt_trials=3,
            final_epochs=50,
            final_eval_interval=25
        )
    ),
    DemoMode.FULL: DemoConfig(
        mode=DemoMode.FULL,
        experiment_settings=ExperimentSettings(
            baseline_epochs=200,
            baseline_eval_interval=50,
            opt_epochs=300,
            opt_eval_interval=50,
            opt_trials=20,
            final_epochs=400,
            final_eval_interval=50
        )
    ),
    DemoMode.INTERACTIVE: DemoConfig(
        mode=DemoMode.INTERACTIVE,
        interactive=True,
        experiment_settings=ExperimentSettings(
            baseline_epochs=200,
            baseline_eval_interval=50,
            opt_epochs=300,
            opt_eval_interval=50,
            opt_trials=20,
            final_epochs=400,
            final_eval_interval=50
        )
    )
}

def get_demo_config(mode: DemoMode, interactive: bool = False) -> DemoConfig:
    """Get a demo configuration based on mode and interactivity."""
    config = DEMO_CONFIGS.get(mode, DEMO_CONFIGS[DemoMode.FULL])
    if interactive:
        config.interactive = True
    return config

# Configuration factory for different algorithms
class AlgorithmConfigFactory:
    """Factory for creating algorithm-specific configurations."""
    
    @staticmethod
    def get_search_space_path(algorithm_class: str) -> str:
        """Get the appropriate search space path for an algorithm."""
        if "hrm" in algorithm_class.lower() and "hrem" not in algorithm_class.lower():
            return SEARCH_SPACE_PATHS["hrm"]
        elif "hrem" in algorithm_class.lower():
            return SEARCH_SPACE_PATHS["hrem"]
        else:
            # Default to HREM search space for other models
            return SEARCH_SPACE_PATHS["hrem"]
    
    @staticmethod
    def create_optimization_config(
        algorithm_class: str,
        model_name: str,
        opt_trials: int = 20,
        storage_path: str = None
    ) -> Dict[str, Any]:
        """Create optimization configuration for a specific algorithm."""
        config = {
            "n_trials": opt_trials,
            "n_jobs": 1,
            "storage": storage_path or STORAGE_PATHS["default_optimization_db"],
            "n_final_runs": 1,
            "model_to_optimize": {
                "name": f"{model_name}_best",
                "algorithm_class": algorithm_class,
                "base_arch_config": AlgorithmConfigFactory._get_base_arch_config(algorithm_class)
            }
        }
        
        # Add search space if needed
        search_space_path = AlgorithmConfigFactory.get_search_space_path(algorithm_class)
        if search_space_path:
            config["search_space"] = {"path": search_space_path}
            
        return config
    
    @staticmethod
    def _get_base_arch_config(algorithm_class: str) -> str:
        """Get the base architecture config for an algorithm."""
        if "hrm" in algorithm_class.lower() and "hrem" not in algorithm_class.lower():
            return "hrm_v1"
        elif "enhanced" in algorithm_class.lower() and "hrem" in algorithm_class.lower():
            return "enhanced_hrem_v1"
        elif "hrem" in algorithm_class.lower():
            return "hrem_v1"
        else:
            # Default to hrem_v1 for unknown algorithms
            return "hrem_v1"