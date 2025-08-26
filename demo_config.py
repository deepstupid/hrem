"""Configuration management for the HRM/HREM demo system."""

from typing import Dict, Any
from dataclasses import dataclass, field
from enum import Enum
from omegaconf import OmegaConf

# Load the unified demo configuration
try:
    DEMO_CONFIG = OmegaConf.load("config/demo_config.yaml")
except FileNotFoundError:
    raise RuntimeError("The main demo configuration file (config/demo_config.yaml) was not found.")

class DemoMode(Enum):
    """Enumeration of demo modes."""
    FAST = "fast"
    FULL = "full"

@dataclass
class DemoConfig:
    """Complete configuration for the demo system, loaded from YAML."""
    mode: DemoMode
    interactive: bool = False
    settings: Dict[str, Any] = field(default_factory=dict)
    ui: Dict[str, Any] = field(default_factory=dict)
    paths: Dict[str, Any] = field(default_factory=dict)
    models: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize based on mode."""
        mode_str = self.mode.value
        self.settings = DEMO_CONFIG.experiment_settings[f"{mode_str}_mode"]
        self.ui = DEMO_CONFIG.ui
        self.paths = DEMO_CONFIG.paths
        self.models = DEMO_CONFIG.models

def get_demo_config(mode: DemoMode, interactive: bool = False) -> DemoConfig:
    """Get a demo configuration based on mode and interactivity."""
    return DemoConfig(mode=mode, interactive=interactive)

class AlgorithmConfigFactory:
    """Factory for creating algorithm-specific configurations from the loaded YAML."""
    
    @staticmethod
    def create_optimization_config(
        model_name: str,
        opt_trials: int,
        storage_path: str = None
    ) -> Dict[str, Any]:
        """Create optimization configuration for a specific algorithm."""
        model_config = DEMO_CONFIG.models.get(model_name)
        if not model_config:
            raise ValueError(f"Model '{model_name}' not found in demo configuration.")

        # Use the provided storage path or fall back to the default from the config
        final_storage_path = storage_path or DEMO_CONFIG.paths.default_optimization_db

        config = {
            "n_trials": opt_trials,
            "n_jobs": 1,
            "storage": final_storage_path,
            "n_final_runs": 1,
            "model_to_optimize": {
                "name": f"{model_name}_best",
                "algorithm_class": model_config.algorithm_class,
                "base_arch_config": model_config.base_arch_config
            }
        }
        
        # Add search space if it exists for the model
        if "search_space" in model_config and model_config.search_space:
            # OmegaConf converts it to a specific type, so we convert it back to a plain dict
            config["search_space"] = OmegaConf.to_container(model_config.search_space, resolve=True)
            
        return config