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

class PatienceLevel(Enum):
    """Enumeration of demo patience levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

@dataclass
class DemoConfig:
    """Complete configuration for the demo system, loaded from YAML."""
    patience: PatienceLevel
    interactive: bool = False
    settings: Dict[str, Any] = field(default_factory=dict)
    resource_settings: Dict[str, Any] = field(default_factory=dict)
    ui: Dict[str, Any] = field(default_factory=dict)
    paths: Dict[str, Any] = field(default_factory=dict)
    models: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize based on patience level."""
        patience_str = self.patience.value
        self.settings = DEMO_CONFIG.experiment_settings.patience_levels[patience_str]
        self.resource_settings = DEMO_CONFIG.experiment_settings.resource_profiles
        self.ui = DEMO_CONFIG.ui
        self.paths = DEMO_CONFIG.paths
        self.models = DEMO_CONFIG.models

def get_demo_config(patience: PatienceLevel, interactive: bool = False) -> DemoConfig:
    """Get a demo configuration based on patience level and interactivity."""
    return DemoConfig(patience=patience, interactive=interactive)