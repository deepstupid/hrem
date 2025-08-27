"""Enhanced configuration management for the HRM/HREM demo system.

This module provides comprehensive configuration management for the demo system,
allowing fine-grained control over all aspects of the demo execution.
"""

import os
import yaml
from typing import Dict, Any, List, Optional
from enum import Enum
from pydantic import BaseModel, Field
from rich.console import Console

console = Console()

class DemoMode(str, Enum):
    """Enumeration of demo modes."""
    LIGHTNING = "lightning"
    ADAPTIVE = "adaptive"
    COMPREHENSIVE = "comprehensive"

class LoopControlConfig(BaseModel):
    """Configuration for controlling loops and iterations."""
    max_epochs: int = Field(default=1, description="Maximum epochs per training run", ge=1)
    eval_interval: int = Field(default=1, description="Evaluation interval in epochs", ge=1)
    max_trials: int = Field(default=1, description="Maximum trials for optimization", ge=1)
    n_jobs: int = Field(default=1, description="Number of parallel jobs for optimization", ge=1)
    n_final_runs: int = Field(default=1, description="Number of final evaluation runs", ge=1)
    batch_size: int = Field(default=256, description="Batch size for training", ge=1)
    time_limit: int = Field(default=60, description="Time limit for demo in seconds", ge=1)

class InstrumentationConfig(BaseModel):
    """Configuration for instrumentation and metrics collection."""
    collect_timing: bool = Field(default=True, description="Whether to collect timing metrics")
    collect_detailed_metrics: bool = Field(default=True, description="Whether to collect detailed metrics")
    export_metrics: bool = Field(default=False, description="Whether to export metrics to file")
    display_progress: bool = Field(default=True, description="Whether to display progress bars")

class DemoConfig(BaseModel):
    """Enhanced configuration for the demo system."""
    demo_mode: DemoMode = DemoMode.LIGHTNING
    smoke_test: bool = True
    loop_control: LoopControlConfig = LoopControlConfig()
    instrumentation: InstrumentationConfig = InstrumentationConfig()
    study_name: str = "demo"
    patience_level: str = "low"
    models: List[str] = ["HRM", "HREM"]
    dataset: str = "synthetic"
    task: str = "reverse"
    num_aug: int = 0

class DemoConfigManager:
    """Enhanced configuration manager with full control over demo parameters."""
    
    @staticmethod
    def load_demo_settings(config_path: str = "config/demo_settings.yaml") -> Dict[str, Any]:
        """Load demo settings from YAML file."""
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    return yaml.safe_load(f)
            else:
                console.print(f"[yellow]Warning: Config file {config_path} not found. Using defaults.[/yellow]")
                return {}
        except Exception as e:
            console.print(f"[red]Error loading config from {config_path}: {str(e)}[/red]")
            return {}

    @staticmethod
    def get_mode_config(mode: DemoMode, settings: Dict[str, Any]) -> LoopControlConfig:
        """Get configuration for a specific demo mode."""
        mode_settings = settings.get("demo_modes", {}).get(mode.value, {})
        
        return LoopControlConfig(
            max_epochs=mode_settings.get("max_epochs", 1),
            max_trials=mode_settings.get("max_trials", 1),
            eval_interval=mode_settings.get("eval_interval", 1),
            batch_size=mode_settings.get("batch_size", 256),
            n_jobs=mode_settings.get("n_jobs", 1),
            n_final_runs=mode_settings.get("n_final_runs", 1),
            time_limit=mode_settings.get("time_limit", 60)
        )

    @staticmethod
    def create_config(
        mode: DemoMode = DemoMode.LIGHTNING,
        models: List[str] = None,
        dataset: str = "synthetic",
        task: str = "reverse",
        smoke_test: bool = True,
        export_metrics: bool = False
    ) -> DemoConfig:
        """Create a demo configuration with sensible defaults."""
        # Load settings
        settings = DemoConfigManager.load_demo_settings()
        
        # Get mode-specific configuration
        loop_control = DemoConfigManager.get_mode_config(mode, settings)
        
        # Set instrumentation based on export flag
        instrumentation = InstrumentationConfig(
            export_metrics=export_metrics,
            collect_timing=True,
            collect_detailed_metrics=(mode != DemoMode.LIGHTNING),
            display_progress=True
        )
        
        # Use provided models or defaults
        if models is None:
            models = settings.get("models", {}).get("default_comparison", ["HRM", "HREM"])
        
        return DemoConfig(
            demo_mode=mode,
            smoke_test=smoke_test,
            loop_control=loop_control,
            instrumentation=instrumentation,
            study_name=f"{mode.value}_demo",
            patience_level=str(loop_control.time_limit),
            models=models,
            dataset=dataset,
            task=task,
            num_aug=0
        )

    @staticmethod
    def load_from_file(config_path: str) -> DemoConfig:
        """Load configuration from a YAML file."""
        try:
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
            return DemoConfig(**config_dict)
        except Exception as e:
            console.print(f"[red]Error loading config from {config_path}: {str(e)}[/red]")
            return DemoConfig()

    @staticmethod
    def save_to_file(config: DemoConfig, config_path: str):
        """Save configuration to a YAML file."""
        try:
            with open(config_path, 'w') as f:
                yaml.dump(config.dict(), f, default_flow_style=False)
        except Exception as e:
            console.print(f"[red]Error saving config to {config_path}: {str(e)}[/red]")