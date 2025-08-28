"""Enhanced configuration management for the HRM/HREM demo system with full control."""

from typing import Dict, Any, List, Tuple, Optional
import yaml
import os
from enum import Enum
from pydantic import BaseModel, Field
from hrm_system.config import (
    DataConfig,
    TrainingConfig,
    OptimizationConfig,
    ModelConfig,
    ExperimentConfig,
    RunConfig,
    EvaluationConfig,
)
from dataset_manager import dataset_manager
from rich.console import Console

console = Console()


# --- Pydantic Models for Configuration ---

class DemoMode(str, Enum):
    """Enumeration of demo modes."""
    LIGHTNING = "lightning"
    ADAPTIVE = "adaptive"
    COMPREHENSIVE = "comprehensive"


class LoopControlConfig(BaseModel):
    """Configuration for controlling loops, iterations, and epochs."""
    max_epochs: int = Field(default=1, description="Maximum epochs per training run", ge=1)
    eval_interval: int = Field(default=1, description="Evaluation interval in epochs", ge=1)
    max_trials: int = Field(default=2, description="Maximum trials for optimization", ge=1)
    n_jobs: int = Field(default=1, description="Number of parallel jobs for optimization", ge=1)
    n_final_runs: int = Field(default=1, description="Number of final evaluation runs", ge=1)
    batch_size: int = Field(default=384, description="Batch size for training", ge=1)


class InstrumentationConfig(BaseModel):
    """Configuration for instrumentation and metrics collection."""
    collect_timing: bool = Field(default=True, description="Whether to collect timing metrics")
    collect_detailed_metrics: bool = Field(default=True, description="Whether to collect detailed metrics")
    log_level: str = Field(default="INFO", description="Logging level")
    export_metrics: bool = Field(default=True, description="Whether to export metrics to file")
    display_progress: bool = Field(default=True, description="Whether to display progress bars")


class DemoConfig(BaseModel):
    """Main configuration class that controls all aspects of the demo execution."""
    demo_mode: DemoMode = DemoMode.LIGHTNING
    smoke_test: bool = True
    loop_control: LoopControlConfig = LoopControlConfig()
    instrumentation: InstrumentationConfig = InstrumentationConfig()
    study_name: str = "enhanced_demo"
    patience_level: str = "low"
    models: List[str] = ["HRM", "HREM"]
    dataset: str = "synthetic-reverse"
    num_aug: int = 0
    task: str = "reverse"


# --- Model Registry ---

class ModelRegistry:
    """Registry for available models, loaded from YAML configuration."""

    def __init__(self, models_dir="config/models", search_dir="config/search"):
        self._models: Dict[str, ModelConfig] = {}
        self._search_spaces: Dict[str, Dict[str, Any]] = {}

        if not os.path.isdir(models_dir):
            return

        for model_file in os.listdir(models_dir):
            if model_file.endswith(".yaml"):
                model_path = os.path.join(models_dir, model_file)
                with open(model_path, 'r') as f:
                    model_data = yaml.safe_load(f)

                model_name = model_data["name"]
                model_config = ModelConfig(**model_data)
                self._models[model_name] = model_config

                if "search_space_config" in model_data:
                    search_space_file = f"{model_data['search_space_config']}.yaml"
                    search_path = os.path.join(search_dir, search_space_file)
                    if os.path.exists(search_path):
                        with open(search_path, 'r') as sf:
                            self._search_spaces[model_name] = yaml.safe_load(sf)

    def get_model(self, name: str) -> Optional[ModelConfig]:
        return self._models.get(name)

    def get_search_space(self, name: str) -> Optional[Dict[str, Any]]:
        return self._search_spaces.get(name)

    def get_models(self, names: List[str]) -> List[ModelConfig]:
        return [self._models[name] for name in names if name in self._models]

    def list_models(self) -> List[str]:
        return list(self._models.keys())


# Global model registry instance
model_registry = ModelRegistry()


# --- Enhanced Configuration Manager ---

class ConfigManager:
    """Enhanced, centralized configuration manager for the HRM/HREM demo system."""

    @staticmethod
    def get_model_to_optimize(model_configs: List[ModelConfig]) -> tuple:
        """Intelligently selects the best model to optimize."""
        model_to_optimize = None
        search_space = None

        if not model_configs:
            return None, None

        # Prefer HREM models with a search space
        for model_config in model_configs:
            if "HREM" in model_config.name:
                search_space = model_registry.get_search_space(model_config.name)
                if search_space:
                    model_to_optimize = model_config
                    break
        
        # Fallback to any model with a search space
        if not model_to_optimize:
            for model_config in model_configs:
                search_space = model_registry.get_search_space(model_config.name)
                if search_space:
                    model_to_optimize = model_config
                    break
        
        # Fallback to the first model if no search space found
        if not model_to_optimize:
            model_to_optimize = model_configs[0]
            search_space = model_registry.get_search_space(model_to_optimize.name)

        return model_to_optimize, search_space

    @staticmethod
    def get_dataset_config(dataset_name: str, smoke_test: bool, num_aug: int) -> DataConfig:
        """Gets dataset configuration with a fallback mechanism."""
        try:
            return dataset_manager.get_config(dataset_name, smoke_test, num_aug)
        except Exception as e:
            console.print(f"[yellow]Warning: {str(e)}. Falling back to synthetic dataset.[/yellow]")
            return dataset_manager.get_config("synthetic", smoke_test, num_aug)

    @staticmethod
    def create_experiment_config(demo_config: DemoConfig) -> ExperimentConfig:
        """Converts a demo configuration into a full experiment configuration."""
        run_config = RunConfig(smoke_test=demo_config.smoke_test, study_name=demo_config.study_name)
        data_config = ConfigManager.get_dataset_config(demo_config.dataset, demo_config.smoke_test, demo_config.num_aug)
        
        training_config = TrainingConfig(
            epochs=demo_config.loop_control.max_epochs,
            eval_interval=demo_config.loop_control.eval_interval,
            global_batch_size=demo_config.loop_control.batch_size,
            smoke_test=demo_config.smoke_test,
        )

        model_configs = model_registry.get_models(demo_config.models)
        
        eval_config_dict = {"n_runs": demo_config.loop_control.n_final_runs}
        for i, model_config in enumerate(model_configs):
            eval_config_dict[f"model_{chr(ord('a') + i)}"] = model_config
        eval_config = EvaluationConfig(**eval_config_dict)

        opt_config = OptimizationConfig(
            n_trials=demo_config.loop_control.max_trials,
            n_jobs=demo_config.loop_control.n_jobs,
            n_final_runs=demo_config.loop_control.n_final_runs,
        )

        model_to_optimize, search_space = ConfigManager.get_model_to_optimize(model_configs)
        if model_to_optimize and search_space:
            opt_config.model_to_optimize = model_to_optimize
            opt_config.search_space = search_space

        mode = "evaluate"
        if demo_config.loop_control.max_trials > 0 and model_to_optimize and search_space:
            mode = "optimize"

        return ExperimentConfig(
            mode=mode,
            run_config=run_config,
            data_config=data_config,
            training_config=training_config,
            evaluation_config=eval_config,
            optimization_config=opt_config,
        )

    @staticmethod
    def get_adaptive_config(
        user_patience: str = "low", hardware_capability: str = "low", desired_detail: str = "basic"
    ) -> DemoConfig:
        """Creates a configuration that adapts to user preferences."""
        config = DemoConfig()

        if user_patience == "high":
            config.loop_control.max_epochs = 10
            config.loop_control.max_trials = 5
            config.loop_control.n_final_runs = 3
            config.patience_level = "high"
        elif user_patience == "medium":
            config.loop_control.max_epochs = 5
            config.loop_control.max_trials = 3
            config.loop_control.n_final_runs = 2
            config.patience_level = "medium"

        if hardware_capability == "high":
            config.loop_control.batch_size = 1024
            config.loop_control.n_jobs = 4
        elif hardware_capability == "medium":
            config.loop_control.batch_size = 512
            config.loop_control.n_jobs = 2

        if desired_detail == "comprehensive":
            config.instrumentation.collect_detailed_metrics = True
            config.instrumentation.export_metrics = True
            config.demo_mode = DemoMode.COMPREHENSIVE
        elif desired_detail == "adaptive":
            config.instrumentation.collect_detailed_metrics = True
            config.instrumentation.export_metrics = True
            config.demo_mode = DemoMode.ADAPTIVE
        else:
            config.instrumentation.collect_detailed_metrics = False
            config.instrumentation.export_metrics = False
            config.demo_mode = DemoMode.LIGHTNING
            
        return config

    @staticmethod
    def load_from_file(config_path: str) -> DemoConfig:
        """Loads configuration from a YAML file."""
        try:
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
            return DemoConfig(**config_dict)
        except Exception as e:
            console.print(f"[red]Error loading config from {config_path}: {str(e)}[/red]")
            return DemoConfig()

    @staticmethod
    def save_to_file(config: DemoConfig, config_path: str):
        """Saves configuration to a YAML file."""
        try:
            with open(config_path, 'w') as f:
                yaml.dump(config.dict(), f, default_flow_style=False)
        except Exception as e:
            console.print(f"[red]Error saving config to {config_path}: {str(e)}[/red]")
