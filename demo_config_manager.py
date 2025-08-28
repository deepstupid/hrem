"""Enhanced configuration management for the HRM/HREM demo system with full control."""

from typing import Dict, Any, List, Tuple, Optional
import yaml
import os
from enum import Enum
from hrm_system.config import (
    DataConfig,
    TrainingConfig,
    OptimizationConfig,
    ModelConfig,
    ExperimentConfig,
)
from demo_models import get_model_configs, get_model_search_space
from dataset_manager import dataset_manager
from demo_model_runner import get_dataset_config
from rich.console import Console
from demo_enhanced_config import EnhancedDemoConfig, DemoMode

console = Console()

class PatienceLevel(Enum):
    """Enumeration of demo patience levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class EnhancedConfigManager:
    """Enhanced centralized configuration manager for the HRM/HREM demo system."""
    
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
        
    @staticmethod
    def create_experiment_config(demo_config: EnhancedDemoConfig) -> ExperimentConfig:
        """Create a full experiment configuration from demo config."""
        from hrm_system.config import RunConfig, EvaluationConfig
        
        # Create run config
        run_config = RunConfig(
            smoke_test=demo_config.smoke_test,
            study_name=demo_config.study_name
        )
        
        # Get dataset config
        try:
            data_config = get_dataset_config(
                demo_config.dataset, 
                demo_config.smoke_test, 
                demo_config.num_aug
            )
        except Exception as e:
            console.print(f"[yellow]Warning: {str(e)}. Falling back to synthetic dataset.[/yellow]")
            data_config = get_dataset_config("synthetic", demo_config.smoke_test, demo_config.num_aug)
        
        # Create training config based on loop control
        training_config = TrainingConfig(
            epochs=demo_config.loop_control.max_epochs,
            eval_interval=demo_config.loop_control.eval_interval,
            global_batch_size=demo_config.loop_control.batch_size,
            smoke_test=demo_config.smoke_test
        )
        
        # Create model configs
        model_configs = get_model_configs(demo_config.models)
        
        # Create evaluation config
        eval_config_dict = {"n_runs": demo_config.loop_control.n_final_runs}
        for i, model_config in enumerate(model_configs):
            eval_config_dict[f"model_{chr(ord('a') + i)}"] = model_config
        eval_config = EvaluationConfig(**eval_config_dict)
        
        # Create optimization config
        opt_config = OptimizationConfig(
            n_trials=demo_config.loop_control.max_trials,
            n_jobs=demo_config.loop_control.n_jobs,
            n_final_runs=demo_config.loop_control.n_final_runs
        )
        
        # Add search space to the model to be optimized if applicable
        if model_configs:
            # For simplicity, we assume the last model in the list is the one to optimize
            model_to_optimize = model_configs[-1]
            search_space = get_model_search_space(model_to_optimize.name)
            if search_space:
                opt_config.model_to_optimize = model_to_optimize
                opt_config.search_space = search_space
        
        # Determine mode based on demo mode and trials
        mode = "evaluate"
        if demo_config.loop_control.max_trials > 0 and demo_config.demo_mode != DemoMode.LIGHTNING:
            mode = "optimize"
            
        return ExperimentConfig(
            mode=mode,
            run_config=run_config,
            data_config=data_config,
            training_config=training_config,
            evaluation_config=eval_config,
            optimization_config=opt_config
        )
    
    @staticmethod
    def get_adaptive_config(user_patience: str = "low", 
                           hardware_capability: str = "low",
                           desired_detail: str = "basic") -> EnhancedDemoConfig:
        """Get adaptive configuration based on user preferences."""
        # Base config
        config = EnhancedDemoConfig()
        
        # Adjust based on user patience
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
        else:  # low patience
            config.loop_control.max_epochs = 1
            config.loop_control.max_trials = 1
            config.loop_control.n_final_runs = 1
            config.patience_level = "low"
            
        # Adjust based on hardware capability
        if hardware_capability == "high":
            config.loop_control.batch_size = 1024
            config.loop_control.n_jobs = 4
        elif hardware_capability == "medium":
            config.loop_control.batch_size = 512
            config.loop_control.n_jobs = 2
        else:  # low hardware
            config.loop_control.batch_size = 256
            config.loop_control.n_jobs = 1
            
        # Adjust based on desired detail
        if desired_detail == "comprehensive":
            config.instrumentation.collect_detailed_metrics = True
            config.instrumentation.export_metrics = True
            config.demo_mode = DemoMode.COMPREHENSIVE
        elif desired_detail == "adaptive":
            config.instrumentation.collect_detailed_metrics = True
            config.instrumentation.export_metrics = True
            config.demo_mode = DemoMode.ADAPTIVE
        else:  # basic
            config.instrumentation.collect_detailed_metrics = False
            config.instrumentation.export_metrics = False
            config.demo_mode = DemoMode.LIGHTNING
            
        return config

    @staticmethod
    def load_from_file(config_path: str) -> EnhancedDemoConfig:
        """Load configuration from a YAML file."""
        try:
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
            # Handle enum conversion
            if 'demo_mode' in config_dict:
                config_dict['demo_mode'] = DemoMode(config_dict['demo_mode'])
            return EnhancedDemoConfig(**config_dict)
        except Exception as e:
            console.print(f"[red]Error loading config from {config_path}: {str(e)}[/red]")
            return EnhancedDemoConfig()

    @staticmethod
    def save_to_file(config: EnhancedDemoConfig, config_path: str):
        """Save configuration to a YAML file."""
        try:
            config_dict = config.dict()
            # Convert enum to string for serialization
            if 'demo_mode' in config_dict:
                config_dict['demo_mode'] = config_dict['demo_mode'].value
            with open(config_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False)
        except Exception as e:
            console.print(f"[red]Error saving config to {config_path}: {str(e)}[/red]")