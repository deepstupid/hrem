"""Enhanced configuration management for the HRM/HREM demo system with full control over all parameters.

This module provides comprehensive configuration management for the demo system,
allowing fine-grained control over all aspects of the demo execution including:
- Demo modes (lightning, adaptive, comprehensive)
- Loop control parameters (epochs, trials, generations)
- Instrumentation settings (metrics collection, export)
- Model selection and optimization parameters
"""

from typing import Dict, Any, List, Optional, Union
from enum import Enum
import yaml
from pydantic import BaseModel, Field
from hrm_system.config import (
    DataConfig,
    TrainingConfig,
    OptimizationConfig,
    ModelConfig,
    RunConfig,
    ExperimentConfig,
    EvaluationConfig
)
from demo_models import get_model_configs, get_model_search_space
from dataset_manager import dataset_manager
from demo_model_runner import get_dataset_config
from rich.console import Console

console = Console()

class DemoMode(str, Enum):
    """Enumeration of demo modes.
    
    Each mode provides a different balance of speed vs. detail:
    - LIGHTNING: Ultra-fast demo with minimal iterations
    - ADAPTIVE: Time-adaptive demo that adjusts based on available time
    - COMPREHENSIVE: Full-featured demo with detailed analysis
    """
    LIGHTNING = "lightning"  # Ultra-fast demo
    ADAPTIVE = "adaptive"    # Time-adaptive demo
    COMPREHENSIVE = "comprehensive"  # Full-featured demo

class LoopControlConfig(BaseModel):
    """Configuration for controlling loops, iterations, and epochs.
    
    This configuration controls the execution parameters for training loops,
    optimization trials, and other iterative processes in the demo.
    """
    max_epochs: int = Field(default=1, description="Maximum epochs per training run", ge=1)
    eval_interval: int = Field(default=1, description="Evaluation interval in epochs", ge=1)
    max_trials: int = Field(default=2, description="Maximum trials for optimization", ge=1)
    n_jobs: int = Field(default=1, description="Number of parallel jobs for optimization", ge=1)
    n_final_runs: int = Field(default=1, description="Number of final evaluation runs", ge=1)
    max_generations: int = Field(default=1, description="Maximum generations for evolutionary algorithms", ge=1)
    batch_size: int = Field(default=384, description="Batch size for training", ge=1)

class InstrumentationConfig(BaseModel):
    """Configuration for instrumentation and metrics collection.
    
    This configuration controls what metrics are collected during the demo
    and how they are processed and exported.
    """
    collect_timing: bool = Field(default=True, description="Whether to collect timing metrics")
    collect_detailed_metrics: bool = Field(default=True, description="Whether to collect detailed metrics")
    log_level: str = Field(default="INFO", description="Logging level")
    export_metrics: bool = Field(default=True, description="Whether to export metrics to file")
    display_progress: bool = Field(default=True, description="Whether to display progress bars")

class EnhancedDemoConfig(BaseModel):
    """Enhanced configuration for the demo system.
    
    This is the main configuration class that controls all aspects of the demo execution.
    It combines settings for demo mode, loop control, instrumentation, and model parameters.
    """
    demo_mode: DemoMode = DemoMode.LIGHTNING
    smoke_test: bool = True
    loop_control: LoopControlConfig = LoopControlConfig()
    instrumentation: InstrumentationConfig = InstrumentationConfig()
    study_name: str = "enhanced_demo"
    patience_level: str = "low"
    models: List[str] = ["HRM", "HREM"]
    dataset: str = "synthetic-reverse"
    num_aug: int = 0

class EnhancedConfigManager:
    """Enhanced configuration manager with full control over demo parameters.
    
    This manager provides methods for creating experiment configurations,
    adapting settings based on user preferences, and loading/saving configurations.
    """
    
    @staticmethod
    def get_model_to_optimize(model_configs: List[ModelConfig]) -> tuple:
        """Get the best model to optimize from a list of model configurations.
        
        This method intelligently selects the best model to optimize based on:
        1. Preference for HREM models (if available)
        2. Availability of search space definitions
        3. Fallback to the first available model
        
        Args:
            model_configs: List of model configurations to evaluate
            
        Returns:
            Tuple of (model_to_optimize, search_space) or (None, None) if no suitable model found
        """
        # Try to find a model with a search space, preferring HREM if available
        model_to_optimize = None
        search_space = None
        
        if model_configs:
            # First try to find HREM model
            for model_config in model_configs:
                if "HREM" in model_config.name:
                    search_space = get_model_search_space(model_config.name)
                    if search_space:
                        model_to_optimize = model_config
                        break
            
            # If no HREM with search space, try any model with search space
            if not model_to_optimize:
                for model_config in model_configs:
                    search_space = get_model_search_space(model_config.name)
                    if search_space:
                        model_to_optimize = model_config
                        break
            
            # If still no model with search space, use the first model
            if not model_to_optimize:
                model_to_optimize = model_configs[0]
                search_space = get_model_search_space(model_to_optimize.name)
        
        return model_to_optimize, search_space
    
    @staticmethod
    def create_experiment_config(demo_config: EnhancedDemoConfig) -> ExperimentConfig:
        """Create a full experiment configuration from demo config.
        
        This method converts the enhanced demo configuration into a complete
        experiment configuration that can be used by the demo runners.
        
        Args:
            demo_config: Enhanced demo configuration
            
        Returns:
            Complete experiment configuration
        """
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
        model_to_optimize, search_space = EnhancedConfigManager.get_model_to_optimize(model_configs)
        
        # Set optimization config if we have a model to optimize
        if model_to_optimize and search_space:
            opt_config.model_to_optimize = model_to_optimize
            opt_config.search_space = search_space
        
        # Determine mode based on demo mode and whether we have optimization
        mode = "evaluate"
        if demo_config.loop_control.max_trials > 0 and model_to_optimize and search_space:
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
        """Get adaptive configuration based on user preferences.
        
        This method creates a configuration that adapts to the user's preferences
        for patience, hardware capability, and desired level of detail.
        
        Args:
            user_patience: User's patience level (low/medium/high)
            hardware_capability: Hardware capability (low/medium/high)
            desired_detail: Desired detail level (basic/adaptive/comprehensive)
            
        Returns:
            Adaptive enhanced demo configuration
        """
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
        """Load configuration from a YAML file.
        
        Args:
            config_path: Path to the YAML configuration file
            
        Returns:
            Loaded enhanced demo configuration
            
        Raises:
            Exception: If there's an error loading the configuration
        """
        try:
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
            return EnhancedDemoConfig(**config_dict)
        except Exception as e:
            console.print(f"[red]Error loading config from {config_path}: {str(e)}[/red]")
            return EnhancedDemoConfig()

    @staticmethod
    def save_to_file(config: EnhancedDemoConfig, config_path: str):
        """Save configuration to a YAML file.
        
        Args:
            config: Configuration to save
            config_path: Path to save the YAML configuration file
            
        Raises:
            Exception: If there's an error saving the configuration
        """
        try:
            with open(config_path, 'w') as f:
                yaml.dump(config.dict(), f, default_flow_style=False)
        except Exception as e:
            console.print(f"[red]Error saving config to {config_path}: {str(e)}[/red]")