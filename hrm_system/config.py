import os
from typing import Literal, Optional, List, Dict, Any, Callable
from pydantic import BaseModel, Field, model_validator, ConfigDict

# A logger callback is a function that takes a string message.
# This will be used to stream logs from the core system to the TUI.
LoggerCallback = Callable[[str], None]

class DataConfig(BaseModel):
    """Configuration for the dataset."""
    dataset: Literal["arc", "sudoku", "maze", "synthetic"] = "arc"
    num_aug: int = Field(0, description="Number of augmentations for the dataset.")
    synthetic_task: Literal["copy", "reverse"] = Field("copy", description="Task type for synthetic dataset.")
    path: Optional[str] = Field(None, description="Path to the dataset. If None, it will be derived based on other settings.")

class HREMParams(BaseModel):
    """Hyperparameters for the HREM model, based on the search space."""
    model_config = ConfigDict(extra="allow")

    m_loc: int = 128
    d_mem: int = 128
    top_k: int = 4
    H_layers: int = 2
    L_layers: int = 2
    H_cycles: int = 2
    L_cycles: int = 8
    hidden_size: int = 256

class ModelConfig(BaseModel):
    """Configuration for the model to be trained or evaluated."""
    name: str = "HREM"
    algorithm_class: str = Field("hrm_system.algorithms.hrem.HREMAlgorithm", description="The full import path to the algorithm class.")
    base_arch_config: str = Field("hrem_v1", description="Base architecture config file name (e.g., 'hrem_v1').")
    hrem_params: Optional[HREMParams] = Field(default_factory=HREMParams, description="HREM-specific hyperparameters.")
    arch_overrides: Dict[str, Any] = Field(default_factory=dict, description="Architecture overrides from the command line.")

class TrainingConfig(BaseModel):
    """Configuration for the training process."""
    epochs: int = 20000
    eval_interval: int = 2000
    global_batch_size: int = 384
    lr: float = 1e-4
    lr_min_ratio: float = 0.1
    lr_warmup_steps: int = 2000
    puzzle_emb_lr: float = 1e-4
    weight_decay: float = 1.0
    puzzle_emb_weight_decay: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.95
    seed: int = 0
    checkpoint_every_eval: bool = False
    eval_save_outputs: List[str] = Field(default_factory=list)
    smoke_test: bool = False

class EvaluationConfig(BaseModel):
    """Configuration for the evaluation mode."""
    n_runs: int = Field(1, description="Number of runs for statistical significance.")
    # The evaluation mode compares up to three models.
    model_a: Optional[ModelConfig] = Field(default_factory=lambda: ModelConfig(name="HRM", algorithm_class="hrm_system.algorithms.hrm.HRMAlgorithm", base_arch_config="hrm_v1"), description="The first model to compare (baseline).")
    model_b: Optional[ModelConfig] = Field(default_factory=lambda: ModelConfig(name="HREM", algorithm_class="hrm_system.algorithms.hrem.HREMAlgorithm", base_arch_config="hrem_v1"), description="The second model to compare.")
    model_c: Optional[ModelConfig] = Field(default_factory=lambda: ModelConfig(name="EnhancedHREM", algorithm_class="hrm_system.algorithms.enhanced_hrem.EnhancedHREMAlgorithm", base_arch_config="enhanced_hrem_v1"), description="The enhanced model to compare.")

class OptimizationConfig(BaseModel):
    """Configuration for the hyperparameter optimization mode."""
    n_trials: int = Field(10, description="Number of optimization trials.")
    n_jobs: int = Field(1, description="Number of parallel jobs for Optuna.")
    storage: str = Field("sqlite:///experiments/optuna_hrem.db", description="Optuna storage URL.")
    search_space: Dict[str, Any] = Field(default_factory=dict, description="Hyperparameter search space for Optuna.")
    n_final_runs: int = Field(1, description="Number of final comparison runs for statistical significance.")
    baseline_model: ModelConfig = Field(default_factory=lambda: ModelConfig(name="HRM", algorithm_class="hrm_system.algorithms.hrm.HRMAlgorithm", base_arch_config="hrm_v1"), description="The baseline model to compare against.")
    model_to_optimize: ModelConfig = Field(default_factory=lambda: ModelConfig(name="HREM_best", algorithm_class="hrm_system.algorithms.hrem.HREMAlgorithm", base_arch_config="hrem_v1"), description="The model to be optimized.")


class RunConfig(BaseModel):
    """High-level configuration for any type of run."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    project_name: str = "HRM_System"
    study_name: str = "hrem_experiment"
    smoke_test: bool = False
    output_dir: str = "experiments"
    run_name: Optional[str] = None
    log_path: Optional[str] = None
    logger_callback: Optional[LoggerCallback] = None


class ExperimentConfig(BaseModel):
    """
    The main configuration object that orchestrates the entire system.
    This object defines what mode to run in and provides the necessary
    configuration for that mode.
    """
    mode: Literal["evaluate", "optimize"] = "evaluate"
    run_config: RunConfig = Field(default_factory=RunConfig)
    data_config: DataConfig = Field(default_factory=DataConfig)
    training_config: TrainingConfig = Field(default_factory=TrainingConfig)
    evaluation_config: Optional[EvaluationConfig] = None
    optimization_config: Optional[OptimizationConfig] = None

    @model_validator(mode='after')
    def set_mode_specific_configs(self) -> 'ExperimentConfig':
        if self.mode == 'evaluate':
            if self.evaluation_config is None:
                self.evaluation_config = EvaluationConfig()
            self.optimization_config = None

        elif self.mode == 'optimize':
            if self.optimization_config is None:
                self.optimization_config = OptimizationConfig()
            self.evaluation_config = None
        return self
