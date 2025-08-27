import os
from typing import Literal, Optional, List, Dict, Any, Callable
from pydantic import BaseModel, Field, model_validator, ConfigDict

# A logger callback is a function that takes a string message.
# This will be used to stream logs from the core system to the TUI.
LoggerCallback = Callable[[str], None]

class DataConfig(BaseModel):
    """Configuration for the dataset."""
    dataset: Literal["arc", "sudoku", "maze", "synthetic", "synthetic-reverse", "synthetic-sort", "synthetic-parity", "synthetic-duplicate"] = "arc"
    num_aug: int = Field(0, description="Number of augmentations for the dataset.")
    synthetic_task: Literal["copy", "reverse", "sort", "parity", "duplicate"] = Field("copy", description="Task type for synthetic dataset.")
    dataset_path: Optional[str] = Field(None, description="Path to the dataset. If None, it will be derived based on other settings.")

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
    training_config_name: Optional[str] = Field(None, description="The name of the training configuration file.")
    training_config: Optional['TrainingConfig'] = Field(None, exclude=True, description="The loaded training configuration.")

class TrainingConfig(BaseModel):
    """Configuration for the training process."""
    epochs: int = 10  # Drastically reduced for demo purposes
    eval_interval: int = 5
    global_batch_size: int = 384
    lr: float = 1e-4
    lr_min_ratio: float = 0.1
    lr_warmup_steps: int = 10
    puzzle_emb_lr: float = 1e-4
    weight_decay: float = 1.0
    puzzle_emb_weight_decay: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.95
    optimizer: Literal["Adam", "AdamW"] = Field("AdamW", description="The optimizer to use for training.")
    optimizer_eps: float = Field(1e-8, description="The epsilon value for the optimizer.")
    seed: int = 0
    checkpoint_every_eval: bool = False
    eval_save_outputs: List[str] = Field(default_factory=list)
    smoke_test: bool = False
    num_workers: int = Field(1, description="Number of workers for the dataloader.")
    prefetch_factor: int = Field(8, description="Prefetch factor for the dataloader.")
    lr_schedule: Literal["cosine", "linear"] = Field("cosine", description="Learning rate schedule.")
    use_amp: bool = Field(False, description="Whether to use Automatic Mixed Precision.")

class EvaluationConfig(BaseModel):
    """Configuration for the evaluation mode."""
    n_runs: int = Field(1, description="Number of runs for statistical significance.")
    # The evaluation mode compares up to five models.
    model_a: Optional[ModelConfig] = Field(None, description="The first model to compare.")
    model_b: Optional[ModelConfig] = Field(None, description="The second model to compare.")
    model_c: Optional[ModelConfig] = Field(None, description="The third model to compare.")
    model_d: Optional[ModelConfig] = Field(None, description="The fourth model to compare.")
    model_e: Optional[ModelConfig] = Field(None, description="The fifth model to compare.")

    def get_models(self) -> List[ModelConfig]:
        """Returns a list of the configured models, filtering out None values."""
        models = [self.model_a, self.model_b, self.model_c, self.model_d, self.model_e]
        return [model for model in models if model is not None]

    def set_models(self, models: List[ModelConfig]):
        """Sets the models from a list, populating model_a, model_b, etc."""
        self.clear_models()
        model_fields = ["model_a", "model_b", "model_c", "model_d", "model_e"]
        for i, model in enumerate(models):
            if i < len(model_fields):
                setattr(self, model_fields[i], model)

    def clear_models(self):
        """Resets all model fields to None."""
        self.model_a = None
        self.model_b = None
        self.model_c = None
        self.model_d = None
        self.model_e = None

class OptimizationConfig(BaseModel):
    """Configuration for the hyperparameter optimization mode."""
    n_trials: int = Field(3, description="Number of optimization trials.")  # Reduced for demo
    n_jobs: int = Field(1, description="Number of parallel jobs for Optuna.")
    storage: str = Field("sqlite:///experiments/optuna_hrem.db", description="Optuna storage URL.")
    search_space: Dict[str, Any] = Field(default_factory=dict, description="Hyperparameter search space for Optuna.")
    n_final_runs: int = Field(1, description="Number of final comparison runs for statistical significance.")
    baseline_model: ModelConfig = Field(default_factory=lambda: ModelConfig(name="HRM", algorithm_class="hrm_system.algorithms.hrm.HRMAlgorithm", base_arch_config="hrm_v1"), description="The baseline model to compare against.")
    model_to_optimize: ModelConfig = Field(default_factory=lambda: ModelConfig(name="HREM_best", algorithm_class="hrm_system.algorithms.hrem.HREMAlgorithm", base_arch_config="hrem_v1"), description="The model to be optimized.")
    patience_level: str = Field("medium", description="User patience level for demo.")


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