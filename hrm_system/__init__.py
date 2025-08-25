"""
HRM System Core Library

This package contains the core logic for running HRM experiments, including
evaluation and optimization. It is designed to be a headless library that can be
used by various clients, such as a command-line interface or a text-based UI.
"""

from .config import (
    ExperimentConfig,
    RunConfig,
    DataConfig,
    ModelConfig,
    TrainingConfig,
    EvaluationConfig,
    OptimizationConfig,
    HREMParams
)
from .evaluation import run_evaluation
from .optimization import run_optimization

__all__ = [
    "ExperimentConfig",
    "RunConfig",
    "DataConfig",
    "ModelConfig",
    "TrainingConfig",
    "EvaluationConfig",
    "OptimizationConfig",
    "HREMParams",
    "run_evaluation",
    "run_optimization",
]
