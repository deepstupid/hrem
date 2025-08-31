from pydantic import BaseModel, Field
from typing import List, Optional


class DatasetSchema(BaseModel):
    """Pydantic schema for a dataset entry."""
    dataset: str


class ChallengeSchema(BaseModel):
    """Pydantic schema for a single challenge configuration."""
    name: str
    id: str
    description: str
    difficulty: str
    hardware: str
    duration: str
    dataset: DatasetSchema
    models: List[str]
    patience_level: str
    optimization: bool
    scientific_question: Optional[str] = None
    hypothesis_space: Optional[List[str]] = None


class ChallengeConfigSchema(BaseModel):
    """Pydantic schema for the main challenge configuration file."""
    challenges: List[ChallengeSchema]


class ModelConfigSchema(BaseModel):
    """Pydantic schema for a model's configuration file."""
    name: str
    algorithm_class: str
    base_arch_config: Optional[str] = None
    search_space_config: Optional[str] = None
    theoretical_advantages: List[str]
    theoretical_limitations: List[str]


class InsightConfigSchema(BaseModel):
    """Pydantic schema for validating the insight generator configuration."""
    statistical_significance_threshold: float = Field(..., ge=0, le=1)

    efficiency_speedup_threshold: float = Field(..., gt=0)
    efficiency_high_speedup_threshold: float = Field(..., gt=0)
    efficiency_potential_boost: float = Field(..., ge=0)
    efficiency_surprising_potential: float = Field(..., ge=0, le=1)
    efficiency_expected_potential: float = Field(..., ge=0, le=1)

    scalability_performance_gap_threshold: float = Field(..., gt=0)
    scalability_high_performance_gap_threshold: float = Field(..., gt=0)
    scalability_potential_boost: float = Field(..., ge=0)
    scalability_surprising_potential_hard: float = Field(..., ge=0, le=1)
    scalability_surprising_potential_easy: float = Field(..., ge=0, le=1)
    scalability_expected_potential_hard: float = Field(..., ge=0, le=1)
    scalability_expected_potential_easy: float = Field(..., ge=0, le=1)

    convergence_threshold_percent: float = Field(..., gt=0, lt=1)
    convergence_speed_ratio_threshold: float = Field(..., gt=0)
    convergence_potential_surprising: float = Field(..., ge=0, le=1)
    convergence_potential_expected: float = Field(..., ge=0, le=1)

    robustness_variance_threshold: float = Field(..., gt=0)
    robustness_high_variance_threshold: float = Field(..., gt=0)
    robustness_potential_boost: float = Field(..., ge=0)
    robustness_surprising_potential: float = Field(..., ge=0, le=1)
    robustness_expected_potential: float = Field(..., ge=0, le=1)

    generalization_drop_off_threshold: float = Field(..., ge=0)
    generalization_potential: float = Field(..., ge=0, le=1)

    adaptability_improvement_threshold: float = Field(..., ge=0)
    adaptability_potential: float = Field(..., ge=0, le=1)

    meta_insight_threshold: int = Field(..., gt=0)

    insight_pipeline: List[str] = Field(...)
