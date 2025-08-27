"""Configuration classes for the Scientific Discovery Engine."""

from typing import Dict, Any, List, Optional, Literal
from enum import Enum
from dataclasses import dataclass
from hrm_system.config import DataConfig, TrainingConfig, ModelConfig

class ChallengeLevel(Enum):
    """Enumeration of challenge difficulty levels."""
    BEGINNER = "BEGINNER"
    INTERMEDIATE = "INTERMEDIATE"
    ADVANCED = "ADVANCED"

@dataclass
class ChallengeConfig:
    """Configuration for a scientific challenge."""
    name: str                    # Descriptive challenge name
    id: str                      # Unique identifier
    description: str             # Scientific context
    dataset: DataConfig          # Challenge dataset
    difficulty: ChallengeLevel   # Complexity indicator
    scientific_question: str     # Core research question
    hypothesis_space: List[str]  # Expected algorithm behaviors

@dataclass
class AlgorithmConfig:
    """Configuration for an algorithm to be compared."""
    name: str                    # Algorithm name
    algorithm_class: str         # Implementation class
    theoretical_advantages: List[str]  # Expected strengths
    theoretical_limitations: List[str] # Known weaknesses
    search_space: Dict[str, Any] # Optimization parameters
    complexity_profile: Optional[Dict[str, Any]] = None  # Resource requirements

@dataclass
class PatienceBudget:
    """User patience budget for exploration."""
    level: Literal["low", "medium", "high"]  # Preset patience levels
    custom_seconds: Optional[int] = None     # Custom time limit
    extension_threshold: float = 0.7         # Threshold for automatic extension

@dataclass
class DiscoveryConfig:
    """Complete configuration for a scientific discovery session."""
    challenge: ChallengeConfig
    algorithms: List[AlgorithmConfig]
    patience_management: Dict[str, Any]
    insights: Dict[str, Any]