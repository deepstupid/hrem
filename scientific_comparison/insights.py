"""Data structures for scientific insights."""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple

class InsightType(Enum):
    """Enumeration of scientific insight types."""
    EFFICIENCY = "efficiency"
    ROBUSTNESS = "robustness"
    SCALABILITY = "scalability"
    GENERALIZATION = "generalization"
    CONVERGENCE = "convergence"
    ADAPTABILITY = "adaptability"
    META = "meta"
    FAILURE = "failure"
    HYPOTHESIS = "hypothesis"

@dataclass
class Evidence:
    """Represents a piece of evidence supporting an insight."""
    metric_name: str
    metric_value: Any
    comparison_metric_value: Optional[Any] = None
    p_value: Optional[float] = None
    effect_size: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    description: Optional[str] = None

@dataclass
class ScientificInsight:
    """A structured scientific insight generated from algorithm comparison."""
    type: InsightType
    confidence: float
    implications: List[str]
    discovery_potential: float
    evidence: List[Evidence] = field(default_factory=list)
    title: str = ""
    summary: str = ""
    hypothesis: Optional[str] = None
    causal_attribution: Optional[str] = None
    recommendations: List[str] = field(default_factory=list)


    def __post_init__(self):
        if not self.title:
            self.title = f"{self.type.name.replace('_', ' ').title()} Insight"
        if not self.summary:
            if self.implications:
                self.summary = self.implications[0]
