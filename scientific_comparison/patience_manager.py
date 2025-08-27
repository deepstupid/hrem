"""Adaptive patience management for scientific discovery."""

import time
from typing import List, Optional, Dict, Any
from enum import Enum
from dataclasses import dataclass
from .config import PatienceBudget

class ExplorationPhase(Enum):
    """Enumeration of exploration phases."""
    BASELINE_EVALUATION = "baseline_evaluation"
    HYPERPARAMETER_OPTIMIZATION = "hyperparameter_optimization"
    FINAL_EVALUATION = "final_evaluation"
    INSIGHT_GENERATION = "insight_generation"

@dataclass
class TimeAllocation:
    """Time allocation for an exploration phase."""
    phase: ExplorationPhase
    allocated_seconds: float
    max_trials: Optional[int] = None

@dataclass
class ScientificInsight:
    """A scientific insight generated from algorithm comparison."""
    type: str                    # EFFICIENCY, ROBUSTNESS, SCALABILITY, etc.
    confidence: float           # Statistical confidence level (0.0-1.0)
    evidence: List[Dict[str, Any]]  # Supporting evidence with statistical tests
    implications: List[str]     # Scientific implications
    discovery_potential: float  # Future research value (0.0-1.0)

class AdaptivePatienceManager:
    """Intelligently allocates user patience across exploration activities."""
    
    def __init__(self, initial_patience: PatienceBudget):
        """Initialize with user's patience budget."""
        self.initial_patience = initial_patience
        self.start_time = time.time()
        self.consumed_time = 0.0
        self.phase_allocations: Dict[ExplorationPhase, float] = {}
        
        # Set total budget based on patience level
        if initial_patience.custom_seconds:
            self.total_budget = initial_patience.custom_seconds
        else:
            budget_map = {
                "low": 120,      # 2 minutes
                "medium": 600,   # 10 minutes
                "high": 1800     # 30 minutes
            }
            self.total_budget = budget_map.get(initial_patience.level, 600)
    
    def get_elapsed_time(self) -> float:
        """Get total elapsed time since initialization."""
        return time.time() - self.start_time
    
    def get_remaining_budget(self) -> float:
        """Get remaining patience budget."""
        return max(0, self.total_budget - self.get_elapsed_time())
    
    def allocate_for_phase(self, phase: ExplorationPhase, discovery_potential: float) -> TimeAllocation:
        """
        Allocate patience budget based on discovery potential.
        
        Args:
            phase: The exploration phase
            discovery_potential: Expected value of insights (0.0-1.0)
            
        Returns:
            TimeAllocation object with allocated time
        """
        remaining_budget = self.get_remaining_budget()
        
        # Base allocation based on phase type
        phase_weights = {
            ExplorationPhase.BASELINE_EVALUATION: 0.2,
            ExplorationPhase.HYPERPARAMETER_OPTIMIZATION: 0.5,
            ExplorationPhase.FINAL_EVALUATION: 0.2,
            ExplorationPhase.INSIGHT_GENERATION: 0.1
        }
        
        # Adjust allocation based on discovery potential
        base_weight = phase_weights.get(phase, 0.2)
        adjusted_weight = base_weight * (0.7 + 0.3 * discovery_potential)
        
        allocated_seconds = remaining_budget * adjusted_weight
        
        # For optimization phase, also calculate max trials
        max_trials = None
        if phase == ExplorationPhase.HYPERPARAMETER_OPTIMIZATION:
            # Estimate trials based on model execution time (simplified)
            avg_trial_time = 10.0  # This would be dynamically calculated
            max_trials = max(1, int(allocated_seconds / avg_trial_time))
            max_trials = min(max_trials, 20)  # Cap at reasonable number
        
        allocation = TimeAllocation(
            phase=phase,
            allocated_seconds=allocated_seconds,
            max_trials=max_trials
        )
        
        self.phase_allocations[phase] = allocated_seconds
        return allocation
    
    def update_patience_consumption(self, actual_time: float, phase: ExplorationPhase):
        """
        Update remaining patience based on actual consumption.
        
        Args:
            actual_time: Actual time consumed
            phase: The phase that consumed the time
        """
        self.consumed_time += actual_time
        # Update allocation tracking
        if phase in self.phase_allocations:
            self.phase_allocations[phase] = actual_time
    
    def should_extend_exploration(self, current_insights: List[ScientificInsight]) -> bool:
        """
        Determine if patience budget should be extended for potential insights.
        
        Args:
            current_insights: List of insights generated so far
            
        Returns:
            Boolean indicating if exploration should be extended
        """
        if not current_insights:
            return False
            
        # Calculate average discovery potential of current insights
        avg_potential = sum(insight.discovery_potential for insight in current_insights) / len(current_insights)
        
        # Extend if high potential insights and we're near the threshold
        remaining_budget = self.get_remaining_budget()
        threshold = self.initial_patience.extension_threshold
        
        return (avg_potential > threshold and 
                remaining_budget < 60 and  # Less than 1 minute remaining
                self.total_budget < 1800)  # Not already at maximum budget