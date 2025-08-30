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

from .insights import ScientificInsight

from .timing_manager import ScientificTimingManager

class AdaptivePatienceManager:
    """Intelligently allocates user patience across exploration activities."""
    
    def __init__(self, initial_patience: PatienceBudget, timing_manager: ScientificTimingManager):
        """Initialize with user's patience budget."""
        self.initial_patience = initial_patience
        self.timing_manager = timing_manager
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
        Allocate patience budget based on discovery potential, using the timing manager
        to make more intelligent decisions.
        
        Args:
            phase: The exploration phase
            discovery_potential: Expected value of insights (0.0-1.0)
            
        Returns:
            TimeAllocation object with allocated time
        """
        import math

        remaining_budget = self.get_remaining_budget()
        
        # Use the timing manager to predict discovery value
        prediction = self.timing_manager.predict_discovery_value(remaining_budget)
        predicted_potential = prediction.get('discovery_value', discovery_potential)

        # Use a sigmoid function to make the allocation more sensitive to potential
        # This will allocate significantly more time for high potential
        potential_factor = 1 / (1 + math.exp(-10 * (predicted_potential - 0.5))) # Sigmoid function

        phase_weights = {
            ExplorationPhase.BASELINE_EVALUATION: 0.2,
            ExplorationPhase.HYPERPARAMETER_OPTIMIZATION: 0.5,
            ExplorationPhase.FINAL_EVALUATION: 0.2,
            ExplorationPhase.INSIGHT_GENERATION: 0.1
        }
        
        base_weight = phase_weights.get(phase, 0.2)
        adjusted_weight = base_weight * potential_factor
        
        allocated_seconds = remaining_budget * adjusted_weight
        
        max_trials = None
        if phase == ExplorationPhase.HYPERPARAMETER_OPTIMIZATION:
            timing_stats = self.timing_manager.get_metric_stats("optimization_trial")
            avg_trial_time = timing_stats.get('avg', 10.0) # Default to 10s if no data

            max_trials = max(1, int(allocated_seconds / avg_trial_time))
            max_trials = min(max_trials, 50)  # Increase cap
        
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
        Determine if patience budget should be extended for potential insights, using
        the timing manager to predict the value of an extension.
        
        Args:
            current_insights: List of insights generated so far
            
        Returns:
            Boolean indicating if exploration should be extended
        """
        remaining_budget = self.get_remaining_budget()
        if remaining_budget > 60 or self.total_budget >= 1800:
            return False # Don't extend if plenty of time or already at max budget

        # Predict the value of a 60-second extension
        prediction = self.timing_manager.predict_discovery_value(60)
        predicted_value = prediction.get('discovery_value', 0.0)

        threshold = self.initial_patience.extension_threshold
        
        # Extend if the predicted value is high enough
        if predicted_value > threshold:
            self.total_budget += 60 # Extend budget by 1 minute
            console.print(f"[bold green]🚀 High potential detected! Extending patience budget by 60 seconds.[/bold green]")
            return True

        return False