"""Timing utilities for the HRM/HREM demo system."""

from typing import Dict, Any, Optional
from demo_timing_utils import TimingManager, TimingContext

class TimingCollector(TimingManager):
    """Base class for collecting timing information."""
    
    def __init__(self):
        super().__init__()
        
    def estimate_model_time(self, model_name: str, smoke_test: bool = False) -> float:
        """Estimate time for a model based on previous runs."""
        if model_name in self.timings and self.timings[model_name]:
            # Return average of last 3 runs, or all runs if less than 3
            recent_runs = self.timings[model_name][-3:]
            return sum(recent_runs) / len(recent_runs)
        # Default estimates based on model type - faster estimates
        if smoke_test:
            return 0.5  # 0.5 second for smoke test
        if "HRM" in model_name:
            return 3.0  # 3 seconds for HRM
        return 5.0  # 5 seconds for HREM

class DemoTimer(TimingContext):
    """Context manager for timing operations in demos."""
    
    def __init__(self, collector: TimingCollector, operation: str):
        super().__init__(collector, operation)