import time
from typing import Dict, Any, List
import numpy as np

class EnhancedTimingManager:
    """A simple timing manager to replace the missing demo_timing_utils."""
    def __init__(self):
        self.metrics: Dict[str, List[float]] = {}
        self.current_iteration = 0

    def record_timing(self, activity: str, elapsed_time: float):
        self.record_metric(activity, elapsed_time)

    def record_metric(self, name: str, value: float):
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(value)

    def get_metric_stats(self, metric_name: str) -> Dict[str, Any]:
        if metric_name not in self.metrics or not self.metrics[metric_name]:
            return {}

        values = self.metrics[metric_name]
        return {
            "avg": np.mean(values),
            "std": np.std(values),
            "min": np.min(values),
            "max": np.max(values),
            "count": len(values),
        }

    def export_metrics(self) -> Dict[str, Any]:
        return {name: self.get_metric_stats(name) for name in self.metrics.keys()}
