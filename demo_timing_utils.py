"""Unified timing utilities for the HRM/HREM demo system."""

import time
from typing import Dict, Any, Optional
from collections import defaultdict
from rich.console import Console

console = Console()

class TimingManager:
    """Unified manager for collecting timing information and metrics."""
    
    def __init__(self):
        self.timings: Dict[str, list] = defaultdict(list)
        self.metrics = defaultdict(list)
        self.current_iteration = 0
        
    def start_timer(self) -> float:
        """Start a timer and return the start time."""
        return time.time()
        
    def end_timer(self, start_time: float) -> float:
        """End a timer and return elapsed time."""
        return time.time() - start_time
        
    def update_timing(self, operation: str, elapsed_time: float):
        """Update timing for an operation."""
        self.timings[operation].append(elapsed_time)
        
    def record_metric(self, name: str, value: Any, iteration: Optional[int] = None):
        """Record a metric value."""
        if iteration is None:
            iteration = self.current_iteration
            
        self.metrics[name].append({
            'value': value,
            'iteration': iteration,
            'timestamp': time.time()
        })
        
    def record_timing(self, operation: str, elapsed_time: float, iteration: Optional[int] = None):
        """Record timing for an operation."""
        self.record_metric(f"timing_{operation}", elapsed_time, iteration)
        # Also update the base timing
        self.update_timing(operation, elapsed_time)
        
    def increment_iteration(self):
        """Increment the current iteration counter."""
        self.current_iteration += 1
        
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
        
    def get_timing_stats(self, operation: str) -> Dict[str, float]:
        """Get statistics for a timing operation."""
        times = self.timings.get(operation, [])
        if not times:
            return {}
            
        return {
            'count': len(times),
            'min': min(times),
            'max': max(times),
            'avg': sum(times) / len(times),
            'last': times[-1]
        }
        
    def get_metric_stats(self, name: str) -> Dict[str, float]:
        """Get statistics for a metric."""
        values = [m['value'] for m in self.metrics[name] if isinstance(m['value'], (int, float))]
        if not values:
            return {}
            
        return {
            'count': len(values),
            'min': min(values),
            'max': max(values),
            'avg': sum(values) / len(values),
            'last': values[-1]
        }
        
    def display_timing_summary(self, title: str = "⏱️  Operation Timing Summary"):
        """Display a summary of recorded timings."""
        timing_metrics = {k: v for k, v in self.metrics.items() if k.startswith('timing_')}
        
        if not timing_metrics:
            return
            
        from rich.table import Table
        from rich import box
        
        table = Table(title=title, show_header=True, header_style="bold magenta", box=box.ROUNDED)
        table.add_column("Metric", style="cyan")
        table.add_column("Count", justify="right")
        table.add_column("Min (s)", justify="right")
        table.add_column("Max (s)", justify="right")
        table.add_column("Avg (s)", justify="right")
        table.add_column("Last (s)", justify="right")
        
        for metric_name, metric_data in timing_metrics.items():
            operation_name = metric_name.replace('timing_', '')
            stats = self.get_metric_stats(metric_name)
            
            if stats:
                table.add_row(
                    operation_name,
                    str(stats['count']),
                    f"{stats['min']:.2f}",
                    f"{stats['max']:.2f}",
                    f"{stats['avg']:.2f}",
                    f"{stats['last']:.2f}"
                )
                
        console.print(table)
        
    def export_metrics(self) -> Dict[str, Any]:
        """Export all collected metrics."""
        return dict(self.metrics)

class TimingContext:
    """Context manager for timing operations."""
    
    def __init__(self, manager: TimingManager, operation: str, iteration: Optional[int] = None):
        self.manager = manager
        self.operation = operation
        self.iteration = iteration
        self.start_time = None
        
    def __enter__(self):
        self.start_time = self.manager.start_timer()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            elapsed = self.manager.end_timer(self.start_time)
            self.manager.record_timing(self.operation, elapsed, self.iteration)