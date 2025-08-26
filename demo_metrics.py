"""Metrics collection and timing utilities for the HRM/HREM demo system."""

import time
from typing import Dict, Any, List, Optional
from collections import defaultdict
from rich.console import Console
from rich.table import Table
from rich import box

console = Console()

class MetricsCollector:
    """Collects and manages timing and performance metrics for the demo."""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.current_iteration = 0
        
    def start_timer(self) -> float:
        """Start a timer and return the start time."""
        return time.time()
        
    def end_timer(self, start_time: float) -> float:
        """End a timer and return elapsed time."""
        return time.time() - start_time
        
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
        
    def increment_iteration(self):
        """Increment the current iteration counter."""
        self.current_iteration += 1
        
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
            
        table = Table(title=title, show_header=True, header_style="bold magenta", box=box.ROUNDED)
        table.add_column("Operation", style="cyan")
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