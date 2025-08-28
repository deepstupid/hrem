"""Core timing functionality for the HRM/HREM demo system."""

import time
from typing import Dict, Optional
from collections import defaultdict
from rich.console import Console
import numpy as np
from contextlib import contextmanager

console = Console()

class TimingManager:
    """A manager for collecting timing information."""
    
    def __init__(self):
        self.timings: Dict[str, list] = defaultdict(list)
        
    def start_timer(self) -> float:
        return time.time()
        
    def end_timer(self, start_time: float) -> float:
        return time.time() - start_time
        
    def update_timing(self, operation: str, elapsed_time: float):
        self.timings[operation].append(elapsed_time)
        
    def record_timing(self, operation: str, elapsed_time: float, iteration: Optional[int] = None):
        self.update_timing(operation, elapsed_time)
        
    def estimate_model_time(self, model_name: str, smoke_test: bool = False) -> float:
        if model_name in self.timings and self.timings[model_name]:
            recent_runs = self.timings[model_name][-3:]
            return sum(recent_runs) / len(recent_runs)
        if smoke_test:
            return 0.5
        if "HRM" in model_name:
            return 3.0
        return 5.0
        
    def get_timing_stats(self, operation: str) -> Dict[str, float]:
        times = self.timings.get(operation, [])
        if not times:
            return {}
            
        return {
            'count': len(times),
            'min': min(times),
            'max': max(times),
            'avg': sum(times) / len(times),
            'last': times[-1],
            'std': float(np.std(times)) if len(times) > 1 else 0.0
        }
        
    def display_timing_summary(self, title: str = "⏱️  Operation Timing Summary"):
        from rich.table import Table
        from rich import box
        
        table = Table(title=title, show_header=True, header_style="bold magenta", box=box.ROUNDED)
        table.add_column("Operation", style="cyan")
        table.add_column("Count", justify="right")
        table.add_column("Min (s)", justify="right")
        table.add_column("Max (s)", justify="right")
        table.add_column("Avg (s)", justify="right")
        table.add_column("Std (s)", justify="right")
        
        for operation, times in self.timings.items():
            stats = self.get_timing_stats(operation)
            if stats:
                table.add_row(
                    operation,
                    str(stats['count']),
                    f"{stats['min']:.2f}",
                    f"{stats['max']:.2f}",
                    f"{stats['avg']:.2f}",
                    f"{stats['std']:.2f}",
                )
        if table.rows:
            console.print(table)

    @contextmanager
    def get_context(self, operation: str, iteration: Optional[int] = None):
        """A context manager for timing operations."""
        start_time = self.start_timer()
        try:
            yield
        finally:
            elapsed = self.end_timer(start_time)
            self.record_timing(operation, elapsed, iteration)
