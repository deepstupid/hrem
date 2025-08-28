"""Enhanced timing utilities for the HRM/HREM demo system with comprehensive instrumentation."""

import time
import json
from typing import Dict, Any, Optional, List, Union
from collections import defaultdict
from rich.console import Console
import numpy as np
from contextlib import contextmanager

console = Console()

class TimingManager:
    """Enhanced manager for collecting timing information and comprehensive metrics."""
    
    def __init__(self, collect_detailed_metrics: bool = True):
        self.timings: Dict[str, list] = defaultdict(list)
        self.metrics = defaultdict(list)
        self.current_iteration = 0
        self.collect_detailed_metrics = collect_detailed_metrics
        self.loop_counters = defaultdict(int)
        self.generation_counters = defaultdict(int)
        self.trial_counters = defaultdict(int)
        
    def start_timer(self) -> float:
        return time.time()
        
    def end_timer(self, start_time: float) -> float:
        return time.time() - start_time
        
    def update_timing(self, operation: str, elapsed_time: float):
        self.timings[operation].append(elapsed_time)
        
    def record_metric(self, name: str, value: Any, iteration: Optional[int] = None):
        if iteration is None:
            iteration = self.current_iteration
            
        self.metrics[name].append({
            'value': value,
            'iteration': iteration,
            'timestamp': time.time()
        })
        
    def record_timing(self, operation: str, elapsed_time: float, iteration: Optional[int] = None):
        self.record_metric(f"timing_{operation}", elapsed_time, iteration)
        self.update_timing(operation, elapsed_time)
        
    def increment_iteration(self):
        self.current_iteration += 1
        
    def increment_loop_counter(self, loop_name: str):
        self.loop_counters[loop_name] += 1
        self.record_metric(f"loop_{loop_name}_count", self.loop_counters[loop_name])
        
    def increment_generation_counter(self, generation_name: str):
        self.generation_counters[generation_name] += 1
        self.record_metric(f"generation_{generation_name}_count", self.generation_counters[generation_name])
        
    def increment_trial_counter(self, trial_name: str):
        self.trial_counters[trial_name] += 1
        self.record_metric(f"trial_{trial_name}_count", self.trial_counters[trial_name])
        
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
        
    def get_metric_stats(self, name: str) -> Dict[str, float]:
        values = [m['value'] for m in self.metrics[name] if isinstance(m['value'], (int, float))]
        if not values:
            return {}
            
        return {
            'count': len(values),
            'min': min(values),
            'max': max(values),
            'avg': sum(values) / len(values),
            'last': values[-1],
            'std': float(np.std(values)) if len(values) > 1 else 0.0
        }
        
    def get_comprehensive_stats(self) -> Dict[str, Dict[str, float]]:
        stats = {}
        for operation in self.timings.keys():
            stats[f"timing_{operation}"] = self.get_timing_stats(operation)
        for metric_name in self.metrics.keys():
            stats[metric_name] = self.get_metric_stats(metric_name)
        return stats
        
    def display_summary(self, title: str = "📊 Metrics and Timing Summary"):
        """Display a comprehensive summary of all metrics."""
        self.display_timing_summary()
        self.display_loop_summary()
        self.display_generation_summary()
        self.display_trial_summary()

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

    def display_loop_summary(self, title: str = "🔁 Loop Counter Summary"):
        if not self.loop_counters: return
        from rich.table import Table
        from rich import box
        table = Table(title=title, show_header=True, header_style="bold magenta", box=box.ROUNDED)
        table.add_column("Loop", style="cyan")
        table.add_column("Count", justify="right")
        for loop_name, count in self.loop_counters.items():
            table.add_row(loop_name, str(count))
        console.print(table)

    def display_generation_summary(self, title: str = "🧬 Generation Counter Summary"):
        if not self.generation_counters: return
        from rich.table import Table
        from rich import box
        table = Table(title=title, show_header=True, header_style="bold magenta", box=box.ROUNDED)
        table.add_column("Generation", style="cyan")
        table.add_column("Count", justify="right")
        for gen_name, count in self.generation_counters.items():
            table.add_row(gen_name, str(count))
        console.print(table)

    def display_trial_summary(self, title: str = "🔬 Trial Counter Summary"):
        if not self.trial_counters: return
        from rich.table import Table
        from rich import box
        table = Table(title=title, show_header=True, header_style="bold magenta", box=box.ROUNDED)
        table.add_column("Trial", style="cyan")
        table.add_column("Count", justify="right")
        for trial_name, count in self.trial_counters.items():
            table.add_row(trial_name, str(count))
        console.print(table)
        
    def export_metrics(self) -> Dict[str, Any]:
        return dict(self.metrics)
        
    def export_comprehensive_data(self) -> Dict[str, Any]:
        return {
            'metrics': dict(self.metrics),
            'timings': dict(self.timings),
            'loop_counters': dict(self.loop_counters),
            'generation_counters': dict(self.generation_counters),
            'trial_counters': dict(self.trial_counters),
            'stats': self.get_comprehensive_stats()
        }
        
    def save_metrics_to_file(self, filepath: str):
        try:
            data = self.export_comprehensive_data()
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            console.print(f"[green]✅ Metrics saved to {filepath}[/green]")
        except Exception as e:
            console.print(f"[red]❌ Failed to save metrics to {filepath}: {str(e)}[/red]")

    @contextmanager
    def get_context(self, operation: str, iteration: Optional[int] = None):
        """A context manager for timing operations."""
        start_time = self.start_timer()
        try:
            yield
        finally:
            elapsed = self.end_timer(start_time)
            self.record_timing(operation, elapsed, iteration)
