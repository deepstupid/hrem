"""Enhanced timing utilities for the HRM/HREM demo system with comprehensive instrumentation.

This module provides comprehensive timing and metrics collection utilities for the demo system,
allowing detailed performance analysis and optimization of the HRM/HREM models.
"""

import time
import json
from typing import Dict, Any, Optional, List, Union
from collections import defaultdict
from rich.console import Console
import numpy as np

console = Console()

class EnhancedTimingManager:
    """Enhanced manager for collecting timing information and comprehensive metrics.
    
    This manager provides comprehensive timing and metrics collection capabilities,
    including:
    - Operation timing tracking
    - Metric value recording
    - Loop counter tracking
    - Generation counter tracking
    - Trial counter tracking
    - Statistical analysis of collected data
    - Data export functionality
    """
    
    def __init__(self, collect_detailed_metrics: bool = True):
        """Initialize the enhanced timing manager.
        
        Args:
            collect_detailed_metrics: Whether to collect detailed metrics (default: True)
        """
        self.timings: Dict[str, list] = defaultdict(list)
        self.metrics = defaultdict(list)
        self.current_iteration = 0
        self.collect_detailed_metrics = collect_detailed_metrics
        self.loop_counters = defaultdict(int)
        self.generation_counters = defaultdict(int)
        self.trial_counters = defaultdict(int)
        
    def start_timer(self) -> float:
        """Start a timer and return the start time.
        
        Returns:
            Start time as a float timestamp
        """
        return time.time()
        
    def end_timer(self, start_time: float) -> float:
        """End a timer and return elapsed time.
        
        Args:
            start_time: Start time returned by start_timer()
            
        Returns:
            Elapsed time in seconds
        """
        return time.time() - start_time
        
    def update_timing(self, operation: str, elapsed_time: float):
        """Update timing for an operation.
        
        Args:
            operation: Name of the operation
            elapsed_time: Time taken for the operation in seconds
        """
        self.timings[operation].append(elapsed_time)
        
    def record_metric(self, name: str, value: Any, iteration: Optional[int] = None):
        """Record a metric value.
        
        Args:
            name: Name of the metric
            value: Value of the metric
            iteration: Iteration number (default: current iteration)
        """
        if iteration is None:
            iteration = self.current_iteration
            
        self.metrics[name].append({
            'value': value,
            'iteration': iteration,
            'timestamp': time.time()
        })
        
    def record_timing(self, operation: str, elapsed_time: float, iteration: Optional[int] = None):
        """Record timing for an operation.
        
        Args:
            operation: Name of the operation
            elapsed_time: Time taken for the operation in seconds
            iteration: Iteration number (default: current iteration)
        """
        self.record_metric(f"timing_{operation}", elapsed_time, iteration)
        # Also update the base timing
        self.update_timing(operation, elapsed_time)
        
    def increment_iteration(self):
        """Increment the current iteration counter."""
        self.current_iteration += 1
        
    def increment_loop_counter(self, loop_name: str):
        """Increment a specific loop counter.
        
        Args:
            loop_name: Name of the loop to increment
        """
        self.loop_counters[loop_name] += 1
        self.record_metric(f"loop_{loop_name}_count", self.loop_counters[loop_name])
        
    def increment_generation_counter(self, generation_name: str):
        """Increment a specific generation counter.
        
        Args:
            generation_name: Name of the generation to increment
        """
        self.generation_counters[generation_name] += 1
        self.record_metric(f"generation_{generation_name}_count", self.generation_counters[generation_name])
        
    def increment_trial_counter(self, trial_name: str):
        """Increment a specific trial counter.
        
        Args:
            trial_name: Name of the trial to increment
        """
        self.trial_counters[trial_name] += 1
        self.record_metric(f"trial_{trial_name}_count", self.trial_counters[trial_name])
        
    def estimate_model_time(self, model_name: str, smoke_test: bool = False) -> float:
        """Estimate time for a model based on previous runs.
        
        Args:
            model_name: Name of the model
            smoke_test: Whether this is a smoke test (affects default estimates)
            
        Returns:
            Estimated time in seconds
        """
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
        """Get statistics for a timing operation.
        
        Args:
            operation: Name of the operation
            
        Returns:
            Dictionary with timing statistics (count, min, max, avg, last, std)
        """
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
        """Get statistics for a metric.
        
        Args:
            name: Name of the metric
            
        Returns:
            Dictionary with metric statistics (count, min, max, avg, last, std)
        """
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
        """Get comprehensive statistics for all metrics and timings.
        
        Returns:
            Dictionary mapping metric/timing names to their statistics
        """
        stats = {}
        
        # Add timing stats
        for operation in self.timings.keys():
            stats[f"timing_{operation}"] = self.get_timing_stats(operation)
            
        # Add metric stats
        for metric_name in self.metrics.keys():
            stats[metric_name] = self.get_metric_stats(metric_name)
            
        return stats
        
    def display_timing_summary(self, title: str = "⏱️  Operation Timing Summary"):
        """Display a summary of recorded timings.
        
        Args:
            title: Title for the timing summary table
        """
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
        table.add_column("Std (s)", justify="right")
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
                    f"{stats['std']:.2f}",
                    f"{stats['last']:.2f}"
                )
                
        console.print(table)
        
    def display_loop_summary(self, title: str = "🔁 Loop Counter Summary"):
        """Display a summary of loop counters.
        
        Args:
            title: Title for the loop summary table
        """
        if not self.loop_counters:
            return
            
        from rich.table import Table
        from rich import box
        
        table = Table(title=title, show_header=True, header_style="bold magenta", box=box.ROUNDED)
        table.add_column("Loop", style="cyan")
        table.add_column("Count", justify="right")
        
        for loop_name, count in self.loop_counters.items():
            table.add_row(loop_name, str(count))
            
        console.print(table)
        
    def display_generation_summary(self, title: str = "🧬 Generation Counter Summary"):
        """Display a summary of generation counters.
        
        Args:
            title: Title for the generation summary table
        """
        if not self.generation_counters:
            return
            
        from rich.table import Table
        from rich import box
        
        table = Table(title=title, show_header=True, header_style="bold magenta", box=box.ROUNDED)
        table.add_column("Generation", style="cyan")
        table.add_column("Count", justify="right")
        
        for gen_name, count in self.generation_counters.items():
            table.add_row(gen_name, str(count))
            
        console.print(table)
        
    def display_trial_summary(self, title: str = "🔬 Trial Counter Summary"):
        """Display a summary of trial counters.
        
        Args:
            title: Title for the trial summary table
        """
        if not self.trial_counters:
            return
            
        from rich.table import Table
        from rich import box
        
        table = Table(title=title, show_header=True, header_style="bold magenta", box=box.ROUNDED)
        table.add_column("Trial", style="cyan")
        table.add_column("Count", justify="right")
        
        for trial_name, count in self.trial_counters.items():
            table.add_row(trial_name, str(count))
            
        console.print(table)
        
    def display_comprehensive_summary(self):
        """Display a comprehensive summary of all metrics."""
        self.display_timing_summary()
        self.display_loop_summary()
        self.display_generation_summary()
        self.display_trial_summary()
        
    def export_metrics(self) -> Dict[str, Any]:
        """Export all collected metrics.
        
        Returns:
            Dictionary of all collected metrics
        """
        return dict(self.metrics)
        
    def export_comprehensive_data(self) -> Dict[str, Any]:
        """Export all collected data including metrics, timings, and counters.
        
        Returns:
            Dictionary containing all collected data
        """
        return {
            'metrics': dict(self.metrics),
            'timings': dict(self.timings),
            'loop_counters': dict(self.loop_counters),
            'generation_counters': dict(self.generation_counters),
            'trial_counters': dict(self.trial_counters),
            'stats': self.get_comprehensive_stats()
        }
        
    def save_metrics_to_file(self, filepath: str):
        """Save all metrics to a JSON file.
        
        Args:
            filepath: Path to save the JSON file
        """
        try:
            data = self.export_comprehensive_data()
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            console.print(f"[green]✅ Metrics saved to {filepath}[/green]")
        except Exception as e:
            console.print(f"[red]❌ Failed to save metrics to {filepath}: {str(e)}[/red]")

class EnhancedTimingContext:
    """Enhanced context manager for timing operations with detailed tracking.
    
    This context manager automatically tracks the timing of operations
    and records them in the timing manager.
    """
    
    def __init__(self, manager: EnhancedTimingManager, operation: str, iteration: Optional[int] = None):
        """Initialize the timing context.
        
        Args:
            manager: EnhancedTimingManager instance
            operation: Name of the operation to time
            iteration: Iteration number (default: None)
        """
        self.manager = manager
        self.operation = operation
        self.iteration = iteration
        self.start_time = None
        
    def __enter__(self):
        """Enter the timing context."""
        self.start_time = self.manager.start_timer()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the timing context and record the timing."""
        if self.start_time is not None:
            elapsed = self.manager.end_timer(self.start_time)
            self.manager.record_timing(self.operation, elapsed, self.iteration)

class LoopTracker:
    """Context manager for tracking loops with detailed metrics.
    
    This context manager tracks loop executions and records their timing
    and count metrics.
    """
    
    def __init__(self, manager: EnhancedTimingManager, loop_name: str):
        """Initialize the loop tracker.
        
        Args:
            manager: EnhancedTimingManager instance
            loop_name: Name of the loop to track
        """
        self.manager = manager
        self.loop_name = loop_name
        self.start_time = None
        
    def __enter__(self):
        """Enter the loop tracking context."""
        self.start_time = self.manager.start_timer()
        self.manager.increment_loop_counter(self.loop_name)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the loop tracking context and record the timing."""
        if self.start_time is not None:
            elapsed = self.manager.end_timer(self.start_time)
            self.manager.record_timing(f"loop_{self.loop_name}", elapsed)

class GenerationTracker:
    """Context manager for tracking generations with detailed metrics.
    
    This context manager tracks generation executions and records their timing
    and count metrics.
    """
    
    def __init__(self, manager: EnhancedTimingManager, generation_name: str):
        """Initialize the generation tracker.
        
        Args:
            manager: EnhancedTimingManager instance
            generation_name: Name of the generation to track
        """
        self.manager = manager
        self.generation_name = generation_name
        self.start_time = None
        
    def __enter__(self):
        """Enter the generation tracking context."""
        self.start_time = self.manager.start_timer()
        self.manager.increment_generation_counter(self.generation_name)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the generation tracking context and record the timing."""
        if self.start_time is not None:
            elapsed = self.manager.end_timer(self.start_time)
            self.manager.record_timing(f"generation_{self.generation_name}", elapsed)

class TrialTracker:
    """Context manager for tracking trials with detailed metrics.
    
    This context manager tracks trial executions and records their timing
    and count metrics.
    """
    
    def __init__(self, manager: EnhancedTimingManager, trial_name: str):
        """Initialize the trial tracker.
        
        Args:
            manager: EnhancedTimingManager instance
            trial_name: Name of the trial to track
        """
        self.manager = manager
        self.trial_name = trial_name
        self.start_time = None
        
    def __enter__(self):
        """Enter the trial tracking context."""
        self.start_time = self.manager.start_timer()
        self.manager.increment_trial_counter(self.trial_name)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the trial tracking context and record the timing."""
        if self.start_time is not None:
            elapsed = self.manager.end_timer(self.start_time)
            self.manager.record_timing(f"trial_{self.trial_name}", elapsed)