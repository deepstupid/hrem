"""Utility functions and classes for the HRM/HREM demo system."""

from typing import Dict, Any, List, Optional
from rich.console import Console
from rich.table import Table
from rich import box
from rich.panel import Panel
from hrm_system.config import HREMParams

console = Console()

class DemoLogger:
    """A logger that captures and displays messages in a structured way."""
    def __init__(self):
        self.messages = []
        
    def log(self, message: str):
        """Log a message and display it."""
        if ("it/s" not in message and "%" not in message and 
            "TensorFloat32" not in message and "Online softmax" not in message and
            "torch._prims_common.check" not in message and
            "FutureWarning" not in message and "UserWarning" not in message):
            console.print(message)
            self.messages.append(message)

class ResultsDisplay:
    """Handles displaying results in various formats, using a config object."""
    
    def __init__(self, ui_config: Dict[str, Any]):
        """Initialize with the UI configuration."""
        self.ui_config = ui_config
        self.colors = ui_config.get("display_colors", ["blue", "green", "yellow", "magenta", "cyan", "red"])
        self.metrics_info = ui_config.get("metrics_info", [])
        self.key_metrics = ui_config.get("key_metrics", [])

    def _create_results_table(self, title: str, model_names: List[str], model_styles: Dict[str, str]) -> Table:
        """Create a standardized results table with proper styling."""
        table = Table(title=title, show_header=True, header_style="bold magenta", box=box.ROUNDED)
        table.add_column("Metric", style="cyan")
        
        for model_name in model_names:
            style = model_styles.get(model_name, "bold white")
            table.add_column(model_name, justify="right", style=style)
        
        return table
    
    def _format_metric_value(self, val, key):
        """Format metric values for display."""
        if isinstance(val, float) and (val != val):
            return "N/A"
        if isinstance(val, (int, float)):
            if key == 'num_params':
                return f"{val:,}"
            else:
                return f"{val:.4f}"
        return str(val)
    
    def display_model_detailed_stats(self, title: str, results: Dict[str, Any], model_names: list = None):
        """Display detailed statistics for models."""
        if not model_names:
            model_names = list(results.keys())
        if not model_names:
            return
            
        model_styles = {}
        for i, model_name in enumerate(model_names):
            color = self.colors[i % len(self.colors)]
            model_styles[model_name] = f"bold {'bright_' if '_best' in model_name else ''}{color}"
                
        table = self._create_results_table(title, model_names, model_styles)
        
        if results:
            model_metrics = {name: results.get(name, {}) for name in model_names}
            
            for key, display_name, description in self.metrics_info:
                if not any(model_metrics[name].get(key, 'N/A') != 'N/A' for name in model_metrics):
                    continue
                    
                row_values = [self._format_metric_value(model_metrics.get(name, {}).get(key, 'N/A'), key) for name in model_names]
                table.add_row(display_name, *row_values)
                
            console.print(table)
    
    def display_final_comparison(self, title: str, final_results: Dict[str, Any]):
        """Display a clear comparison of final results for all models."""
        console.print(Panel(f"[bold]{title}[/bold]", expand=False))
        
        model_names = sorted(list(final_results.keys()))
        if not model_names:
            console.print("[dim]No models to compare[/dim]")
            return
            
        model_styles = {name: f"bold {'bright_' if '_best' in name else ''}{self.colors[i % len(self.colors)]}" for i, name in enumerate(model_names)}
        table = self._create_results_table("", model_names, model_styles)
        
        table.columns[0].header = "Metric"
        for i, model_name in enumerate(model_names, 1):
            display_name = model_name.replace('_best', '') + (' (Optimized)' if '_best' in model_name else '')
            table.columns[i]._header = display_name
            table.columns[i].style = model_styles.get(model_name, "white")
            table.columns[i].justify = "right"
        
        for key, display_name in self.key_metrics:
            row_values = [self._format_value(final_results.get(name, {}).get(key, 'N/A'), key) for name in model_names]
            table.add_row(display_name, *row_values)
        
        console.print(table)
    
    def _format_value(self, val, key):
        """Helper method to format values for display."""
        if isinstance(val, float) and (val != val):
            return "N/A"
        if isinstance(val, (int, float)):
            if key == 'num_params':
                return f"{val:,}"
            else:
                return f"{val:.4f}"
        return str(val)
    
    def display_current_leader(self, results: Dict[str, Any]):
        """Display the current leader with a colorful panel based on accuracy."""
        if not results:
            return
            
        best_model, best_accuracy = None, -1
        for model_name, metrics in results.items():
            accuracy = metrics.get('all/accuracy', 0)
            try:
                accuracy = float(accuracy) if isinstance(accuracy, str) else accuracy
            except (ValueError, TypeError):
                continue
            if accuracy > best_accuracy:
                best_accuracy, best_model = accuracy, model_name
        
        if best_model:
            console.print(f"\n[bold green]👑 Current Leader: {best_model}[/bold green]")
            console.print(f"[dim]Accuracy: {best_accuracy:.4f}[/dim]")
    
    def display_final_leader(self, final_results: Dict[str, Any]):
        """Display the final leader based on accuracy."""
        if not final_results:
            return
            
        best_model_final, best_accuracy_final = None, -1
        for model_name, metrics in final_results.items():
            accuracy = metrics.get('all/accuracy', 0)
            try:
                accuracy = float(accuracy) if isinstance(accuracy, str) else accuracy
            except (ValueError, TypeError):
                continue
            if accuracy > best_accuracy_final:
                best_accuracy_final, best_model_final = accuracy, model_name
        
        if best_model_final:
            console.print(f"\n[bold green]🏆 Final Leader: {best_model_final}[/bold green]")
            console.print(f"[dim]Accuracy: {best_accuracy_final:.4f}[/dim]")
    
    def display_hrem_params(self, title: str, params: HREMParams):
        """Display HREM parameters in a formatted table."""
        if not params:
            return
            
        table = Table(title=title, show_header=True, header_style="bold cyan", box=box.ROUNDED)
        table.add_column("Parameter", style="dim")
        table.add_column("Value", justify="right")
        
        param_dict = params.model_dump()
        for key, value in param_dict.items():
            table.add_row(key, str(value))
                
        console.print(table)
    
    @staticmethod
    def display_iteration_header(title: str, description: str = ""):
        """Display a header for each iteration with a clean screen."""
        console.clear()
        console.print(Panel(f"[bold blue]{title}[/bold blue]", expand=False))
        if description:
            console.print(f"[dim]{description}[/dim]")
        console.print()

def display_optimization_results(model_name: str, opt_result: Dict[str, Any]):
    """Display optimization results for a single model."""
    if opt_result and "best_params" in opt_result:
        console.print(f"\n[bold blue]{model_name} Optimization Results:[/bold blue]")
        # Display parameters in a table
        params_table = Table(show_header=True, header_style="bold cyan", box=box.ROUNDED)
        params_table.add_column("Parameter", style="dim")
        params_table.add_column("Value", justify="right")
        for key, value in opt_result["best_params"].items():
            params_table.add_row(key, str(value))
        console.print(params_table)
    else:
        console.print(f"\n[bold blue]{model_name} Optimization Results:[/bold blue]")
        console.print("  No optimization parameters found")