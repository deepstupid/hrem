"""Utility functions and classes for the HRM/HREM demo system."""

from typing import Dict, Any, List, Optional
from rich.console import Console
from rich.table import Table
from rich import box
from rich.panel import Panel
from hrm_system.config import HREMParams

# Import centralized parameters
from demo_parameters import DISPLAY_COLORS, METRICS_INFO, KEY_METRICS

console = Console()

class DemoLogger:
    """A logger that captures and displays messages in a structured way."""
    def __init__(self):
        self.messages = []
        
    def log(self, message: str):
        """Log a message and display it."""
        # Filter out progress bar updates and warning messages to reduce verbosity
        if ("it/s" not in message and "%" not in message and 
            "TensorFloat32" not in message and "Online softmax" not in message and
            "torch._prims_common.check" not in message and
            "FutureWarning" not in message and "UserWarning" not in message):
            console.print(message)
            self.messages.append(message)

class ResultsDisplay:
    """Handles displaying results in various formats."""
    
    @staticmethod
    def _create_results_table(title: str, model_names: List[str], model_styles: Dict[str, str]) -> Table:
        """Create a standardized results table with proper styling."""
        table = Table(title=title, show_header=True, header_style="bold magenta", box=box.ROUNDED)
        table.add_column("Metric", style="cyan")
        
        # Add columns for each model
        for model_name in model_names:
            style = model_styles.get(model_name, "bold white")
            table.add_column(model_name, justify="right", style=style)
        
        return table
    
    @staticmethod
    def _format_metric_value(val, key):
        """Format metric values for display."""
        # Handle NaN values explicitly
        if isinstance(val, float) and (val != val):  # NaN check
            return "N/A"
        if isinstance(val, (int, float)):
            if key == 'num_params':
                return f"{val:,}"  # Add commas for large numbers
            else:
                return f"{val:.4f}"
        return str(val)
    
    @staticmethod
    def display_model_detailed_stats(title: str, results: Dict[str, Any], model_names: list = None):
        """Display detailed statistics for models including parameter counts and performance metrics."""
        if not model_names:
            # Determine which models are present in the results
            model_names = list(results.keys())
        
        if not model_names:
            return
            
        # Create table with consistent styling
        # Dynamically generate styles based on model names
        model_styles = {}
        colors = DISPLAY_COLORS
        for i, model_name in enumerate(model_names):
            color = colors[i % len(colors)]
            if "_best" in model_name:
                model_styles[model_name] = f"bold bright_{color}"
            else:
                model_styles[model_name] = f"bold {color}"
                
        table = ResultsDisplay._create_results_table(title, model_names, model_styles)
        
        if results:
            # Get metrics for available models
            model_metrics = {}
            for model_name in model_names:
                if model_name in results:
                    model_metrics[model_name] = results.get(model_name, {})
            
            # Common metrics to display with descriptions
            metrics_info = METRICS_INFO
            
            for key, display_name, description in metrics_info:
                # Check if any model has this metric
                has_metric = any(model_metrics[model_name].get(key, 'N/A') != 'N/A' for model_name in model_metrics)
                if not has_metric:
                    continue
                    
                row_values = []
                for model_name in model_names:
                    if model_name in model_metrics:
                        val = model_metrics[model_name].get(key, 'N/A')
                        # Format values
                        row_values.append(ResultsDisplay._format_metric_value(val, key))
                    else:
                        row_values.append('N/A')
                
                table.add_row(display_name, *row_values)
                
            console.print(table)
    
    @staticmethod
    def display_final_comparison(title: str, final_results: Dict[str, Any]):
        """Display a clear comparison of final results for all models."""
        console.print(Panel(f"[bold]{title}[/bold]", expand=False))
        
        # Get all model names from final results
        model_names = list(final_results.keys())
        
        if not model_names:
            console.print("[dim]No models to compare[/dim]")
            return
            
        # Create comparison table
        # Dynamically generate styles based on model names
        model_styles = {}
        colors = DISPLAY_COLORS
        for i, model_name in enumerate(model_names):
            color = colors[i % len(colors)]
            if "_best" in model_name:
                model_styles[model_name] = f"bold bright_{color}"
            else:
                model_styles[model_name] = f"bold {color}"
                
        table = ResultsDisplay._create_results_table("", sorted(model_names), model_styles)
        
        # Add columns for each model with custom display names
        table.columns[0].header = "Metric"  # Reset first column header
        
        # Update column headers with styled names
        for i, model_name in enumerate(sorted(model_names), 1):  # Start from 1 because first column is Metric
            base_style = model_styles.get(model_name, "white")
            # Remove '_best' suffix for cleaner display
            display_name = model_name.replace('_best', '') + (' (Optimized)' if '_best' in model_name else '')
            table.columns[i]._header = display_name
            table.columns[i].style = base_style
            table.columns[i].justify = "right"
        
        # Key metrics for comparison
        metrics_info = KEY_METRICS
        
        for key, display_name in metrics_info:
            row_values = []
            for model_name in sorted(model_names):
                val = final_results.get(model_name, {}).get(key, 'N/A')
                row_values.append(ResultsDisplay._format_value(val, key))
            
            table.add_row(display_name, *row_values)
        
        console.print(table)
    
    @staticmethod
    def _format_value(val, key):
        """Helper method to format values for display."""
        # Handle NaN values explicitly
        if isinstance(val, float) and (val != val):  # NaN check
            return "N/A"
        if isinstance(val, (int, float)):
            if key == 'num_params':
                return f"{val:,}"  # Add commas for large numbers
            else:
                return f"{val:.4f}"
        return str(val)
    
    @staticmethod
    def display_current_leader(results: Dict[str, Any]):
        """Display the current leader with a colorful panel based on accuracy."""
        if not results:
            return
            
        # Find the model with the highest accuracy
        best_model = None
        best_accuracy = -1
        for model_name, metrics in results.items():
            accuracy = metrics.get('all/accuracy', 0)
            try:
                accuracy = float(accuracy) if isinstance(accuracy, str) else accuracy
            except (ValueError, TypeError):
                continue
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model_name
        
        if best_model:
            console.print(f"\n[bold green]👑 Current Leader: {best_model}[/bold green]")
            console.print(f"[dim]Accuracy: {best_accuracy:.4f}[/dim]")
    
    @staticmethod
    def display_final_leader(final_results: Dict[str, Any]):
        """Display the final leader based on accuracy."""
        if not final_results:
            return
            
        # Find the model with the highest accuracy
        best_model_final = None
        best_accuracy_final = -1
        for model_name, metrics in final_results.items():
            accuracy = metrics.get('all/accuracy', 0)
            try:
                accuracy = float(accuracy) if isinstance(accuracy, str) else accuracy
            except (ValueError, TypeError):
                continue
            if accuracy > best_accuracy_final:
                best_accuracy_final = accuracy
                best_model_final = model_name
        
        if best_model_final:
            console.print(f"\n[bold green]🏆 Final Leader: {best_model_final}[/bold green]")
            console.print(f"[dim]Accuracy: {best_accuracy_final:.4f}[/dim]")
    
    @staticmethod
    def display_hrem_params(title: str, params: HREMParams):
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