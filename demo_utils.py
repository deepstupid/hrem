"""Demo utilities for displaying results in the TUI."""

from typing import Dict, Any, List
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from hrm_system.reporting import display_final_comparison as reporting_display_final_comparison

console = Console()

class ResultsDisplay:
    """Utility class for displaying results in various formats."""
    
    def __init__(self, ui_config: Dict[str, Any]):
        self.ui_config = ui_config or {}
        
    def display_model_detailed_stats(self, title: str, results: Dict[str, Any], model_names: List[str] = None):
        """Display detailed statistics for models."""
        if not results:
            console.print(f"[dim]No results to display for {title}[/dim]")
            return
            
        console.print(Panel(f"[bold]{title}[/bold]", expand=False))
        
        # If specific model names provided, use those
        if model_names:
            display_names = [name for name in model_names if name in results]
        else:
            # Otherwise display all results
            display_names = list(results.keys())
            
        if not display_names:
            console.print("[dim]No models to display[/dim]")
            return
            
        # Create detailed table
        table = Table(box=box.ROUNDED)
        table.add_column("Model", style="cyan")
        table.add_column("Accuracy", justify="right")
        table.add_column("Loss", justify="right")
        table.add_column("Steps", justify="right")
        table.add_column("Parameters", justify="right")
        
        for name in display_names:
            metrics = results.get(name, {})
            if not metrics:
                continue
                
            accuracy = str(metrics.get('all/accuracy', 'N/A'))
            loss = str(metrics.get('all/lm_loss', 'N/A'))
            steps = str(metrics.get('all/steps', 'N/A'))
            params = str(metrics.get('num_params', 'N/A'))
            
            # Style the best values
            table.add_row(name, accuracy, loss, steps, params)
            
        console.print(table)
        
    def display_final_comparison(self, title: str, final_results: Dict[str, Any]):
        """Display a clear comparison of final results."""
        reporting_display_final_comparison(title, final_results)
        
    def display_final_leader(self, final_results: Dict[str, Any]):
        """Display the winning model based on accuracy."""
        if not final_results:
            return
            
        # Find model with highest accuracy
        best_model = None
        best_accuracy = -1
        
        for model_name, metrics in final_results.items():
            accuracy_str = metrics.get('all/accuracy', '0')
            try:
                accuracy = float(accuracy_str) if accuracy_str != 'N/A' else 0
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model = model_name
            except (ValueError, TypeError):
                pass
                
        if best_model:
            console.print(f"\n[bold gold3]🏆 Winner: {best_model} with {best_accuracy:.4f} accuracy![/bold gold3]")