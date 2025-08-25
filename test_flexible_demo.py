#!/usr/bin/env python3
"""
Simple test script to demonstrate the flexible demo CLI functionality.
"""

import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rich.console import Console
from rich.table import Table

console = Console()

def display_model_detailed_stats(title: str, results: dict):
    """Display detailed statistics for models including parameter counts and performance metrics."""
    # Get all model names from results
    model_names = list(results.keys())
    
    if not model_names:
        return
        
    table = Table(title=title, show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    
    # Add columns for each model with distinct colors
    colors = ["bold blue", "bold green", "bold bright_green", "bold purple", "bold yellow", "bold red", "bold cyan"]
    model_styles = {model_name: colors[i % len(colors)] for i, model_name in enumerate(model_names)}
    
    for model_name in model_names:
        style = model_styles.get(model_name, "bold white")
        table.add_column(model_name, justify="right", style=style)
    
    if results:
        # Get metrics for available models
        model_metrics = {}
        for model_name in model_names:
            if model_name in results:
                model_metrics[model_name] = results.get(model_name, {})
        
        # Common metrics to display
        metrics_info = [
            ('all/accuracy', 'Accuracy'),
            ('all/lm_loss', 'Loss'),
            ('all/steps', 'Steps'),
            ('num_params', 'Parameters')
        ]
        
        for key, display_name in metrics_info:
            # Check if any model has this metric
            has_metric = any(model_metrics[model_name].get(key, 'N/A') != 'N/A' for model_name in model_metrics)
            if not has_metric:
                continue
                
            row_values = []
            for model_name in model_names:
                if model_name in model_metrics:
                    val = model_metrics[model_name].get(key, 'N/A')
                    # Format values
                    if isinstance(val, (int, float)):
                        if key == 'num_params':
                            val = f"{val:,}"  # Add commas for large numbers
                        else:
                            val = f"{val:.4f}"
                    row_values.append(str(val))
                else:
                    row_values.append('N/A')
            
            table.add_row(display_name, *row_values)
            
        console.print(table)

def display_current_leader(results: dict, metric: str = 'all/accuracy'):
    """Display the current leader with a colorful panel."""
    if not results:
        return
        
    # Find the model with the best score for the specified metric
    best_model = None
    best_score = None
    
    for model_name, metrics in results.items():
        score = metrics.get(metric, 'N/A')
        if isinstance(score, (int, float)):
            if best_score is None or score > best_score:
                best_score = score
                best_model = model_name
    
    if best_model is None or best_score is None:
        console.print("[bold yellow]🤝 No clear leader - Insufficient data for comparison[/bold yellow]")
        return
    
    # Count how many models have this score
    tied_models = [model for model, metrics in results.items() 
                   if metrics.get(metric, 'N/A') == best_score]
    
    if len(tied_models) > 1:
        winner_text = ", ".join(tied_models)
        console.print(f"[bold yellow]🤝 Current Leaders: {winner_text} (Tied {metric} scores)[/bold yellow]")
    else:
        console.print(f"[bold green]👑 Current Leader: {best_model} (Highest {metric}: {best_score:.4f})[/bold green]")

# Test the functions
if __name__ == "__main__":
    # Sample data with 3 models
    sample_results = {
        "HRM": {
            "all/accuracy": 0.75,
            "all/lm_loss": 0.25,
            "all/steps": 10,
            "num_params": 1000000
        },
        "HREM": {
            "all/accuracy": 0.85,
            "all/lm_loss": 0.15,
            "all/steps": 8,
            "num_params": 1200000
        },
        "EnhancedHREM": {
            "all/accuracy": 0.90,
            "all/lm_loss": 0.10,
            "all/steps": 6,
            "num_params": 1500000
        }
    }
    
    console.print("[bold blue]Testing flexible model display functions[/bold blue]\n")
    
    # Test display_model_detailed_stats
    display_model_detailed_stats("📊 Model Comparison", sample_results)
    
    console.print()
    
    # Test display_current_leader
    display_current_leader(sample_results, 'all/accuracy')