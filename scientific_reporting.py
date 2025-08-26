"""Scientific reporting utilities for the HRM/HREM demo system."""

from typing import Dict, Any, List, Tuple
import math
from rich.console import Console
from rich.table import Table
from rich import box
from rich.panel import Panel

console = Console()

class ScientificReporter:
    """Handles scientific reporting and analysis of results."""
    
    @staticmethod
    def calculate_statistical_significance(results: Dict[str, Dict[str, Any]], 
                                        metric: str = 'all/accuracy') -> Dict[str, Dict[str, float]]:
        """
        Calculate statistical significance between models for a given metric.
        
        Returns a dictionary with p-values for pairwise comparisons.
        """
        # This is a simplified implementation - in a real scientific setting,
        # you would use proper statistical tests like t-test or ANOVA
        model_names = list(results.keys())
        significance_results = {}
        
        for i, model_a in enumerate(model_names):
            significance_results[model_a] = {}
            for j, model_b in enumerate(model_names):
                if i != j:
                    # Simplified comparison - in reality, you'd need multiple runs
                    val_a = results[model_a].get(metric, 0)
                    val_b = results[model_b].get(metric, 0)
                    
                    # Convert to float if needed
                    try:
                        val_a = float(val_a) if isinstance(val_a, str) else val_a
                        val_b = float(val_b) if isinstance(val_b, str) else val_b
                    except (ValueError, TypeError):
                        val_a, val_b = 0, 0
                    
                    # Simple heuristic: if difference is > 1%, consider significant
                    diff = abs(val_a - val_b)
                    significance_results[model_a][model_b] = diff > 0.01
                    
        return significance_results
    
    @staticmethod
    def generate_scientific_summary(final_results: Dict[str, Dict[str, Any]]) -> str:
        """Generate a scientific summary of the results."""
        if not final_results:
            return "No results available for scientific analysis."
        
        model_names = list(final_results.keys())
        summary_parts = []
        
        # Accuracy comparison
        accuracies = {}
        for model_name in model_names:
            acc_str = final_results[model_name].get('all/accuracy', '0')
            try:
                accuracies[model_name] = float(acc_str) if isinstance(acc_str, str) else acc_str
            except (ValueError, TypeError):
                accuracies[model_name] = 0
        
        if accuracies:
            best_model = max(accuracies, key=accuracies.get)
            summary_parts.append(f"🏆 Best performing model: {best_model} (Accuracy: {accuracies[best_model]:.4f})")
        
        # Parameter efficiency
        param_counts = {}
        for model_name in model_names:
            param_str = final_results[model_name].get('num_params', '0')
            try:
                # Handle formatted strings like "1,234,567"
                if isinstance(param_str, str):
                    param_str = param_str.replace(',', '')
                param_counts[model_name] = int(float(param_str)) if isinstance(param_str, str) else int(param_str)
            except (ValueError, TypeError):
                param_counts[model_name] = 0
        
        if param_counts:
            most_efficient = min(param_counts, key=param_counts.get)
            summary_parts.append(f"⚙️  Most parameter-efficient model: {most_efficient} ({param_counts[most_efficient]:,} parameters)")
        
        # Statistical significance
        significance = ScientificReporter.calculate_statistical_significance(final_results)
        significant_comparisons = []
        for model_a, comparisons in significance.items():
            for model_b, is_significant in comparisons.items():
                if is_significant:
                    significant_comparisons.append(f"{model_a} vs {model_b}")
        
        if significant_comparisons:
            summary_parts.append(f"🔬 Statistically significant differences found in: {', '.join(significant_comparisons)}")
        
        return "\n".join(summary_parts)
    
    @staticmethod
    def display_scientific_analysis(final_results: Dict[str, Dict[str, Any]]):
        """Display a comprehensive scientific analysis of the results."""
        console.print(Panel("[bold blue]🔬 Scientific Analysis[/bold blue]", expand=False))
        
        # Generate and display summary
        summary = ScientificReporter.generate_scientific_summary(final_results)
        console.print(summary)
        
        # Create detailed analysis table
        model_names = list(final_results.keys())
        if not model_names:
            return
            
        table = Table(title="Scientific Metrics Analysis", show_header=True, 
                     header_style="bold magenta", box=box.ROUNDED)
        table.add_column("Metric", style="cyan")
        
        # Add columns for each model
        model_styles = {"HRM": "bold blue", "HREM": "bold green", "HREM_best": "bold bright_green", "HRM_best": "bold blue"}
        for model_name in model_names:
            style = model_styles.get(model_name, "bold white")
            table.add_column(model_name, justify="right", style=style)
        
        if final_results:
            # Key scientific metrics
            scientific_metrics = [
                ('all/accuracy', 'Accuracy', '🎯'),
                ('all/lm_loss', 'Loss', '📉'),
                ('all/steps', 'Steps', '⚡'),
                ('num_params', 'Parameters', '⚙️'),
                ('all/exact_accuracy', 'Exact Accuracy', '🎯'),
                ('all/q_halt_accuracy', 'Q-Halt Accuracy', '🎯'),
            ]
            
            for key, display_name, emoji in scientific_metrics:
                # Check if any model has this metric
                has_metric = any(final_results[model_name].get(key, 'N/A') != 'N/A' for model_name in model_names)
                if not has_metric:
                    continue
                    
                row_values = []
                for model_name in model_names:
                    val = final_results[model_name].get(key, 'N/A')
                    # Format values
                    if isinstance(val, float) and (val != val):  # NaN check
                        row_values.append("N/A")
                    elif isinstance(val, (int, float)):
                        if key == 'num_params':
                            row_values.append(f"{val:,}")
                        else:
                            row_values.append(f"{val:.4f}")
                    else:
                        row_values.append(str(val))
                
                table.add_row(f"{emoji} {display_name}", *row_values)
                
            console.print(table)
        
        # Show statistical significance
        significance = ScientificReporter.calculate_statistical_significance(final_results)
        sig_table = Table(title="Statistical Significance (Pairwise Comparisons)", 
                         show_header=True, header_style="bold magenta", box=box.ROUNDED)
        sig_table.add_column("Model A", style="cyan")
        sig_table.add_column("vs", style="dim")
        sig_table.add_column("Model B", style="cyan")
        sig_table.add_column("Significant Difference", style="bold")
        
        added_comparisons = set()
        for model_a, comparisons in significance.items():
            for model_b, is_significant in comparisons.items():
                # Avoid duplicate comparisons
                comparison_key = tuple(sorted([model_a, model_b]))
                if comparison_key in added_comparisons:
                    continue
                added_comparisons.add(comparison_key)
                
                sig_status = "[bold green]Yes[/bold green]" if is_significant else "[dim]No[/dim]"
                sig_table.add_row(model_a, "vs", model_b, sig_status)
        
        console.print(sig_table)
    
    @staticmethod
    def export_results_to_csv(final_results: Dict[str, Dict[str, Any]], filename: str = "demo_results.csv"):
        """Export results to CSV for further analysis."""
        import csv
        from pathlib import Path
        
        if not final_results:
            console.print("[yellow]No results to export.[/yellow]")
            return
            
        # Get all metrics
        all_metrics = set()
        for metrics in final_results.values():
            all_metrics.update(metrics.keys())
        
        # Sort metrics for consistent ordering
        sorted_metrics = sorted(all_metrics)
        model_names = sorted(final_results.keys())
        
        # Write to CSV
        filepath = Path(filename)
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Header row
            header = ['Model'] + sorted_metrics
            writer.writerow(header)
            
            # Data rows
            for model_name in model_names:
                row = [model_name]
                for metric in sorted_metrics:
                    value = final_results[model_name].get(metric, 'N/A')
                    # Clean up the value for CSV
                    if isinstance(value, str) and '±' in value:
                        # For values like "0.8500 ± 0.0200", just take the mean
                        value = value.split('±')[0].strip()
                    row.append(value)
                writer.writerow(row)
        
        console.print(f"[green]Results exported to {filepath}[/green]")