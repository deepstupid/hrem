"""Scientific reporting utilities for the HRM/HREM demo system."""

from typing import Dict, Any, List, Tuple
import math
from rich.console import Console
from rich.table import Table
from rich import box
from rich.panel import Panel
from scipy.stats import ttest_ind
import pandas as pd

console = Console()

class ScientificReporter:
    """Handles scientific reporting and analysis of results."""
    
    @staticmethod
    def calculate_statistical_significance(
        raw_results: Dict[str, List[Dict[str, Any]]],
        metric: str = 'all/accuracy'
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate statistical significance (p-value) between models for a given metric.
        
        Returns a dictionary with p-values for pairwise comparisons using an independent t-test.
        """
        model_names = list(raw_results.keys())
        p_values: Dict[str, Dict[str, float]] = {}

        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                model_a = model_names[i]
                model_b = model_names[j]

                # Extract metric values for each model from all runs
                try:
                    metrics_a = [
                        pd.json_normalize(run, sep='/').to_dict(orient='records')[0][metric]
                        for run in raw_results[model_a]
                    ]
                    metrics_b = [
                        pd.json_normalize(run, sep='/').to_dict(orient='records')[0][metric]
                        for run in raw_results[model_b]
                    ]
                except KeyError:
                    # Metric not found in one of the models, cannot compare
                    continue

                # Ensure we have enough data for a t-test
                if len(metrics_a) < 2 or len(metrics_b) < 2:
                    continue

                # Perform independent t-test
                t_stat, p_value = ttest_ind(metrics_a, metrics_b, equal_var=False, nan_policy='omit')

                if model_a not in p_values:
                    p_values[model_a] = {}
                if model_b not in p_values:
                    p_values[model_b] = {}

                p_values[model_a][model_b] = p_value
                p_values[model_b][model_a] = p_value
        
        return p_values

    @staticmethod
    def generate_scientific_summary(
        final_results: Dict[str, Dict[str, Any]],
        raw_results: Dict[str, List[Dict[str, Any]]],
    ) -> str:
        """Generate a scientific summary of the results."""
        if not final_results:
            return "No results available for scientific analysis."
        
        model_names = list(final_results.keys())
        summary_parts = []
        
        # Accuracy comparison
        accuracies = {}
        for model_name in model_names:
            acc_str = final_results[model_name].get('all/accuracy', '0').split('±')[0].strip()
            try:
                accuracies[model_name] = float(acc_str)
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
                if isinstance(param_str, str):
                    param_str = param_str.replace(',', '')
                param_counts[model_name] = int(float(param_str))
            except (ValueError, TypeError):
                param_counts[model_name] = 0
        
        if param_counts:
            most_efficient = min(param_counts, key=param_counts.get)
            summary_parts.append(f"⚙️  Most parameter-efficient model: {most_efficient} ({param_counts[most_efficient]:,} parameters)")
        
        # Statistical significance for accuracy
        p_values = ScientificReporter.calculate_statistical_significance(raw_results, metric='all/accuracy')
        significant_comparisons = []
        for model_a, comparisons in p_values.items():
            for model_b, p_value in comparisons.items():
                if p_value < 0.05 and tuple(sorted((model_a, model_b))) not in significant_comparisons:
                    significant_comparisons.append(tuple(sorted((model_a, model_b))))
        
        if significant_comparisons:
            comp_str = ', '.join([f"{m1} vs {m2}" for m1, m2 in significant_comparisons])
            summary_parts.append(f"🔬 Statistically significant differences in accuracy found between: {comp_str}")
        
        return "\n".join(summary_parts)

    @staticmethod
    def display_scientific_analysis(
        final_results: Dict[str, Dict[str, Any]],
        raw_results: Dict[str, List[Dict[str, Any]]],
    ):
        """Display a comprehensive scientific analysis of the results."""
        console.print(Panel("[bold blue]🔬 Scientific Analysis[/bold blue]", expand=False))
        
        # Generate and display summary
        summary = ScientificReporter.generate_scientific_summary(final_results, raw_results)
        console.print(summary)
        
        model_names = list(final_results.keys())
        if not model_names:
            return
            
        # Create detailed analysis table
        table = Table(title="Scientific Metrics Analysis (Aggregated)", show_header=True,
                     header_style="bold magenta", box=box.ROUNDED)
        table.add_column("Metric", style="cyan")
        
        model_styles = {"HRM": "bold blue", "HREM": "bold green", "EnhancedHREM": "bold yellow", "HREM_best": "bold bright_green", "HRM_best": "bold blue"}
        for model_name in model_names:
            style = model_styles.get(model_name, "bold white")
            table.add_column(model_name, justify="right", style=style)
        
        if final_results:
            scientific_metrics = [
                ('all/accuracy', 'Accuracy', '🎯'),
                ('all/lm_loss', 'Loss', '📉'),
                ('all/steps', 'Steps', '⚡'),
                ('num_params', 'Parameters', '⚙️'),
                ('all/exact_accuracy', 'Exact Accuracy', '🎯'),
                ('all/q_halt_accuracy', 'Q-Halt Accuracy', '🎯'),
            ]
            
            for key, display_name, emoji in scientific_metrics:
                has_metric = any(final_results[model].get(key, 'N/A') != 'N/A' for model in model_names)
                if not has_metric:
                    continue
                    
                row_values = [final_results[model].get(key, 'N/A') for model in model_names]
                table.add_row(f"{emoji} {display_name}", *row_values)
                
            console.print(table)
        
        # Show statistical significance for accuracy
        p_values = ScientificReporter.calculate_statistical_significance(raw_results, metric='all/accuracy')
        if p_values:
            sig_table = Table(title="Statistical Significance (p-value for Accuracy)",
                             show_header=True, header_style="bold magenta", box=box.ROUNDED)
            sig_table.add_column("Model A", style="cyan")
            sig_table.add_column("vs", style="dim")
            sig_table.add_column("Model B", style="cyan")
            sig_table.add_column("p-value", style="bold")
            sig_table.add_column("Significant (p < 0.05)", style="bold")

            added_comparisons = set()
            for model_a, comparisons in p_values.items():
                for model_b, p_value in comparisons.items():
                    comparison_key = tuple(sorted([model_a, model_b]))
                    if comparison_key in added_comparisons:
                        continue
                    added_comparisons.add(comparison_key)

                    is_significant = p_value < 0.05
                    sig_status = "[bold green]Yes[/bold green]" if is_significant else "[dim]No[/dim]"
                    p_value_str = f"{p_value:.4f}"
                    sig_table.add_row(model_a, "vs", model_b, p_value_str, sig_status)

            console.print(sig_table)
    
    @staticmethod
    def export_results_to_csv(final_results: Dict[str, Dict[str, Any]], filename: str = "demo_results.csv"):
        """Export results to CSV for further analysis."""
        import csv
        from pathlib import Path
        
        if not final_results:
            console.print("[yellow]No results to export.[/yellow]")
            return
            
        all_metrics = set(key for metrics in final_results.values() for key in metrics.keys())
        sorted_metrics = sorted(list(all_metrics))
        model_names = sorted(final_results.keys())
        
        filepath = Path(filename)
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            header = ['Model'] + sorted_metrics
            writer.writerow(header)
            
            for model_name in model_names:
                row = [model_name]
                for metric in sorted_metrics:
                    value = final_results[model_name].get(metric, 'N/A')
                    if isinstance(value, str) and '±' in value:
                        value = value.split('±')[0].strip()
                    row.append(value)
                writer.writerow(row)
        
        console.print(f"[green]Results exported to {filepath}[/green]")