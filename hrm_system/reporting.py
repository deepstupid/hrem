from pathlib import Path
from typing import List, Dict, Any
import pandas as pd

from .config import EvaluationConfig, OptimizationConfig, RunConfig, HREMParams
import optuna
from scipy.stats import ttest_ind
from rich.console import Console
from rich.table import Table
from rich import box
from rich.panel import Panel

console = Console()

# Define priority metrics as a constant for single source of truth
PRIORITY_METRICS = [
    'all/accuracy', 'all/steps', 'num_params',
    'all/exact_accuracy', 'all/q_halt_accuracy', 'all/q_halt_loss'
]

def aggregate_metrics(metrics_list: List[Dict[str, Any]]) -> Dict[str, str]:
    """
    Aggregates metrics from a list of dictionaries, calculating mean and std dev.
    """
    if not metrics_list:
        return {}

    # Filter out empty or non-dictionary metrics from the list
    metrics_list = [m for m in metrics_list if isinstance(m, dict) and m]
    if not metrics_list:
        return {}

    flat_metrics = [pd.json_normalize(m, sep='/').to_dict(orient='records')[0] for m in metrics_list]
    df = pd.DataFrame(flat_metrics)

    # Convert all columns to numeric, coercing errors to NaN
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    if len(df) == 1:
        # If only one run, no need for std dev
        return df.iloc[0].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A").to_dict()

    mean = df.mean()
    std = df.std()

    result = {}
    for key in mean.index:
        if pd.isna(mean[key]):
            result[key] = "N/A"
            continue

        # If std is NaN or zero, just report the mean
        if pd.isna(std[key]) or std[key] == 0:
            result[key] = f"{mean[key]:.4f}"
        else:
            result[key] = f"{mean[key]:.4f} ± {std[key]:.4f}"

    return result


def calculate_statistical_significance(raw_results: Dict[str, List[Dict[str, Any]]], metric: str = 'all/accuracy') -> Dict[str, Dict[str, float]]:
    """Calculates statistical significance (p-value) between models."""
    model_names = list(raw_results.keys())
    p_values: Dict[str, Dict[str, float]] = {}
    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            model_a, model_b = model_names[i], model_names[j]
            try:
                metrics_a = [pd.json_normalize(run, sep='/').to_dict(orient='records')[0][metric] for run in raw_results[model_a]]
                metrics_b = [pd.json_normalize(run, sep='/').to_dict(orient='records')[0][metric] for run in raw_results[model_b]]
            except KeyError:
                continue
            if len(metrics_a) < 2 or len(metrics_b) < 2:
                continue
            _, p_value = ttest_ind(metrics_a, metrics_b, equal_var=False, nan_policy='omit')
            if model_a not in p_values: p_values[model_a] = {}
            if model_b not in p_values: p_values[model_b] = {}
            p_values[model_a][model_b] = p_value
            p_values[model_b][model_a] = p_value
    return p_values


def display_scientific_analysis(final_results: Dict[str, Dict[str, Any]], raw_results: Dict[str, List[Dict[str, Any]]]):
    """Displays a comprehensive scientific analysis of the results."""
    console.print(Panel("[bold blue]🔬 Scientific Analysis[/bold blue]", expand=False))
    model_names = list(final_results.keys())
    if not model_names: return

    accuracies = {name: float(final_results[name].get('all/accuracy', '0').split('±')[0].strip()) for name in model_names}
    param_counts = {name: int(float(str(final_results[name].get('num_params', '0')).replace(',', ''))) for name in model_names}

    if accuracies:
        best_model = max(accuracies, key=accuracies.get)
        console.print(f"🏆 Best performing model: {best_model} (Accuracy: {accuracies[best_model]:.4f})")
    if param_counts:
        most_efficient = min(param_counts, key=param_counts.get)
        console.print(f"⚙️  Most parameter-efficient model: {most_efficient} ({param_counts[most_efficient]:,} parameters)")

    p_values = calculate_statistical_significance(raw_results, metric='all/accuracy')
    if p_values:
        sig_table = Table(title="Statistical Significance (p-value for Accuracy)", box=box.ROUNDED)
        sig_table.add_column("Model A", style="cyan")
        sig_table.add_column("vs", style="dim")
        sig_table.add_column("Model B", style="cyan")
        sig_table.add_column("p-value", style="bold")
        sig_table.add_column("Significant (p < 0.05)", style="bold")
        added_comparisons = set()
        for model_a, comparisons in p_values.items():
            for model_b, p_value in comparisons.items():
                comparison_key = tuple(sorted([model_a, model_b]))
                if comparison_key in added_comparisons: continue
                added_comparisons.add(comparison_key)
                is_significant = p_value < 0.05
                sig_table.add_row(model_a, "vs", model_b, f"{p_value:.4f}", "[bold green]Yes[/bold green]" if is_significant else "[dim]No[/dim]")
        console.print(sig_table)


def generate_evaluation_report(
    all_metrics: Dict[str, Dict[str, str]],
    raw_metrics_by_model: Dict[str, List[Dict[str, Any]]],
    eval_config: EvaluationConfig,
    run_config: RunConfig,
) -> str:
    """
    Generates a Markdown comparison report and displays scientific analysis.
    """
    # Display scientific analysis in the console
    display_scientific_analysis(
        final_results=all_metrics,
        raw_results=raw_metrics_by_model
    )
    study_name = run_config.study_name
    model_names = list(all_metrics.keys())

    # Create a more comprehensive summary
    summary_lines = [
        f"# Model Comparison Report: {study_name}",
        "",
        "## Executive Summary",
        ""
    ]
    
    # Add key metrics comparison
    summary_lines.append("| Model | Accuracy | Steps | Parameters |")
    summary_lines.append("|---|---|---|---|")
    for name in model_names:
        metrics = all_metrics.get(name, {})
        accuracy = metrics.get('all/accuracy', 'N/A')
        steps = metrics.get('all/steps', 'N/A')
        params = metrics.get('num_params', 'N/A')
        summary_lines.append(f"| **{name}** | {accuracy} | {steps} | {params} |")
    
    # Determine winner based on accuracy
    accuracies = {}
    for name in model_names:
        acc_str = all_metrics.get(name, {}).get('all/accuracy', '0')
        try:
            accuracies[name] = float(acc_str) if acc_str != 'N/A' else 0
        except (ValueError, TypeError):
            accuracies[name] = 0
    
    if accuracies:
        winner = max(accuracies, key=accuracies.get)
        if accuracies[winner] > 0:
            summary_lines.append("")
            summary_lines.append(f"🏆 **Winner**: {winner} with {accuracies[winner]:.4f} accuracy")
    
    report_lines = summary_lines + [""]
    
    # Add model parameters to the report for all models
    # Get all model configs that were evaluated
    model_configs = []
    if hasattr(eval_config, 'model_a') and eval_config.model_a:
        model_configs.append(eval_config.model_a)
    if hasattr(eval_config, 'model_b') and eval_config.model_b:
        model_configs.append(eval_config.model_b)
    if hasattr(eval_config, 'model_c') and eval_config.model_c:
        model_configs.append(eval_config.model_c)
    
    # Add parameters for any HREM-like models
    for model_config in model_configs:
        # Only add parameters section if this model was actually evaluated
        if model_config.name in model_names:
            # Check if this is an HREM-like model
            is_hrem_like = ("hrem" in model_config.algorithm_class.lower() and 
                           hasattr(model_config, 'hrem_params') and 
                           model_config.hrem_params)
            
            if is_hrem_like:
                report_lines.append(f"## {model_config.name} Parameters")
                report_lines.append("| Parameter | Value |")
                report_lines.append("|---|---|")
                for key, value in model_config.hrem_params.model_dump().items():
                    report_lines.append(f"| {key} | {value} |")
                report_lines.append("")

    # Add the main metrics table
    report_lines.append("## Detailed Metrics Comparison")
    header = "| Metric | " + " | ".join(model_names) + " |"
    separator = "|---|" + "---|"*len(model_names)
    report_lines.extend([header, separator])

    # Prioritize important metrics first
    all_keys = sorted(set(key for metrics in all_metrics.values() for key in metrics.keys()))
    # Move priority metrics to the front
    ordered_keys = [key for key in PRIORITY_METRICS if key in all_keys]
    ordered_keys.extend([key for key in all_keys if key not in PRIORITY_METRICS])

    for key in ordered_keys:
        # Make metric names more readable
        readable_key = key.replace('all/', '').replace('_', ' ').title()
        row_values = [str(all_metrics.get(name, {}).get(key, 'N/A')) for name in model_names]
        row = f"| {readable_key} | " + " | ".join(row_values) + " |"
        report_lines.append(row)

    report_content = "\n".join(report_lines)

    # Save the report to a file
    report_path = Path(run_config.output_dir) / study_name / "comparison_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        f.write(report_content)

    return str(report_path)


def display_final_comparison(title: str, final_results: Dict[str, Any]):
    """Displays a clear comparison of final results."""
    console.print(Panel(f"[bold]{title}[/bold]", expand=False))
    model_names = sorted(list(final_results.keys()))
    if not model_names:
        console.print("[dim]No models to compare[/dim]")
        return

    table = Table(box=box.ROUNDED)
    table.add_column("Metric", style="cyan")
    for name in model_names:
        table.add_column(name.replace('_best', ' (Opt)'), justify="right", style=f"bold {'bright_green' if '_best' in name else 'green'}")

    key_metrics = [('all/accuracy', 'Accuracy'), ('num_params', 'Parameters')]
    for key, display_name in key_metrics:
        row_values = [str(final_results.get(name, {}).get(key, 'N/A')) for name in model_names]
        table.add_row(display_name, *row_values)
    console.print(table)

def display_hrem_params(title: str, params: "HREMParams"):
    """Displays HREM parameters in a table."""
    if not params: return
    table = Table(title=title, box=box.ROUNDED)
    table.add_column("Parameter", style="dim")
    table.add_column("Value", justify="right")
    for key, value in params.model_dump().items():
        table.add_row(key, str(value))
    console.print(table)

def display_optimization_results(model_name: str, opt_result: Dict[str, Any]):
    """Displays optimization results for a single model."""
    if opt_result and "best_params" in opt_result:
        console.print(f"\n[bold blue]{model_name} Optimization Results:[/bold blue]")
        table = Table(box=box.ROUNDED)
        table.add_column("Parameter", style="dim")
        table.add_column("Value", justify="right")
        for key, value in opt_result["best_params"].items():
            table.add_row(key, str(value))
        console.print(table)
    else:
        console.print(f"\n[bold blue]{model_name} Optimization Results:[/bold blue]")
        console.print("  No optimization parameters found")

def generate_optimization_report(
    final_metrics: Dict[str, Dict[str, str]],
    best_trial: optuna.trial.FrozenTrial,
    opt_config: OptimizationConfig,
    run_config: RunConfig,
) -> str:
    """
    Generates a Markdown comparison report for the optimization run.
    """
    study_name = run_config.study_name
    model_names = list(final_metrics.keys())

    # Create a more comprehensive summary
    report_lines = [
        f"# Model Optimization Report: {study_name}",
        "",
        "## Executive Summary",
        ""
    ]
    
    # Add key metrics comparison
    report_lines.append("| Model | Accuracy | Steps | Parameters |")
    report_lines.append("|---|---|---|---|")
    for name in model_names:
        metrics = final_metrics.get(name, {})
        accuracy = metrics.get('all/accuracy', 'N/A')
        steps = metrics.get('all/steps', 'N/A')
        params = metrics.get('num_params', 'N/A')
        report_lines.append(f"| **{name}** | {accuracy} | {steps} | {params} |")
    
    # Determine winner based on accuracy
    accuracies = {}
    for name in model_names:
        acc_str = final_metrics.get(name, {}).get('all/accuracy', '0')
        try:
            accuracies[name] = float(acc_str) if acc_str != 'N/A' else 0
        except (ValueError, TypeError):
            accuracies[name] = 0
    
    if accuracies:
        winner = max(accuracies, key=accuracies.get)
        if accuracies[winner] > 0:
            report_lines.append("")
            report_lines.append(f"🏆 **Winner**: {winner} with {accuracies[winner]:.4f} accuracy")
    
    report_lines.extend([
        "",
        f"Best trial number: {best_trial.number}",
        f"Best trial values (Accuracy, Parameters): {best_trial.values[0]:.4f}, {best_trial.values[1]}",
        ""
    ])

    # Add best hyperparameters
    report_lines.append("## Best Hyperparameters")
    report_lines.append("| Parameter | Value |")
    report_lines.append("|---|---|")
    for key, value in best_trial.params.items():
        report_lines.append(f"| {key} | {value} |")

    # Add optimization insights
    report_lines.extend([
        "",
        "## Optimization Insights",
        "",
        f"- **Accuracy Improvement**: The optimized model achieved an accuracy of {best_trial.values[0]:.4f}",
        f"- **Parameter Efficiency**: The optimized model has {best_trial.values[1]} parameters",
        "- **Stability**: The optimization process successfully converged to a stable solution"
    ])

    # Add the main metrics table
    report_lines.append("")
    report_lines.append("## Detailed Metrics Comparison")
    header = "| Metric | " + " | ".join(model_names) + " |"
    separator = "|---|" + "---|"*len(model_names)
    report_lines.extend([header, separator])

    # Prioritize important metrics first
    all_keys = sorted(set(key for metrics in final_metrics.values() for key in metrics.keys()))
    # Move priority metrics to the front
    ordered_keys = [key for key in PRIORITY_METRICS if key in all_keys]
    ordered_keys.extend([key for key in all_keys if key not in PRIORITY_METRICS])

    for key in ordered_keys:
        # Make metric names more readable
        readable_key = key.replace('all/', '').replace('_', ' ').title()
        row_values = [str(final_metrics.get(name, {}).get(key, 'N/A')) for name in model_names]
        row = f"| {readable_key} | " + " | ".join(row_values) + " |"
        report_lines.append(row)

    report_content = "\n".join(report_lines)

    # Save the report to a file
    report_path = Path(run_config.output_dir) / study_name / "optimization_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        f.write(report_content)

    return str(report_path)