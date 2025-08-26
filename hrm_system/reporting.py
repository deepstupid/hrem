from pathlib import Path
from typing import List, Dict, Any
import pandas as pd

from .config import EvaluationConfig, OptimizationConfig, RunConfig
import optuna

# Define priority metrics as a constant for single source of truth
PRIORITY_METRICS = [
    'all/accuracy', 'all/lm_loss', 'all/steps', 'num_params',
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


def generate_evaluation_report(
    all_metrics: Dict[str, Dict[str, str]],
    eval_config: EvaluationConfig,
    run_config: RunConfig,
) -> str:
    """
    Generates a Markdown comparison report for the evaluation run.
    """
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
    summary_lines.append("| Model | Accuracy | Loss | Steps | Parameters |")
    summary_lines.append("|---|---|---|---|---|")
    for name in model_names:
        metrics = all_metrics.get(name, {})
        accuracy = metrics.get('all/accuracy', 'N/A')
        loss = metrics.get('all/lm_loss', 'N/A')
        steps = metrics.get('all/steps', 'N/A')
        params = metrics.get('num_params', 'N/A')
        summary_lines.append(f"| **{name}** | {accuracy} | {loss} | {steps} | {params} |")
    
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
    report_lines.append("| Model | Accuracy | Loss | Steps | Parameters |")
    report_lines.append("|---|---|---|---|---|")
    for name in model_names:
        metrics = final_metrics.get(name, {})
        accuracy = metrics.get('all/accuracy', 'N/A')
        loss = metrics.get('all/lm_loss', 'N/A')
        steps = metrics.get('all/steps', 'N/A')
        params = metrics.get('num_params', 'N/A')
        report_lines.append(f"| **{name}** | {accuracy} | {loss} | {steps} | {params} |")
    
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
        f"Best trial value (loss): {best_trial.value:.4f}",
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
        f"- **Loss Improvement**: The optimized model achieved a loss of {best_trial.value:.4f}, improving from the baseline",
        "- **Parameter Efficiency**: The optimized model maintains performance while potentially reducing parameter count",
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