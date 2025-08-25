from pathlib import Path
from typing import List, Dict, Any
import pandas as pd

from .config import EvaluationConfig, OptimizationConfig, RunConfig
import optuna

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

    report_lines = [
        f"# Model Comparison Report: {study_name}\n",
        "## Summary",
        *[f"**{name}**: Final loss = {all_metrics.get(name, {}).get('all/lm_loss', 'N/A')}" for name in model_names],
    ]

    # Add model parameters to the report
    models_in_report = [eval_config.model_a, eval_config.model_b]
    for model_config in models_in_report:
        # A simple check to see if this is an HREM-like model
        if "hrem" in model_config.algorithm_class.lower() and model_config.hrem_params:
            report_lines.append(f"\n## {model_config.name} Parameters")
            report_lines.append("| Parameter | Value |")
            report_lines.append("|---|---|")
            for key, value in model_config.hrem_params.model_dump().items():
                report_lines.append(f"| {key} | {value} |")

    # Add the main metrics table
    report_lines.append("\n## Final Metrics Comparison")
    header = "| Metric | " + " | ".join(model_names) + " |"
    separator = "|---|" + "---|"*len(model_names)
    report_lines.extend([header, separator])

    all_keys = sorted(set(key for metrics in all_metrics.values() for key in metrics.keys()))

    for key in all_keys:
        row_values = [str(all_metrics.get(name, {}).get(key, 'N/A')) for name in model_names]
        row = f"| {key} | " + " | ".join(row_values) + " |"
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

    report_lines = [
        f"# HREM Optimization Report: {study_name}\n",
        "## Summary",
        *[f"**{name}**: Final loss = {final_metrics.get(name, {}).get('all/lm_loss', 'N/A')}" for name in model_names],
        f"\nBest trial number: {best_trial.number}",
        f"Best trial value (loss): {best_trial.value:.4f}",
    ]

    # Add best hyperparameters
    report_lines.append("\n## Best HREM Parameters")
    report_lines.append("| Parameter | Value |")
    report_lines.append("|---|---|")
    for key, value in best_trial.params.items():
        report_lines.append(f"| {key} | {value} |")

    # Add the main metrics table
    report_lines.append("\n## Final Metrics Comparison")
    header = "| Metric | " + " | ".join(model_names) + " |"
    separator = "|---|" + "---|"*len(model_names)
    report_lines.extend([header, separator])

    all_keys = sorted(set(key for metrics in final_metrics.values() for key in metrics.keys()))

    for key in all_keys:
        row_values = [str(final_metrics.get(name, {}).get(key, 'N/A')) for name in model_names]
        row = f"| {key} | " + " | ".join(row_values) + " |"
        report_lines.append(row)

    report_content = "\n".join(report_lines)

    # Save the report to a file
    report_path = Path(run_config.output_dir) / study_name / "optimization_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        f.write(report_content)

    return str(report_path)
