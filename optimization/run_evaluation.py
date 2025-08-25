import argparse
from pathlib import Path
from rich.console import Console
from rich.table import Table

from optimization.utils import run_model, aggregate_metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--dataset", type=str, default="arc", choices=["arc", "sudoku", "maze", "synthetic"])
    parser.add_argument("--num-aug", type=int, default=0)
    parser.add_argument("--synthetic-task", type=str, default="copy", choices=["copy", "reverse"])
    parser.add_argument("--n-runs", type=int, default=1, help="Number of runs for statistical significance.")
    parser.add_argument("--study-name", type=str, default="hrem_evaluation", help="Name for the evaluation.")
    args = parser.parse_args()

    if args.smoke_test:
        args.dataset = "synthetic"

    console = Console()
    study_name = f"{args.study_name}-smoke" if args.smoke_test else args.study_name

    models_to_run = {
        "HRM": {"config": "hrm_v1"},
        "HREM": {
            "config": "hrm_v1",
            "hparams": {
                "name": "hrm.hrem@HREM",
                "use_memory": True,
                "m_loc": 128,
                "d_mem": 128,
                "top_k": 4,
                "sparse_addressing": True,
                "use_location_addressing": True,
            },
        },
    }

    all_metrics = {}
    for model_name, model_info in models_to_run.items():
        console.print(f"\n[bold blue]--- Running Evaluation for {model_name} ---[/bold blue]")
        if args.n_runs > 1:
            console.print(f"Running evaluation {args.n_runs} times for statistical significance...")

        metrics_list = [run_model(args, console, model_name, model_info["config"], hparams=model_info.get("hparams"), study_name=study_name, run_idx=i) for i in range(args.n_runs)]
        all_metrics[model_name] = aggregate_metrics(metrics_list)

    console.print("\n[bold blue]--- Comparison Report ---[/bold blue]")
    comparison_table = Table(title=f"HRM vs HREM ({study_name})")
    comparison_table.add_column("Metric", style="cyan")

    model_names = list(models_to_run.keys())
    for model_name in model_names:
        comparison_table.add_column(model_name, style="magenta" if model_name == "HRM" else "green")

    all_keys = set()
    for model_name in model_names:
        all_keys.update(all_metrics.get(model_name, {}).keys())

    for key in sorted(all_keys):
        row = [key]
        for model_name in model_names:
            row.append(str(all_metrics.get(model_name, {}).get(key, "N/A")))
        comparison_table.add_row(*row)

    console.print(comparison_table)

    report = [
        f"# HREM vs HRM Performance Comparison ({study_name})\n",
        "## Summary",
        *[f"**{name}**: Final loss = {all_metrics.get(name, {}).get('test/all/total_loss', 'N/A')}" for name in model_names],
    ]

    for model_name, model_info in models_to_run.items():
        if "hparams" in model_info:
            report.append(f"\n## {model_name} Parameters")
            report.append("| Parameter | Value |")
            report.append("|---|---|")
            for key, value in model_info["hparams"].items():
                report.append(f"| {key} | {value} |")

    report.append("\n## Final Metrics Comparison")
    report.append("| Metric | " + " | ".join(model_names) + " |")
    report.append("|---|" + "---|"*len(model_names))
    report.append(*[f"| {key} | " + " | ".join([str(all_metrics.get(name, {}).get(key, 'N/A')) for name in model_names]) + " |" for key in sorted(all_keys)])

    report_path = Path(f"optimization/comparison_report_{study_name}.md")
    with open(report_path, "w") as f:
        f.write("\n".join(report))
    console.print(f"[bold]Generated comparison report: {report_path}[/bold]")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        console = Console()
        console.print(f"[bold red]An unexpected error occurred: {e}[/bold red]")
        # Optionally, print traceback for debugging
        # console.print_exception()
