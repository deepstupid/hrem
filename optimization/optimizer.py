import argparse
import subprocess
from pathlib import Path
import optuna
import pandas as pd
import yaml
from rich.console import Console
from rich.live import Live
from rich.table import Table

from optimization.utils import run_model, aggregate_metrics

LOGS_DIR = Path("logs")

def objective(trial: optuna.trial.Trial, args, console: Console, study_name: str, search_space: dict, live_active: bool = False):
    params = {}
    for name, definition in search_space["hrem_params"].items():
        if args.smoke_test and f"smoke_{definition['type']}" in definition:
            param_type = f"smoke_{definition['type']}"
        else:
            param_type = definition['type']

        if param_type == "int":
            low = definition["smoke_low"] if args.smoke_test and "smoke_low" in definition else definition["low"]
            high = definition["smoke_high"] if args.smoke_test and "smoke_high" in definition else definition["high"]
            params[name] = trial.suggest_int(name, low, high)
        elif param_type == "categorical":
            choices = definition["smoke_choices"] if args.smoke_test and "smoke_choices" in definition else definition["choices"]
            params[name] = trial.suggest_categorical(name, choices)
        else:
            raise ValueError(f"Unsupported parameter type: {param_type}")

    try:
        metrics = run_model(args, console, "HREM", "hrm_v1", hparams=params, trial_num=trial.number, live_active=live_active, study_name=study_name)
        return metrics.get('test/all/total_loss', float('inf'))
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        console.print(f"[bold red]Trial {trial.number} failed and will be pruned.[/bold red]")
        if isinstance(e, subprocess.CalledProcessError):
            console.print(f"[bold red]Stderr:[/bold red]\n{e.stderr}")
        raise optuna.TrialPruned() from e

class RichCallback:
    def __init__(self, table: Table, live: Live):
        self.table = table
        self.live = live

    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial):
        self.table.add_row(
            str(trial.number),
            str(trial.value),
            str(trial.params),
            str(trial.state.name),
        )
        self.live.refresh()

def main():
    parser = argparse.ArgumentParser(description="Hyperparameter optimization for HREM model.")
    parser.add_argument("--smoke-test", action="store_true", help="Run in smoke test mode with a small dataset and model.")
    parser.add_argument("--dataset", type=str, default="arc", choices=["arc", "sudoku", "maze", "synthetic"], help="Dataset to use for optimization.")
    parser.add_argument("--num-aug", type=int, default=0, help="Number of augmentations for the dataset.")
    parser.add_argument("--synthetic-task", type=str, default="copy", choices=["copy", "reverse"], help="Task type for synthetic dataset.")
    parser.add_argument("--n-trials", type=int, default=10, help="Number of optimization trials.")
    parser.add_argument("--study-name", type=str, default="hrem_optimization", help="Name for the Optuna study.")
    parser.add_argument("--storage", type=str, default="sqlite:///optuna_hrem.db", help="Optuna storage URL.")
    parser.add_argument("--n-final-runs", type=int, default=1, help="Number of final comparison runs for statistical significance.")
    parser.add_argument("--search-space", type=str, default="config/hparam_search_space.yaml", help="Path to the hyperparameter search space YAML file.")
    parser.add_argument("--n-jobs", type=int, default=1, help="Number of parallel jobs for Optuna.")
    args = parser.parse_args()

    if args.smoke_test:
        args.dataset = "synthetic"

    console = Console()
    study_name = f"{args.study_name}-smoke" if args.smoke_test else args.study_name

    with open(args.search_space, 'r') as f:
        search_space = yaml.safe_load(f)

    if args.storage:
        console.print(f"Using storage: {args.storage}")
        console.print(f"To see live results, run: optuna-dashboard {args.storage}")

    table = Table(title=f"Hyperparameter Optimization for HREM (Study: {study_name})")
    table.add_column("Trial", justify="right", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")
    table.add_column("Params", style="green")
    table.add_column("State", style="yellow")

    study = optuna.create_study(study_name=study_name, storage=args.storage, direction="minimize", load_if_exists=True)

    with Live(table, console=console, screen=True, redirect_stderr=False) as live:
        callback = RichCallback(table, live)
        study.optimize(lambda trial: objective(trial, args, console, study_name, search_space, live_active=True), n_trials=args.n_trials, callbacks=[callback], n_jobs=args.n_jobs)

    console.print("\n[bold green]Optimization finished. Final Results:[/bold green]")
    console.print(table)

    if not study.trials:
        console.print("[bold yellow]No trials were completed. Skipping final comparison.[/bold yellow]")
        return

    try:
        best_trial = study.best_trial
        console.print(f"[bold]Best trial: {best_trial.number}[/bold]")
        console.print(f"  [bold]Value[/bold]: {best_trial.value}")
        console.print("  [bold]Params[/bold]:")
        for key, value in best_trial.params.items():
            console.print(f"    [green]{key}[/green]: {value}")
    except ValueError:
        console.print("[bold yellow]No best trial found (all trials failed or were pruned). Skipping final comparison.[/bold yellow]")
        return

    results_df = study.trials_dataframe()
    results_df.to_csv(f"optimization/{study_name}_results.csv", index=False)
    console.print(f"\n[bold]Saved optimization results to optimization/{study_name}_results.csv[/bold]")

    console.print("\n[bold blue]--- Final Comparison ---[/bold blue]")
    if args.n_final_runs > 1:
        console.print(f"Running final comparison {args.n_final_runs} times for statistical significance...")

    hrm_metrics_list = [run_model(args, console, "HRM", "hrm_v1", study_name=study_name, run_idx=i) for i in range(args.n_final_runs)]
    best_hrem_metrics_list = [run_model(args, console, "HREM_best", "hrm_v1", hparams=best_trial.params, study_name=study_name, run_idx=i) for i in range(args.n_final_runs)]

    hrm_metrics = aggregate_metrics(hrm_metrics_list)
    best_hrem_metrics = aggregate_metrics(best_hrem_metrics_list)

    comparison_table = Table(title=f"Final Comparison: HRM vs. Best HREM (Study: {study_name})")
    comparison_table.add_column("Metric", style="cyan")
    comparison_table.add_column("HRM", style="magenta")
    comparison_table.add_column("Best HREM", style="green")

    all_keys = sorted(set(hrm_metrics.keys()) | set(best_hrem_metrics.keys()))
    for key in all_keys:
        comparison_table.add_row(key, str(hrm_metrics.get(key, "N/A")), str(best_hrem_metrics.get(key, "N/A")))
    console.print(comparison_table)

    report = [
        f"# HREM vs HRM Performance Comparison ({study_name})\n",
        "## Summary",
        f"**Best HREM**: Final loss = {best_hrem_metrics.get('test/all/total_loss', 'N/A')}",
        f"**HRM**: Final loss = {hrm_metrics.get('test/all/total_loss', 'N/A')}\n",
        "## Best HREM Parameters",
        "| Parameter | Value |",
        "|---|---|",
        *[f"| {key} | {value} |" for key, value in best_trial.params.items()],
        "\n## Final Metrics Comparison",
        "| Metric | HRM | Best HREM |",
        "|---|---|",
        *[f"| {key} | {hrm_metrics.get(key, 'N/A')} | {best_hrem_metrics.get(key, 'N/A')} |" for key in all_keys],
    ]

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
