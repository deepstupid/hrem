import os
import subprocess
import json
import argparse
from pathlib import Path
import optuna
import pandas as pd
import numpy as np
from rich.console import Console
from rich.live import Live
from rich.table import Table
import sys

LOGS_DIR = Path("logs")

def _run_command(command: list[str], console: Console, error_message: str):
    """Executes a command and handles potential errors."""
    try:
        # Using sys.executable to ensure we use the same python interpreter
        result = subprocess.run(
            [sys.executable, *command],
            check=True,
            capture_output=True,
            text=True
        )
        return result
    except subprocess.CalledProcessError as e:
        console.print(f"[bold red]{error_message}[/bold red]")
        console.print(f"[bold]STDOUT:[/bold]\n{e.stdout}")
        console.print(f"[bold]STDERR:[/bold]\n{e.stderr}")
        raise

def run_model(args, console: Console, model_name: str, config_name: str, hparams=None, trial_num=None, live_active=False, study_name=None, run_idx=0):
    smoke_test = args.smoke_test
    if smoke_test:
        data_dir = f"data/{args.dataset}-smoke"
        num_aug = 0
        epochs = 1
        eval_interval = 1
    else:
        data_dir = f"data/{args.dataset}-full"
        num_aug = args.num_aug
        epochs = 20000
        eval_interval = 2000

    dataset_builder_script = f"dataset/build_{args.dataset}_dataset.py"
    if not os.path.exists(data_dir):
        build_command = [
            dataset_builder_script,
            f"--output-dir={data_dir}",
            f"--num-aug={num_aug}"
        ]
        if args.dataset == "synthetic":
            build_command.extend([
                f"--task-type={args.synthetic_task}",
                "--num-samples=10"
            ])

        build_msg = f"[bold green]Building {args.dataset} dataset..."
        error_msg = "Error running dataset builder:"

        if live_active:
            console.print(build_msg)
            _run_command(build_command, console, error_msg)
        else:
            with console.status(build_msg):
                _run_command(build_command, console, error_msg)

    base_config = config_name
    hparam_args = []
    if hparams:
        base_config = "hrm_v1"
        hparam_args.extend(["arch.name=hrm.hrem@HREM", "+arch.use_memory=True"])
        new_hrem_params = ["m_loc", "d_mem", "top_k"]
        for key, value in hparams.items():
            if key in new_hrem_params:
                hparam_args.append(f"+arch.{key}={value}")
            else:
                hparam_args.append(f"arch.{key}={value}")

    run_name_parts = [model_name, args.dataset]
    if trial_num is not None:
        run_name_parts.append(f"trial_{trial_num}")
    if run_idx > 0:
        run_name_parts.append(f"run_{run_idx}")
    run_name = "_".join(run_name_parts)

    log_path = Path(f"results_{run_name}.json")

    command = [
        "-m", "torch.distributed.run",
        "--nproc-per-node", "1",
        "--rdzv-backend", "c10d",
        "--rdzv-endpoint", "localhost:0",
        "pretrain.py",
        f"data_path={data_dir}",
        f"epochs={epochs}",
        f"eval_interval={eval_interval}",
        f"arch={base_config}",
        f"+log_path={str(log_path)}",
        f"+project_name=HREM_vs_HRM_Opt",
        f"+run_name={run_name}",
        *hparam_args
    ]

    if smoke_test:
        command.extend([
            "+smoke_test=True",
            "arch.puzzle_emb_ndim=16", "arch.num_heads=1",
            "arch.expansion=1.0", "global_batch_size=1", "checkpoint_every_eval=True"
        ])

    run_msg = f"[bold green]Running {model_name} (trial: {trial_num}, run: {run_idx})...[/bold green]"
    error_msg = f"Error running pretrain.py for {model_name}:"

    if live_active:
        console.print(run_msg)
        _run_command(command, console, error_msg)
    else:
        with console.status(run_msg):
            _run_command(command, console, error_msg)

    with open(log_path, "r") as f:
        data = json.load(f)
    final_metrics = data[-1]

    if log_path.exists():
        if trial_num is not None and study_name:
            log_dir = LOGS_DIR / study_name
            log_dir.mkdir(parents=True, exist_ok=True)
            new_log_path = log_dir / f"{run_name}.json"
            log_path.rename(new_log_path)
        else:
            log_path.unlink()

    return final_metrics
def objective(trial: optuna.trial.Trial, args, console: Console, study_name: str, live_active: bool = False):
    params = {
        "m_loc": trial.suggest_int("m_loc", 64, 128 if args.smoke_test else 256),
        "d_mem": trial.suggest_int("d_mem", 64, 128 if args.smoke_test else 256),
        "top_k": trial.suggest_int("top_k", 2, 4 if args.smoke_test else 8),
        "H_layers": trial.suggest_int("H_layers", 1, 2 if args.smoke_test else 4),
        "L_layers": trial.suggest_int("L_layers", 1, 2 if args.smoke_test else 4),
        "H_cycles": trial.suggest_int("H_cycles", 1, 2 if args.smoke_test else 4),
        "L_cycles": trial.suggest_int("L_cycles", 1, 4 if args.smoke_test else 16),
        "hidden_size": trial.suggest_categorical("hidden_size", [16, 32] if args.smoke_test else [128, 256, 512]),
    }

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

def aggregate_metrics(metrics_list: list[dict]):
    if not metrics_list:
        return {}

    flat_metrics = [pd.json_normalize(m, sep='/').to_dict(orient='records')[0] for m in metrics_list]
    df = pd.DataFrame(flat_metrics)

    if len(df) == 1:
        return df.iloc[0].apply(lambda x: f"{x:.4f}").to_dict()

    mean = df.mean()
    std = df.std()

    result = {key: f"{mean[key]:.4f} ± {std[key]:.4f}" for key in mean.index}
    return result

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
    args = parser.parse_args()

    if args.smoke_test:
        args.dataset = "synthetic"

    console = Console()
    study_name = f"{args.study_name}-smoke" if args.smoke_test else args.study_name

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
        study.optimize(lambda trial: objective(trial, args, console, study_name, live_active=True), n_trials=args.n_trials, callbacks=[callback])

    console.print("\n[bold green]Optimization finished.[/bold green]")

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
    results_df.to_csv(f"{study_name}_results.csv", index=False)
    console.print(f"\n[bold]Saved optimization results to {study_name}_results.csv[/bold]")

    console.print("\n[bold blue]--- Final Comparison ---[/bold blue]")
    if args.n_final_runs > 1:
        console.print(f"Running final comparison {args.n_final_runs} times for statistical significance...")

    hrm_metrics_list = [run_model(args, console, "HRM", "hrm_v1", study_name=study_name, run_idx=i) for i in range(args.n_final_runs)]
    best_hrem_metrics_list = [run_model(args, console, "HREM_best", "hrm_v1", hparams=best_trial.params, study_name=study_name, run_idx=i) for i in range(args.n_final_runs)]

    hrm_metrics = aggregate_metrics(hrm_metrics_list)
    best_hrem_metrics = aggregate_metrics(best_hrem_metrics_list)

    comparison_table = Table(title="HRM vs Best HREM")
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

    with open("comparison_report.md", "w") as f:
        f.write("\n".join(report))
    console.print("[bold]Generated comparison report: comparison_report.md[/bold]")

if __name__ == "__main__":
    main()
