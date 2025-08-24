import os
import subprocess
import json
import argparse
from pathlib import Path
import optuna
import pandas as pd
from rich.console import Console
from rich.live import Live
from rich.table import Table
import sys

def _run_command(command: list[str], console: Console, error_message: str):
    """Executes a command and handles potential errors."""
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        console.print(f"[bold red]{error_message}[/bold red]")
        console.print(e.stdout)
        console.print(e.stderr)
        raise

def run_model(args, console, model_name, config_name, hparams=None, trial_num=None, live_active=False):
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
            sys.executable, dataset_builder_script,
            f"--output-dir={data_dir}",
            f"--num-aug={num_aug}"
        ]
        if args.dataset == "synthetic":
            build_command.append(f"--task-type={args.synthetic_task}")
            build_command.append("--num-samples=10")

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
        # For HREM, the base config is hrm_v1, which we'll override.
        base_config = "hrm_v1"
        hparam_args.extend(["arch.name=hrm.hrem@HREM", "+arch.use_memory=True"])

        new_hrem_params = ["m_loc", "d_mem", "top_k"]

        for key, value in hparams.items():
            if key in new_hrem_params:
                hparam_args.append(f"+arch.{key}={value}")
            else:
                hparam_args.append(f"arch.{key}={value}")

    log_path = f"results_{model_name}_{trial_num if trial_num is not None else ''}.json"

    run_name = f"{model_name}_{args.dataset}"
    if trial_num is not None:
        run_name += f"_trial_{trial_num}"

    command = [
        sys.executable, "-m", "torch.distributed.run",
        "--nproc-per-node", "1",
        "--rdzv-backend", "c10d",
        "--rdzv-endpoint", "localhost:0",
        "pretrain.py",
        f"data_path={data_dir}",
        f"epochs={epochs}",
        f"eval_interval={eval_interval}",
        f"arch={base_config}",
        f"+log_path={log_path}",
        f"+project_name=HREM_vs_HRM_Opt",
        f"+run_name={run_name}"
    ]
    if hparam_args:
        command.extend(hparam_args)

    if smoke_test:
        command.extend([
            "+smoke_test=True",
            "arch.puzzle_emb_ndim=16", "arch.num_heads=1",
            "arch.expansion=1.0", "global_batch_size=1", "checkpoint_every_eval=True"
        ])

    run_msg = f"[bold green]Running {model_name}..."
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

    if Path(log_path).exists():
        Path(log_path).unlink()

    return final_metrics

def objective(trial: optuna.trial.Trial, args, console, live_active: bool = False):
    if args.smoke_test:
        params = {
            "m_loc": trial.suggest_int("m_loc", 64, 128),
            "d_mem": trial.suggest_int("d_mem", 64, 128),
            "top_k": trial.suggest_int("top_k", 2, 4),
            "H_layers": trial.suggest_int("H_layers", 1, 2),
            "L_layers": trial.suggest_int("L_layers", 1, 2),
            "H_cycles": trial.suggest_int("H_cycles", 1, 2),
            "L_cycles": trial.suggest_int("L_cycles", 1, 4),
            "hidden_size": trial.suggest_categorical("hidden_size", [16, 32]),
        }
    else:
        params = {
            "m_loc": trial.suggest_int("m_loc", 64, 256),
            "d_mem": trial.suggest_int("d_mem", 64, 256),
            "top_k": trial.suggest_int("top_k", 2, 8),
            "H_layers": trial.suggest_int("H_layers", 1, 4),
            "L_layers": trial.suggest_int("L_layers", 1, 4),
            "H_cycles": trial.suggest_int("H_cycles", 1, 4),
            "L_cycles": trial.suggest_int("L_cycles", 1, 16),
            "hidden_size": trial.suggest_categorical("hidden_size", [128, 256, 512]),
        }

    try:
        metrics = run_model(args, console, "HREM", "hrm_v1", hparams=params, trial_num=trial.number, live_active=live_active)
        return metrics.get('test/all/total_loss', float('inf'))
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        console.print(f"[bold red]Trial {trial.number} failed.[/bold red]")
        return float('inf')


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--dataset", type=str, default="arc", choices=["arc", "sudoku", "maze", "synthetic"])
    parser.add_argument("--num-aug", type=int, default=0)
    parser.add_argument("--synthetic-task", type=str, default="copy", choices=["copy", "reverse"])
    parser.add_argument("--n-trials", type=int, default=10)
    parser.add_argument("--study-name", type=str, default="hrem_optimization")
    parser.add_argument("--storage", type=str, default="sqlite:///optuna_hrem.db")
    args = parser.parse_args()

    console = Console()

    storage_name = args.storage
    study_name = args.study_name
    if args.smoke_test:
        study_name += "-smoke"


    if storage_name:
        console.print(f"Using storage: {storage_name}")
        console.print(f"To see live results, run: optuna-dashboard {storage_name}")

    table = Table(title=f"Hyperparameter Optimization for HREM (Study: {study_name})")
    table.add_column("Trial", justify="right", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")
    table.add_column("Params", style="green")
    table.add_column("State", style="yellow")

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        direction="minimize",
        load_if_exists=True,
    )

    with Live(table, console=console, screen=True, redirect_stderr=False) as live:
        callback = RichCallback(table, live)
        study.optimize(lambda trial: objective(trial, args, console, live_active=True), n_trials=args.n_trials, callbacks=[callback])

    console.print("\n[bold green]Optimization finished.[/bold green]")

    if len(study.trials) == 0:
        console.print("[bold yellow]No trials were completed. Skipping final comparison.[/bold yellow]")
        return

    best_trial = study.best_trial
    if best_trial is None:
        console.print("[bold yellow]No best trial found. Skipping final comparison.[/bold yellow]")
        return

    console.print(f"[bold]Best trial: {best_trial.number}[/bold]")
    console.print(f"  [bold]Value[/bold]: {best_trial.value}")
    console.print("  [bold]Params[/bold]:")
    for key, value in best_trial.params.items():
        console.print(f"    [green]{key}[/green]: {value}")

    # --- Save results ---
    results_df = study.trials_dataframe()
    results_df.to_csv(f"{study_name}_results.csv", index=False)
    console.print(f"\n[bold]Saved optimization results to {study_name}_results.csv[/bold]")


    console.print("\n[bold blue]--- Final Comparison ---[/bold blue]")

    hrm_metrics = run_model(args, console, "HRM", "hrm_v1")
    best_hrem_metrics = run_model(args, console, "HREM_best", "hrm_v1", hparams=best_trial.params)

    comparison_table = Table(title="HRM vs Best HREM")
    comparison_table.add_column("Metric", style="cyan")
    comparison_table.add_column("HRM", style="magenta")
    comparison_table.add_column("Best HREM", style="green")

    all_keys = set(hrm_metrics.keys()) | set(best_hrem_metrics.keys())
    for key in sorted(all_keys):
        if key != 'step':
            comparison_table.add_row(
                key,
                str(hrm_metrics.get(key, "N/A")),
                str(best_hrem_metrics.get(key, "N/A"))
            )

    console.print(comparison_table)

    # --- Generate Markdown Report ---
    report = []
    summary = []

    report.append("## Best HREM Parameters")
    report.append("| Parameter | Value |")
    report.append("|---|---|")
    for key, value in best_trial.params.items():
        report.append(f"| {key} | {value} |")
    report.append("\n")

    report.append("## Final Metrics Comparison")
    report.append("| Metric | HRM | Best HREM |")
    report.append("|---|---|---|")

    for key in sorted(all_keys):
        if key != 'step':
            report.append(f"| {key} | {hrm_metrics.get(key, 'N/A')} | {best_hrem_metrics.get(key, 'N/A')} |")

    summary.append(f"**Best HREM**: Final loss = {best_hrem_metrics.get('test/all/total_loss', 'N/A')}")
    summary.append(f"**HRM**: Final loss = {hrm_metrics.get('test/all/total_loss', 'N/A')}")

    with open("comparison_report.md", "w") as f:
        f.write(f"# HREM vs HRM Performance Comparison ({study_name})\n\n")
        f.write("## Summary\n\n")
        f.write("\n".join(summary))
        f.write("\n\n")
        f.write("\n".join(report))

    console.print("[bold]Generated comparison report: comparison_report.md[/bold]")


if __name__ == "__main__":
    main()
