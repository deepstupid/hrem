import os
import subprocess
import yaml
import json
import argparse
from pathlib import Path
import optuna
from rich.console import Console
from rich.live import Live
from rich.table import Table
import sys

def get_hrem_config(hrm_config_path: str, hrem_config_path: str, params: dict) -> Path:
    """Create HREM config from HRM config and trial params."""
    with open(hrm_config_path, "r") as f:
        hrm_config = yaml.safe_load(f)

    hrem_config = hrm_config.copy()
    hrem_config["name"] = "hrm.hrem@HREM"
    hrem_config["use_memory"] = True
    hrem_config.update(params)

    with open(hrem_config_path, "w") as f:
        yaml.dump(hrem_config, f)

    return Path(hrem_config_path)

def run_model(args, console, model_name, config_name, hparams=None, trial_num=None):
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
        with console.status(f"[bold green]Building {args.dataset} dataset..."):
            build_command = [
                sys.executable, dataset_builder_script,
                f"--output-dir={data_dir}",
                f"--num-aug={num_aug}"
            ]
            if args.dataset == "synthetic":
                build_command.append(f"--task-type={args.synthetic_task}")
                build_command.append("--num-samples=10")
            try:
                subprocess.run(build_command, check=True, capture_output=True, text=True)
            except subprocess.CalledProcessError as e:
                console.print("[bold red]Error running dataset builder:[/bold red]")
                console.print(e.stdout)
                console.print(e.stderr)
                raise

    config_path = Path(f"config/arch/{config_name}.yaml")
    if hparams:
        config_path = Path(f"config/arch/{config_name}_{trial_num}.yaml")
        get_hrem_config("config/arch/hrm_v1.yaml", str(config_path), hparams)

    log_path = f"results_{model_name}_{trial_num if trial_num is not None else ''}.json"

    run_name = f"{model_name}_{args.dataset}"
    if trial_num is not None:
        run_name += f"_trial_{trial_num}"

    command = [
        sys.executable, "-m", "torch.distributed.run", "--nproc-per-node", "1", "pretrain.py",
        f"data_path={data_dir}",
        f"epochs={epochs}",
        f"eval_interval={eval_interval}",
        f"arch={config_path.stem}",
        f"+log_path={log_path}",
        f"+project_name=HREM_vs_HRM_Opt",
        f"+run_name={run_name}"
    ]
    if smoke_test:
        command.extend([
            "+smoke_test=True", "arch.hidden_size=16", "arch.H_layers=1",
            "arch.L_layers=1", "arch.puzzle_emb_ndim=16", "arch.num_heads=1",
            "arch.expansion=1.0", "global_batch_size=1", "checkpoint_every_eval=True"
        ])

    with console.status(f"[bold green]Running {model_name}..."):
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            console.print(f"[bold red]Error running pretrain.py for {model_name}:[/bold red]")
            console.print(e.stdout)
            console.print(e.stderr)
            raise

    with open(log_path, "r") as f:
        data = json.load(f)

    final_metrics = data[-1]

    if config_path.exists() and trial_num is not None:
        config_path.unlink()
    if Path(log_path).exists():
        Path(log_path).unlink()

    return final_metrics

def objective(trial: optuna.trial.Trial, args, console):
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

    metrics = run_model(args, console, "HREM", "hrem_v1_temp", hparams=params, trial_num=trial.number)
    return metrics.get('test/all/total_loss', float('inf'))

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
    args = parser.parse_args()

    console = Console()

    table = Table(title="Hyperparameter Optimization for HREM")
    table.add_column("Trial", justify="right", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")
    table.add_column("Params", style="green")
    table.add_column("State", style="yellow")

    with Live(table, console=console, screen=True, redirect_stderr=False) as live:
        study = optuna.create_study(direction="minimize")
        callback = RichCallback(table, live)
        study.optimize(lambda trial: objective(trial, args, console), n_trials=args.n_trials, callbacks=[callback])

        console.print("\n[bold green]Optimization finished.[/bold green]")
        console.print(f"[bold]Best trial: {study.best_trial.number}[/bold]")
        console.print(f"  [bold]Value[/bold]: {study.best_trial.value}")
        console.print("  [bold]Params[/bold]:")
        for key, value in study.best_trial.params.items():
            console.print(f"    [green]{key}[/green]: {value}")

    console.print("\n[bold blue]--- Final Comparison ---[/bold blue]")

    hrm_metrics = run_model(args, console, "HRM", "hrm_v1")
    best_hrem_metrics = run_model(args, console, "HREM_best", "hrem_v1_temp", hparams=study.best_trial.params)

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

if __name__ == "__main__":
    main()
