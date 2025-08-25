import subprocess
import sys
import json
import os
from pathlib import Path
import pandas as pd
from rich.console import Console

LOGS_DIR = Path("logs")

def run_command(command: list[str], console: Console, error_message: str, live_active: bool = False):
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
            run_command(build_command, console, error_msg)
        else:
            with console.status(build_msg):
                run_command(build_command, console, error_msg)

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
        run_command(command, console, error_msg)
    else:
        with console.status(run_msg):
            run_command(command, console, error_msg)

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
