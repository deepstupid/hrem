import subprocess
import sys
import json
import os
from pathlib import Path
import pandas as pd
import yaml

from .config import RunConfig, DataConfig, ModelConfig, TrainingConfig

def _run_command(command: list[str], logger_callback: callable, error_message: str):
    """
    Executes a command, streams its output to the logger, and handles potential errors.
    """
    try:
        process = subprocess.Popen(
            [sys.executable, *command],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )
        for line in process.stdout:
            logger_callback(line.strip())

        process.wait()

        if process.returncode != 0:
            error_details = f"Command exited with non-zero code {process.returncode}."
            logger_callback(f"[bold red]{error_message}[/bold red]")
            logger_callback(error_details)
            raise subprocess.CalledProcessError(process.returncode, command)

    except FileNotFoundError as e:
        logger_callback(f"[bold red]Error: {e}[/bold red]")
        raise
    except subprocess.CalledProcessError as e:
        # The error is already logged, just re-raise
        raise


def run_single_model(
    run_config: RunConfig,
    data_config: DataConfig,
    model_config: ModelConfig,
    training_config: TrainingConfig,
    run_identifier: str = "run_0"
):
    """
    Runs a single model training and evaluation.
    This is a refactored version of the original `run_model` function.
    """
    logger = run_config.logger_callback or print

    # 1. Determine dataset path and build dataset if it doesn't exist
    if run_config.smoke_test:
        data_dir = f"data/{data_config.dataset}-smoke"
        num_aug = 0
    else:
        data_dir = f"data/{data_config.dataset}-full"
        num_aug = data_config.num_aug

    data_config.path = data_dir

    dataset_builder_script = f"dataset/build_{data_config.dataset}_dataset.py"
    if not os.path.exists(data_dir):
        build_command = [
            dataset_builder_script,
            f"--output-dir={data_dir}",
            f"--num-aug={num_aug}"
        ]
        if data_config.dataset == "synthetic":
            build_command.extend([
                f"--task-type={data_config.synthetic_task}",
                "--num-samples=10" if run_config.smoke_test else "--num-samples=1000"
            ])

        logger(f"Building {data_config.dataset} dataset...")
        _run_command(build_command, logger, "Error running dataset builder:")
        logger(f"Dataset built successfully at {data_dir}")

    # 2. Construct the command for pretrain.py
    run_name = f"{model_config.name}_{data_config.dataset}_{run_identifier}"
    log_path = Path(run_config.output_dir) / f"{run_config.study_name}" / f"tmp_results_{run_name}.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Base command
    command = [
        "-m", "torch.distributed.run", "--nproc-per-node", "1",
        "--rdzv-backend", "c10d", "--rdzv-endpoint", "localhost:0",
        "pretrain.py",
        f"data_path={data_dir}",
        f"arch={model_config.base_arch_config}",
        f"+log_path={str(log_path)}",
        f"+project_name={run_config.project_name}",
        f"+run_name={run_name}",
    ]

    # Adjust config for smoke test
    if run_config.smoke_test:
        training_config.epochs = 1
        training_config.eval_interval = 1
        training_config.global_batch_size = 1

    # Add training params
    command.extend([
        f"epochs={training_config.epochs}",
        f"eval_interval={training_config.eval_interval}",
        f"global_batch_size={training_config.global_batch_size}",
    ])

    # Add model params
    hparam_args = []
    if model_config.type == "HREM" and model_config.hrem_params:
        # Load the base config to check for existing keys
        base_model_config_path = f"config/arch/{model_config.base_arch_config}.yaml"
        with open(base_model_config_path, 'r') as f:
            base_model_config = yaml.safe_load(f)

        for key, value in model_config.hrem_params.model_dump().items():
            if key not in base_model_config:
                hparam_args.append(f"+arch.{key}={value}")
            else:
                hparam_args.append(f"arch.{key}={value}")

    command.extend(hparam_args)

    # Add smoke test params if applicable
    if run_config.smoke_test:
        command.extend([
            "+smoke_test=True", "arch.puzzle_emb_ndim=16", "arch.num_heads=1",
            "arch.expansion=1.0", "checkpoint_every_eval=True"
        ])

    # 3. Run the command
    logger(f"Running model: {model_config.name} ({run_identifier})...")
    logger(f"Command: {' '.join(command)}")
    _run_command(command, logger, f"Error running pretrain.py for {model_config.name}")

    # 4. Parse and return results
    if not log_path.exists():
        logger(f"[bold yellow]Warning: Log file {log_path} not found.[/bold yellow]")
        return {}

    with open(log_path, "r") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            logger(f"[bold yellow]Warning: Log file {log_path} is empty or corrupted.[/bold yellow]")
            return {}

    if not data:
        logger(f"[bold yellow]Warning: Log file {log_path} is empty.[/bold yellow]")
        return {}

    best_entry = min(data, key=lambda x: x.get('all', {}).get('lm_loss', float('inf')), default=None)
    if best_entry is None:
        best_entry = data[-1] if data else {}

    final_metrics = pd.json_normalize(best_entry, sep='/').to_dict(orient='records')[0] if best_entry else {}

    # Clean up temporary log file
    # log_path.unlink()

    logger(f"Finished running model: {model_config.name}. Final loss: {final_metrics.get('all/lm_loss', 'N/A')}")
    return final_metrics
