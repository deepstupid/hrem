import importlib
import json
import os
import sys
from pathlib import Path

import pandas as pd
import yaml
import subprocess

from .config import RunConfig, DataConfig, ModelConfig, TrainingConfig

def _run_dataset_builder(command: list[str], logger_callback: callable):
    """
    Executes the dataset builder script.
    This is kept separate as it's a one-time setup process per dataset.
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
            raise subprocess.CalledProcessError(process.returncode, command)
    except FileNotFoundError as e:
        logger_callback(f"[bold red]Error: {e}[/bold red]")
        raise
    except subprocess.CalledProcessError as e:
        logger_callback(f"[bold red]Dataset builder failed with exit code {e.returncode}[/bold red]")
        raise


def run_single_model(
    run_config: RunConfig,
    data_config: DataConfig,
    model_config: ModelConfig,
    training_config: TrainingConfig,
    run_identifier: str = "run_0"
):
    """
    Runs a single model training and evaluation using the new pluggable algorithm architecture.
    """
    logger = run_config.logger_callback or print

    # 1. Determine dataset path and build dataset if it doesn't exist
    if run_config.smoke_test:
        data_dir = f"data/{data_config.dataset}-smoke"
        num_aug = 0
        training_config.smoke_test = True
        training_config.epochs = 1
        training_config.eval_interval = 1
        # Note: Smoke test overrides for model params are now handled inside the algorithm.
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
        _run_dataset_builder(build_command, logger)
        logger(f"Dataset built successfully at {data_dir}")

    # 2. Instantiate and run the algorithm
    logger(f"Running model: {model_config.name} ({run_identifier})...")

    # Dynamically import the algorithm class
    try:
        module_path, class_name = model_config.algorithm_class.rsplit('.', 1)
        module = importlib.import_module(module_path)
        algorithm_class = getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        logger(f"[bold red]Error: Could not import algorithm class '{model_config.algorithm_class}'. {e}[/bold red]")
        raise

    # Instantiate the algorithm
    algorithm = algorithm_class(model_config, training_config)

    # Prepare paths
    run_name = f"{model_config.name}_{data_config.dataset}_{run_identifier}"
    output_dir = Path(run_config.output_dir) / f"{run_config.study_name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run training
    algorithm.train(
        data_path=data_dir,
        logger_callback=logger,
        checkpoint_path=str(output_dir),
        run_name=run_name
    )

    # 3. Parse and return results from the log file
    log_path = output_dir / f"tmp_results_{run_name}.json"
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

    # Find the best result based on the lowest validation loss
    best_entry = min(data, key=lambda x: x.get('all', {}).get('lm_loss', float('inf')), default=None)
    if best_entry is None:
        best_entry = data[-1] if data else {}

    # Normalize the nested dictionary into a flat dictionary
    final_metrics = pd.json_normalize(best_entry, sep='/').to_dict(orient='records')[0] if best_entry else {}

    logger(f"Finished running model: {model_config.name}. Final loss: {final_metrics.get('all/lm_loss', 'N/A')}")
    return final_metrics
