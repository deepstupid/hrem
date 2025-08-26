import subprocess
from functools import partial
from typing import Dict, Any, List
from pathlib import Path
import optuna
import yaml

from .config import ExperimentConfig, ModelConfig, DataConfig
from .reporting import aggregate_metrics, generate_optimization_report
from adaptive_demo_runner import AdaptiveDemoRunner
from demo_config import get_demo_config, PatienceLevel
from demo_runner import DemoRunner
from demo_metrics import MetricsCollector
from demo_utils import ResultsDisplayer, UIConfig

def run_optimization(config: ExperimentConfig) -> Dict[str, Any]:
    """
    Runs hyperparameter optimization using the AdaptiveDemoRunner.
    """
    if config.mode != "optimize":
        raise ValueError("ExperimentConfig must be in 'optimize' mode.")

    run_config = config.run_config
    opt_config = config.optimization_config
    logger = run_config.logger_callback or print

    logger(f"--- Starting Adaptive Optimization: {run_config.study_name} ---")

    # 1. Create a DemoConfig based on the ExperimentConfig
    patience = PatienceLevel.LOW if run_config.smoke_test else PatienceLevel.MEDIUM
    demo_config = get_demo_config(patience=patience)

    # Override settings from ExperimentConfig
    demo_config.settings["max_trials_per_model"] = opt_config.n_trials

    # The TUI passes a logger callback, which we can use to redirect output
    # to the TUI log widget. We need a simple way to display rich content from the runner.
    # For now, we will just use the console.
    # A proper implementation would use a queue to send rich renderables to the TUI.
    ui_config = UIConfig(main_display=demo_config.ui)
    results_displayer = ResultsDisplayer(ui_config)

    # 2. Instantiate the AdaptiveDemoRunner
    runner = AdaptiveDemoRunner(
        config=demo_config,
        results_displayer=results_displayer,
        metrics_collector=MetricsCollector(),
        ui_config=ui_config.main_display,
    )
    
    # 3. Prepare model configs for the runner
    # The adaptive runner can optimize multiple models at once.
    # The old optimization function was designed for one model at a time.
    models_to_optimize = [opt_config.model_to_optimize]

    # 4. Run the optimization
    optimization_results = runner.run_hyperparameter_optimization(
        study_name=run_config.study_name,
        data_config=config.data_config,
        model_configs=models_to_optimize,
        storage_path=opt_config.storage,
        n_jobs=opt_config.n_jobs,
        rich_callback=logger
    )

    # 5. Adapt results to the expected format
    # The adaptive runner returns a dictionary keyed by model name.
    # We need to extract the results for the single model we optimized.
    model_name = opt_config.model_to_optimize.name
    result = optimization_results.get(model_name, {})

    if not result:
        logger("[bold yellow]No optimization results found.[/bold yellow]")
        return {}

    final_results = {
        "best_trial": result.get("best_trial"),
        "best_params": result.get("best_params"),
        "best_value": result.get("best_value"),
        "report_path": run_config.output_dir / model_name / "scientific_report.md"
    }

    logger(f"Best trial for {model_name}: {final_results['best_trial']} with value: {final_results['best_value']:.4f}")
    logger(f"Scientific report generated at: {final_results['report_path']}")

    return final_results
