import subprocess
from functools import partial
from typing import Dict, Any, List
from pathlib import Path
import optuna
import yaml

from .config import ExperimentConfig, HREMParams
from .runner import run_single_model
from .reporting import aggregate_metrics, generate_optimization_report

def _objective(
    trial: optuna.trial.Trial,
    config: ExperimentConfig,
) -> float:
    """
    The objective function for Optuna to minimize.
    """
    run_config = config.run_config
    opt_config = config.optimization_config
    logger = run_config.logger_callback or print

    # 1. Suggest hyperparameters
    params = {}
    search_space_path = opt_config.search_space.get("path", "config/hparam_search_space.yaml")
    with open(search_space_path, 'r') as f:
        search_space = yaml.safe_load(f)

    # Determine which parameter section to use based on the model being optimized
    param_section = "hrem_params"
    if "hrm" in opt_config.model_to_optimize.name.lower() or "hrm" in opt_config.model_to_optimize.algorithm_class.lower():
        param_section = "hrm_params"
    
    # Fallback to hrem_params if the specific section doesn't exist
    if param_section not in search_space:
        param_section = "hrem_params"

    for name, definition in search_space[param_section].items():
        param_type = definition['type']
        if run_config.smoke_test and f"smoke_{param_type}" in definition:
            param_type = f"smoke_{definition['type']}"

        if param_type == "int":
            low = definition.get("smoke_low", definition["low"]) if run_config.smoke_test else definition["low"]
            high = definition.get("smoke_high", definition["high"]) if run_config.smoke_test else definition["high"]
            params[name] = trial.suggest_int(name, low, high)
        elif param_type == "categorical":
            choices = definition.get("smoke_choices", definition["choices"]) if run_config.smoke_test else definition["choices"]
            params[name] = trial.suggest_categorical(name, choices)

    # 2. Create model config for this trial
    trial_model_config = opt_config.model_to_optimize.model_copy(deep=True)
    
    # Set parameters based on model type
    if "hrem" in opt_config.model_to_optimize.name.lower() or "hrem" in opt_config.model_to_optimize.algorithm_class.lower():
        trial_model_config.hrem_params = HREMParams(**params)
    else:
        # For HRM, we'll pass parameters as arch_overrides
        trial_model_config.arch_overrides = params

    # 3. Run the model
    try:
        metrics = run_single_model(
            run_config=run_config.model_copy(deep=True),
            data_config=config.data_config.model_copy(deep=True),
            model_config=trial_model_config,
            training_config=config.training_config.model_copy(deep=True),
            run_identifier=f"trial_{trial.number}"
        )
        loss = metrics.get('all/lm_loss', float('inf'))
        return float(loss) if loss is not None else float('inf')
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logger(f"[bold red]Trial {trial.number} failed and will be pruned.[/bold red]")
        raise optuna.TrialPruned() from e


def run_optimization(config: ExperimentConfig) -> Dict[str, Any]:
    """
    Runs hyperparameter optimization using Optuna.
    """
    if config.mode != "optimize":
        raise ValueError("ExperimentConfig must be in 'optimize' mode.")

    run_config = config.run_config
    opt_config = config.optimization_config
    logger = run_config.logger_callback or print

    logger(f"--- Starting Optimization: {run_config.study_name} ---")
    logger(f"Optuna storage: {opt_config.storage}")
    logger(f"To monitor, run: optuna-dashboard {opt_config.storage}")

    # 1. Create and run the study
    study = optuna.create_study(
        study_name=run_config.study_name,
        storage=opt_config.storage,
        direction="minimize",
        load_if_exists=True
    )

    # Use partial to pass the config to the objective function
    objective_with_config = partial(_objective, config=config)

    study.optimize(
        objective_with_config,
        n_trials=opt_config.n_trials,
        n_jobs=opt_config.n_jobs
    )

    logger("\n[bold green]Optimization finished.[/bold green]")

    # 2. Get best trial and save results
    if not study.trials:
        logger("[bold yellow]No trials were completed. Skipping final comparison.[/bold yellow]")
        return {}

    try:
        best_trial = study.best_trial
        logger(f"Best trial: {best_trial.number} with value: {best_trial.value:.4f}")
    except ValueError:
        logger("[bold yellow]No best trial found. Skipping final comparison.[/bold yellow]")
        return {}

    results_df = study.trials_dataframe()
    results_path = Path(f"{run_config.output_dir}/{run_config.study_name}/optimization_results.csv")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(results_path, index=False)
    logger(f"Saved optimization results to {results_path}")

    # 3. Run final comparison
    logger("\n[bold blue]--- Running Final Comparison ---[/bold blue]")
    final_metrics = {}

    # Run baseline model
    baseline_metrics_list = [
        run_single_model(
            run_config, config.data_config, opt_config.baseline_model, config.training_config, f"baseline_run_{i}"
        ) for i in range(opt_config.n_final_runs)
    ]
    final_metrics[opt_config.baseline_model.name] = aggregate_metrics(baseline_metrics_list)

    # Run best model (could be HRM or HREM)
    best_model_config = opt_config.model_to_optimize.model_copy(deep=True)
    
    # Set parameters based on model type
    if "hrem" in opt_config.model_to_optimize.name.lower() or "hrem" in opt_config.model_to_optimize.algorithm_class.lower():
        best_model_config.hrem_params = HREMParams(**best_trial.params)
    else:
        # For HRM, we'll pass parameters as arch_overrides
        best_model_config.arch_overrides = best_trial.params

    best_hrem_metrics_list = [
        run_single_model(
            run_config, config.data_config, best_model_config, config.training_config, f"best_model_run_{i}"
        ) for i in range(opt_config.n_final_runs)
    ]
    final_metrics[best_model_config.name] = aggregate_metrics(best_hrem_metrics_list)

    # 4. Generate report
    logger("\n[bold blue]--- Generating Optimization Report ---[/bold blue]")
    report_path = generate_optimization_report(
        final_metrics=final_metrics,
        best_trial=best_trial,
        opt_config=opt_config,
        run_config=run_config
    )
    logger(f"Generated optimization report: {report_path}")

    return {
        "best_trial": best_trial.number,
        "best_params": best_trial.params,
        "best_value": best_trial.value,
        "results": final_metrics,
        "report_path": report_path
    }
