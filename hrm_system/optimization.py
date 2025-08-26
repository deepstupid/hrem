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

    # 1. Suggest hyperparameters from the provided search space
    params = {}
    search_space = opt_config.search_space
    if not search_space:
        logger("[bold red]No search space provided for optimization.[/bold red]")
        raise optuna.TrialPruned()

    # The top-level key in the search space (e.g., 'hrem_params', 'arch_overrides')
    # determines where the parameters will be applied.
    param_section_key = next(iter(search_space), None)
    if not param_section_key:
        logger("[bold red]Search space is empty.[/bold red]")
        raise optuna.TrialPruned()
    
    param_definitions = search_space[param_section_key]

    for name, definition in param_definitions.items():
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
    
    # Dynamically set the attribute on the model config
    if hasattr(trial_model_config, param_section_key):
        # For structured pydantic models like hrem_params
        param_model = getattr(trial_model_config, param_section_key)
        if param_model and isinstance(param_model, HREMParams):
             # Create a new HREMParams object with the suggested values
            setattr(trial_model_config, param_section_key, HREMParams(**params))
        else:
             # Fallback for other potential structured models
            setattr(trial_model_config, param_section_key, params)
    else:
        # For simple dictionaries like arch_overrides
        setattr(trial_model_config, param_section_key, params)

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

    # Run best model
    best_model_config = opt_config.model_to_optimize.model_copy(deep=True)
    
    # Dynamically determine where to set the best parameters
    param_section_key = next(iter(opt_config.search_space), None)
    if param_section_key:
        if hasattr(best_model_config, param_section_key):
            param_model = getattr(best_model_config, param_section_key)
            if param_model and isinstance(param_model, HREMParams):
                setattr(best_model_config, param_section_key, HREMParams(**best_trial.params))
            else:
                setattr(best_model_config, param_section_key, best_trial.params)
        else:
            setattr(best_model_config, param_section_key, best_trial.params)

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
