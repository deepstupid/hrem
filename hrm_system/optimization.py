"""Core hyperparameter optimization logic for the HRM system."""

from typing import Dict, Any
import optuna

from .config import ExperimentConfig, HREMParams
from .runner import run_single_model
from . import reporting

def run_optimization(config: ExperimentConfig) -> Dict[str, Any]:
    """
    Runs hyperparameter optimization for a single model.
    """
    if config.mode != "optimize":
        raise ValueError("ExperimentConfig must be in 'optimize' mode.")

    opt_config = config.optimization_config
    if not opt_config or not opt_config.model_to_optimize or not opt_config.search_space:
        raise ValueError("Optimization mode requires a model to optimize and a search space.")

    def objective(trial: optuna.trial.Trial) -> tuple[float, int]:
        """The multi-objective function for Optuna to optimize.

        Objectives:
        1. Maximize accuracy
        2. Minimize parameter count
        """
        model_config = opt_config.model_to_optimize
        search_space = opt_config.search_space or {}

        params = {}
        # This assumes a flat search space, which is what we have now.
        # If the search space becomes nested, this will need to be updated.
        param_definitions = search_space.get(f"{model_config.name.lower()}_params", {})

        for param_name, definition in param_definitions.items():
            param_type = definition.get("type")
            if config.run_config.smoke_test and "smoke_choices" in definition:
                params[param_name] = trial.suggest_categorical(param_name, definition["smoke_choices"])
            elif config.run_config.smoke_test and "smoke_low" in definition:
                params[param_name] = trial.suggest_int(param_name, definition["smoke_low"], definition["smoke_high"])
            elif param_type == "categorical":
                params[param_name] = trial.suggest_categorical(param_name, definition["choices"])
            elif param_type == "int":
                params[param_name] = trial.suggest_int(param_name, definition["low"], definition["high"])

        trial_model_config = model_config.model_copy(deep=True)
        if "hrem" in trial_model_config.algorithm_class.lower():
            trial_model_config.hrem_params = HREMParams(**params)
        else:
            trial_model_config.arch_overrides = params

        try:
            metrics = run_single_model(
                run_config=config.run_config,
                data_config=config.data_config,
                model_config=trial_model_config,
                training_config=config.training_config,
                run_identifier=f"trial_{trial.number}"
            )
            # Return accuracy (to be maximized) and parameter count (to be minimized)
            accuracy = metrics.get('all/accuracy', 0.0)
            num_params = metrics.get('num_parameters', float('inf'))
            return accuracy, num_params
        except Exception as e:
            # Log the error and let Optuna handle it as a pruned trial.
            print(f"Trial {trial.number} failed with error: {e}")
            raise optuna.TrialPruned()

    study = optuna.create_study(
        study_name=config.run_config.study_name,
        storage=opt_config.storage,
        directions=["maximize", "minimize"],
        load_if_exists=True
    )

    study.optimize(
        objective,
        n_trials=opt_config.n_trials,
        n_jobs=opt_config.n_jobs,
        callbacks=[config.run_config.logger_callback] if config.run_config.logger_callback else None,
    )

    # After optimization, prepare and return the results from the best trial(s)
    best_trials = study.best_trials
    if not best_trials:
        return {"best_trials": []}

    # For now, we select the first best trial from the Pareto front
    best_trial = best_trials[0]

    results = {
        "best_trial": best_trial.number,
        "best_params": best_trial.params,
        "best_values": best_trial.values,
    }

    if config.run_config.output_dir:
        # This part is a bit tricky. The new generate_optimization_report needs final_metrics, which we don't have here.
        # For now, let's just skip the report generation in this function.
        # The reporting is handled in the `run.py` demo command.
        pass

    return results
