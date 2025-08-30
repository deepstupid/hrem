from abc import ABC, abstractmethod
from typing import Dict, Any, Callable, List
import optuna

class HyperparameterOptimizer(ABC):
    """
    Abstract base class for hyperparameter optimizers.
    """
    @abstractmethod
    def optimize(self, objective: Callable, search_space: Dict[str, Any], n_trials: int) -> Dict[str, Any]:
        """
        Run the optimization process.

        Args:
            objective: The objective function to minimize.
            search_space: The search space for the hyperparameters.
            n_trials: The number of trials to run.

        Returns:
            A dictionary containing the best parameters found.
        """
        pass

class OptunaOptimizer(HyperparameterOptimizer):
    """
    A hyperparameter optimizer that uses Optuna.
    """
    def optimize(self, objective: Callable, search_space: Dict[str, Any], n_trials: int) -> Dict[str, Any]:
        """
        Run the optimization process using Optuna.
        """
        def optuna_objective(trial: optuna.Trial) -> float:
            hparams = {}
            for param_name, param_config in search_space.items():
                if param_config['type'] == 'int':
                    hparams[param_name] = trial.suggest_int(param_name, param_config['low'], param_config['high'])
                elif param_config['type'] == 'float':
                    hparams[param_name] = trial.suggest_float(param_name, param_config['low'], param_config['high'], log=param_config.get('log', False))
                elif param_config['type'] == 'categorical':
                    hparams[param_name] = trial.suggest_categorical(param_name, param_config['choices'])
            return objective(hparams)

        study = optuna.create_study(direction="minimize")
        study.optimize(optuna_objective, n_trials=n_trials)

        return study.best_params
