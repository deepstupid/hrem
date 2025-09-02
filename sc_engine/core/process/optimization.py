from typing import Dict, Any

from ..trainer import Trainer


class Optimization:
    """Handles hyperparameter optimization."""

    def __init__(self, engine):
        self.engine = engine

    def run_hyperparameter_optimization(self, baseline_results: Dict[str, Any]) -> Dict[str, Any]:
        self.engine.send_progress('start_phase', {'phase': 'optimization'})
        optimization_results = {}
        for alg in self.engine.algorithms:
            if self.engine.cancel_event.is_set():
                self.engine.send_progress('optimization_cancelled', {'reason': 'Cancelled by user.'})
                break
            if not alg.search_space:
                self.engine.send_progress('skip_optimization', {'algorithm': alg.name, 'reason': 'No search space defined.'})
                optimization_results[alg.name] = baseline_results.get(alg.name, {})
                continue
            self.engine.send_progress('start_optimization_alg', {'algorithm': alg.name})
            def objective(hparams: Dict[str, Any]) -> float:
                self.engine.send_progress('start_trial', {'params': hparams})
                model_config = alg.config.copy()
                model_config['arch_overrides'] = hparams
                run_config = {"study_name": f"{self.engine.challenge.id}_optimize_{alg.name}", "output_dir": "experiments"}
                training_config = self.engine.default_training_config
                trainer = Trainer(training_config, model_config, self.engine.challenge.dataset, run_config, progress_handler=self.engine.progress_handler)
                metrics, _ = trainer.train_and_evaluate()
                loss = metrics.get('all/lm_loss', float('inf'))
                self.engine.send_progress('end_trial', {'params': hparams, 'loss': loss})
                return loss
            best_params = self.engine.optimizer.optimize(objective=objective, search_space=alg.search_space, n_trials=self.engine.config.get("n_trials", 10))
            self.engine.send_progress('end_optimization_alg', {'algorithm': alg.name, 'best_params': best_params})
            optimization_results[alg.name] = {'best_params': best_params}
        self.engine.send_progress('end_phase', {'phase': 'optimization', 'results': optimization_results})
        return optimization_results
