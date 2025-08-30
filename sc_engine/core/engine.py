"""Scientific Discovery Engine for algorithm comparison."""

import time
import os
import torch
import torch.distributed as dist
import tqdm
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from rich.console import Console
import optuna

from .config import ChallengeConfig, AlgorithmConfig, PatienceBudget
from .patience_manager import AdaptivePatienceManager, ExplorationPhase, ScientificInsight
from .scheduler import DiscoveryAwareScheduler
from .insight_generator import ScientificInsightGenerator
from .report_generator import ScientificReportGenerator
from .timing_manager import ScientificTimingManager
from sc_engine.plugins.algorithms.utils import (
    LocalLogger,
    create_dataloader,
    train_batch,
    evaluate,
)
from puzzle_dataset import PuzzleDatasetMetadata
import importlib

console = Console()

@dataclass
class DiscoveryResults:
    """Results from a scientific discovery session."""
    challenge: ChallengeConfig
    algorithm_results: Dict[str, Any]
    insights: List[ScientificInsight]
    timing_data: Dict[str, Any]
    metadata: Dict[str, Any]

class ScientificDiscoveryEngine:
    """Central orchestrator for scientific exploration process."""
    
    def __init__(self, challenge: ChallengeConfig, algorithms: List[AlgorithmConfig], config: Dict[str, Any] = None):
        """Initialize with a scientific challenge and algorithms to compare."""
        self.challenge = challenge
        self.algorithms = algorithms
        self.config = config or {}
        self.patience_manager: Optional[AdaptivePatienceManager] = None
        self.scheduler = DiscoveryAwareScheduler(algorithms, challenge)
        self.insight_generator = ScientificInsightGenerator(challenge, algorithms, self.config.get("insights", "config/insight_config.yaml"))
        self.timing_manager = ScientificTimingManager()
        self.results: Dict[str, Any] = {}
        
    def execute_discovery_session(self, patience_budget: PatienceBudget) -> DiscoveryResults:
        """Execute a scientifically-driven comparison within patience constraints."""
        console.print(f"[bold blue]🔬 Starting Scientific Discovery Session[/bold blue]")
        console.print(f"[cyan]Challenge: {self.challenge.name}[/cyan]")
        
        self.patience_manager = AdaptivePatienceManager(patience_budget, self.timing_manager)
        
        baseline_results = self._run_baseline_evaluation()
        optimization_results = self._run_hyperparameter_optimization(baseline_results)
        final_results = self._run_final_evaluation(optimization_results)
        insights = self._generate_insights(final_results)
        
        discovery_results = DiscoveryResults(
            challenge=self.challenge,
            algorithm_results=final_results,
            insights=insights,
            timing_data=self.timing_manager.export_metrics(),
            metadata={
                'session_start_time': self.patience_manager.start_time,
                'total_elapsed_time': self.patience_manager.get_elapsed_time(),
            }
        )
        
        console.print("[green]✅ Scientific discovery session completed![/green]")
        return discovery_results
    
    def _get_default_training_config(self) -> dict:
        """Returns a default training configuration dictionary."""
        return {
            "epochs": 20000, "eval_interval": 2000, "global_batch_size": 384,
            "lr": 1e-4, "lr_min_ratio": 0.1, "lr_warmup_steps": 2000,
            "puzzle_emb_lr": 1e-4, "weight_decay": 1.0, "puzzle_emb_weight_decay": 1.0,
            "beta1": 0.9, "beta2": 0.95, "optimizer": "AdamW", "optimizer_eps": 1e-8,
            "seed": 0, "checkpoint_every_eval": False, "eval_save_outputs": [],
            "smoke_test": False, "num_workers": 1, "prefetch_factor": 8,
            "lr_schedule": "cosine", "use_amp": False,
        }

    def _train_and_evaluate_model(self, model_config: dict, data_config: dict, run_config: dict, training_config: dict) -> Dict[str, Any]:
        """The main training and evaluation loop."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        RANK = 0
        WORLD_SIZE = 1

        if "LOCAL_RANK" in os.environ:
            dist.init_process_group(backend="nccl" if device == "cuda" else "gloo")
            RANK = dist.get_rank()
            WORLD_SIZE = dist.get_world_size()
            if device == "cuda":
                torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

        torch.random.manual_seed(training_config['seed'] + RANK)

        # Dataset preparation
        dataset_name = data_config['dataset']
        task_name = data_config.get('synthetic_task', 'default')
        if dataset_name.startswith("synthetic-"):
            parts = dataset_name.split('-', 1)
            dataset_name = parts[0]
            task_name = parts[1]

        smoke_test = run_config.get('smoke_test', False)
        if smoke_test:
            data_dir = f"data/{dataset_name}-{task_name}-smoke"
        else:
            data_dir = f"data/{dataset_name}-{task_name}-full"

        train_epochs_per_iter = training_config.get('eval_interval', training_config['epochs'])
        total_iters = training_config['epochs'] // train_epochs_per_iter

        train_loader, train_metadata = create_dataloader(training_config, data_dir, "train", test_set_mode=False, epochs_per_iter=train_epochs_per_iter, global_batch_size=training_config['global_batch_size'], rank=RANK, world_size=WORLD_SIZE)
        eval_loader, eval_metadata = create_dataloader(training_config, data_dir, "test", test_set_mode=True, epochs_per_iter=1, global_batch_size=training_config['global_batch_size'], rank=RANK, world_size=WORLD_SIZE)

        # Instantiate algorithm
        module_path, class_name = model_config['algorithm_class'].rsplit('.', 1)
        module = importlib.import_module(module_path)
        algorithm_class = getattr(module, class_name)
        algorithm = algorithm_class(model_config, training_config)

        # Initialize training state
        algorithm.initialize_train_state(train_metadata, world_size=WORLD_SIZE, rank=RANK)
        train_state = algorithm.train_state

        if training_config['smoke_test']:
            train_state.total_steps = 1

        progress_bar = None
        logger = None
        if RANK == 0:
            progress_bar = tqdm.tqdm(total=train_state.total_steps)
            log_path = os.path.join(run_config['output_dir'], f"tmp_results_{run_config['study_name']}.json")
            logger = LocalLogger(log_path=log_path)
            logger.log({"num_params": sum(x.numel() for x in train_state.model.parameters())}, step=0)

        final_metrics = {}
        for _iter_id in range(total_iters):
            train_state.model.train()
            iter_start_time = time.time()

            for set_name, batch, global_batch_size in train_loader:
                metrics = train_batch(training_config, train_state, batch, global_batch_size, rank=RANK, world_size=WORLD_SIZE)
                if RANK == 0 and metrics is not None:
                    if logger:
                        logger.log(metrics, train_state.step)
                    progress_bar.update(train_state.step - progress_bar.n)

            iter_end_time = time.time()
            iter_duration = iter_end_time - iter_start_time
            avg_epoch_time = iter_duration / train_epochs_per_iter if train_epochs_per_iter > 0 else 0

            train_state.model.eval()
            checkpoint_path = run_config['output_dir']
            metrics = evaluate(training_config, checkpoint_path, train_state, eval_loader, eval_metadata, rank=RANK, world_size=WORLD_SIZE)
            if RANK == 0 and metrics is not None:
                if logger:
                    logger.log(metrics, train_state.step)
                final_metrics = metrics
                final_metrics['avg_epoch_time'] = avg_epoch_time

        if logger:
            logger.finish()
        if dist.is_initialized():
            dist.destroy_process_group()

        return final_metrics

    def _run_baseline_evaluation(self) -> Dict[str, Any]:
        """Run baseline evaluation for all algorithms."""
        console.print("[bold blue]⚡ Running Baseline Evaluation...[/bold blue]")
        allocation = self.patience_manager.allocate_for_phase(ExplorationPhase.BASELINE_EVALUATION, 0.5)
        start_time = time.time()
        baseline_results = {}
        
        for alg in self.algorithms:
            run_config = {"study_name": f"{self.challenge.id}_baseline_{alg.name}", "output_dir": "experiments"}
            training_config = self._get_default_training_config()
            model_config = alg.config

            metrics = self._train_and_evaluate_model(model_config, self.challenge.dataset, run_config, training_config)
            baseline_results[alg.name] = metrics
        
        elapsed_time = time.time() - start_time
        self.patience_manager.update_patience_consumption(elapsed_time, ExplorationPhase.BASELINE_EVALUATION)
        self.timing_manager.record_discovery_timing("baseline_evaluation", elapsed_time, 0)
        return baseline_results

    def _run_hyperparameter_optimization(self, baseline_results: Dict[str, Any]) -> Dict[str, Any]:
        """Run hyperparameter optimization with adaptive depth."""
        console.print("[bold blue]🔍 Running Hyperparameter Optimization...[/bold blue]")
        
        optimization_results = {}

        for alg in self.algorithms:
            if not alg.search_space:
                console.print(f"[yellow]No search space defined for {alg.name}, skipping optimization.[/yellow]")
                optimization_results[alg.name] = baseline_results.get(alg.name, {})
                continue

            def objective(trial: optuna.Trial) -> float:
                hparams = {}
                for param_name, param_config in alg.search_space.items():
                    if param_config['type'] == 'int':
                        hparams[param_name] = trial.suggest_int(param_name, param_config['low'], param_config['high'])
                    elif param_config['type'] == 'float':
                        hparams[param_name] = trial.suggest_float(param_name, param_config['low'], param_config['high'], log=param_config.get('log', False))
                    elif param_config['type'] == 'categorical':
                        hparams[param_name] = trial.suggest_categorical(param_name, param_config['choices'])

                model_config = alg.config.copy()
                model_config['arch_overrides'] = hparams

                run_config = {"study_name": f"{self.challenge.id}_optimize_{alg.name}_{trial.number}", "output_dir": "experiments"}
                training_config = self._get_default_training_config()

                metrics = self._train_and_evaluate_model(model_config, self.challenge.dataset, run_config, training_config)

                # We want to minimize the validation loss
                return metrics.get('all/lm_loss', float('inf'))

            study = optuna.create_study(direction="minimize")
            study.optimize(objective, n_trials=self.config.get("n_trials", 10))

            best_params = study.best_params
            console.print(f"[green]Best parameters for {alg.name}: {best_params}[/green]")
            optimization_results[alg.name] = {'best_params': best_params}

        return optimization_results

    def _run_final_evaluation(self, optimization_results: Dict[str, Any]) -> Dict[str, Any]:
        """Run final evaluation with optimized parameters."""
        console.print("[bold blue]🏆 Running Final Evaluation...[/bold blue]")
        allocation = self.patience_manager.allocate_for_phase(ExplorationPhase.FINAL_EVALUATION, 0.7)
        start_time = time.time()
        final_results = {}
        
        for alg in self.algorithms:
            run_config = {"study_name": f"{self.challenge.id}_final_{alg.name}", "output_dir": "experiments"}
            training_config = self._get_default_training_config()

            model_config = alg.config.copy()
            model_config['arch_overrides'] = optimization_results.get(alg.name, {}).get('best_params', {})

            metrics = self._train_and_evaluate_model(model_config, self.challenge.dataset, run_config, training_config)
            final_results[f"{alg.name}_optimized"] = metrics

        elapsed_time = time.time() - start_time
        self.patience_manager.update_patience_consumption(elapsed_time, ExplorationPhase.FINAL_EVALUATION)
        self.timing_manager.record_discovery_timing("final_evaluation", elapsed_time, 1)
        return final_results
    
    def _generate_insights(self, final_results: Dict[str, Any]) -> List[ScientificInsight]:
        """Generate scientific insights from comparison results."""
        console.print("[bold blue]💡 Generating Scientific Insights...[/bold blue]")
        
        allocation = self.patience_manager.allocate_for_phase(ExplorationPhase.INSIGHT_GENERATION, discovery_potential=0.9)
        start_time = time.time()
        
        insights = self.insight_generator.extract_insights(final_results)
        
        for insight in insights:
            insight.discovery_potential = self.insight_generator.classify_discovery_potential(insight.discovery_potential)
        
        elapsed_time = time.time() - start_time
        self.patience_manager.update_patience_consumption(elapsed_time, ExplorationPhase.INSIGHT_GENERATION)
        self.timing_manager.record_discovery_timing("insight_generation", elapsed_time, len(insights))
        
        if insights:
            console.print(f"[green]✅ Generated {len(insights)} scientific insights.[/green]")
            report_generator = ScientificReportGenerator(challenge_name=self.challenge.name, algorithm_names=[alg.name for alg in self.algorithms])
            report_content = report_generator.generate_report(insights)

            report_path = "scientific_report.md"
            with open(report_path, "w") as f:
                f.write(report_content)
            console.print(f"[bold green]📄 Scientific report saved to {report_path}[/bold green]")
        else:
            console.print("[yellow]⚠️ No significant insights generated[/yellow]")
        
        return insights