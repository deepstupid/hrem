import time
from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass
from rich.console import Console
import yaml

from .config import ChallengeConfig, AlgorithmConfig, PatienceBudget
from .patience_manager import AdaptivePatienceManager, ExplorationPhase, ScientificInsight
from .scheduler import DiscoveryAwareScheduler
from .insight_generator import ScientificInsightGenerator
from .report_generator import ScientificReportGenerator
from .timing_manager import ScientificTimingManager
from .optimization import HyperparameterOptimizer, OptunaOptimizer
from .trainer import Trainer
from .progress_handler import ProgressHandler
from .interactive_runner import InteractiveRunner
from .process import Evaluation, Optimization, InsightGeneration
import importlib
import torch
import numpy as np
from sc_engine.utils.plotting import generate_performance_plot
from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig
from dataset.common import PuzzleDatasetMetadata
from models.hrm.hrem import HREM
import threading


console = Console()

class ScientificDiscoveryEngine:
    """Core engine for running scientific exploration phases."""
    
    def __init__(self, challenge: ChallengeConfig, algorithms: List[AlgorithmConfig], default_training_config: Dict[str, Any], config: Dict[str, Any] = None, progress_handler: Optional[ProgressHandler] = None, cancel_event: Optional[threading.Event] = None):
        self.challenge = challenge
        self.algorithms = algorithms
        self.config = config or {}
        self.default_training_config = default_training_config
        self.patience_manager: Optional[AdaptivePatienceManager] = None
        self.scheduler = DiscoveryAwareScheduler(algorithms, challenge)
        self.insight_generator = ScientificInsightGenerator(challenge, algorithms, self.config.get("insights", "config/insight_config.yaml"))
        self.timing_manager = ScientificTimingManager()
        self.results: Dict[str, Any] = {}
        self.optimizer: HyperparameterOptimizer = OptunaOptimizer()
        self.progress_handler = progress_handler
        self.cancel_event = cancel_event or threading.Event()
        self.evaluation = Evaluation(self)
        self.optimization = Optimization(self)
        self.insight_generation = InsightGeneration(self)

    def run_baseline_evaluation(self) -> Tuple[Dict[str, Any], Dict[str, List]]:
        self.send_progress('start_phase', {'phase': 'baseline_evaluation'})
        is_demo_mode = self.config.get("is_demo", False)

        results, histories = self.evaluation.run_evaluation(
            title="⚡ Running Baseline Evaluation",
            phase=ExplorationPhase.BASELINE_EVALUATION,
            patience_allocation=0.5,
            run_suffix="baseline",
            model_config_fn=lambda alg: alg.config,
            result_key_fn=lambda alg: alg.name,
            timing_value=0,
            interleaved=is_demo_mode
        )
        self.send_progress('end_phase', {'phase': 'baseline_evaluation', 'results': results})
        return results, histories

    def run_hyperparameter_optimization(self, baseline_results: Dict[str, Any]) -> Dict[str, Any]:
        return self.optimization.run_hyperparameter_optimization(baseline_results)

    def run_final_evaluation(self, optimization_results: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, List]]:
        self.send_progress('start_phase', {'phase': 'final_evaluation'})
        is_demo_mode = self.config.get("is_demo", False)

        def model_config_fn(alg):
            config = alg.config.copy()
            config['arch_overrides'] = optimization_results.get(alg.name, {}).get('best_params', {})
            return config
        results, histories = self.evaluation.run_evaluation(
            title="🏆 Running Final Evaluation",
            phase=ExplorationPhase.FINAL_EVALUATION,
            patience_allocation=0.7,
            run_suffix="final",
            model_config_fn=model_config_fn,
            result_key_fn=lambda alg: f"{alg.name}_optimized",
            timing_value=1,
            interleaved=is_demo_mode
        )
        self.send_progress('end_phase', {'phase': 'final_evaluation', 'results': results})
        return results, histories
    
    def generate_insights(self, final_results: Dict[str, Any], plot_path: Optional[str], report_path: str, log_histories: Optional[Dict[str, List[Dict[str, Any]]]]) -> List[ScientificInsight]:
        return self.insight_generation.generate_insights(final_results, plot_path, report_path, log_histories)

    def send_progress(self, event_type: str, data: Dict = None):
        if self.progress_handler:
            self.progress_handler.on_progress(event_type, data if data is not None else {})