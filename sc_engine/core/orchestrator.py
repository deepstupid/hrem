from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from rich.console import Console

from .config import ChallengeConfig, PatienceBudget
from .engine import ScientificDiscoveryEngine
from .patience_manager import AdaptivePatienceManager, ExplorationPhase
from .report_generator import ScientificReportGenerator
from sc_engine.utils.plotting import generate_performance_plot


console = Console()

@dataclass
class DiscoveryResults:
    """Results from a scientific discovery session."""
    challenge: ChallengeConfig
    algorithm_results: Dict[str, Any]
    insights: List[Any]
    timing_data: Dict[str, Any]
    metadata: Dict[str, Any]
    log_histories: Dict[str, List[Dict[str, Any]]] = None


class EngineOrchestrator:
    """Orchestrates the scientific discovery process."""

    def __init__(self, engine: ScientificDiscoveryEngine):
        self.engine = engine
        self.patience_manager: Optional[AdaptivePatienceManager] = None

    def run(self, patience_budget: PatienceBudget) -> DiscoveryResults:
        """Execute a scientifically-driven comparison within patience constraints."""
        self.engine._send_progress('start_session', {'challenge': self.engine.challenge.name})

        self.patience_manager = AdaptivePatienceManager(patience_budget, self.engine.timing_manager)
        self.engine.patience_manager = self.patience_manager

        baseline_results, baseline_histories = self.engine._run_baseline_evaluation()
        optimization_results = self.engine._run_hyperparameter_optimization(baseline_results)
        final_results, final_histories = self.engine._run_final_evaluation(optimization_results)

        log_histories = {**baseline_histories, **final_histories}
        plot_path = generate_performance_plot(log_histories) if log_histories else None

        report_path = "scientific_report.md"
        insights = self.engine._generate_insights(final_results, plot_path, report_path, log_histories)

        discovery_results = DiscoveryResults(
            challenge=self.engine.challenge,
            algorithm_results=final_results,
            insights=insights,
            timing_data=self.engine.timing_manager.export_metrics(),
            metadata={
                'session_start_time': self.patience_manager.start_time,
                'total_elapsed_time': self.patience_manager.get_elapsed_time(),
                'plot_path': plot_path
            },
            log_histories=log_histories
        )

        console.print("[green]✅ Scientific discovery session completed![/green]")
        return discovery_results
