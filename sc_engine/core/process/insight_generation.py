import time
from typing import List, Dict, Any, Optional

from ..patience_manager import ExplorationPhase
from ..report_generator import ScientificReportGenerator
from ..insights import ScientificInsight


class InsightGeneration:
    """Handles the generation of scientific insights."""

    def __init__(self, engine):
        self.engine = engine

    def generate_insights(self, final_results: Dict[str, Any], plot_path: Optional[str], report_path: str, log_histories: Optional[Dict[str, List[Dict[str, Any]]]]) -> List[ScientificInsight]:
        self.engine.send_progress('start_phase', {'phase': 'insight_generation'})
        allocation = self.engine.patience_manager.allocate_for_phase(ExplorationPhase.INSIGHT_GENERATION, discovery_potential=0.9)
        start_time = time.time()
        insights = self.engine.insight_generator.extract_insights(final_results)
        for insight in insights:
            insight.discovery_potential = self.engine.insight_generator.classify_discovery_potential(insight.discovery_potential)
        elapsed_time = time.time() - start_time
        self.engine.patience_manager.update_patience_consumption(elapsed_time, ExplorationPhase.INSIGHT_GENERATION)
        self.engine.timing_manager.record_discovery_timing("insight_generation", elapsed_time, len(insights))
        if insights:
            self.engine.send_progress('insights_generated', {
                'insights': insights,
                'plot_path': plot_path,
                'report_path': report_path,
                'log_histories': log_histories
            })
            report_generator = ScientificReportGenerator(
                challenge_name=self.engine.challenge.name,
                algorithm_names=[alg.name for alg in self.engine.algorithms],
                final_results=final_results,
                plot_path=plot_path
            )
            report_content = report_generator.generate_report(insights)
            with open(report_path, "w") as f:
                f.write(report_content)
        else:
            self.engine.send_progress('no_insights')
        self.engine.send_progress('end_phase', {'phase': 'insight_generation'})
        return insights
