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

from .demo_timing_utils import EnhancedTimingManager as TimingManager

class ScientificTimingManager(TimingManager):
    """Extended timing manager for scientific discovery metrics."""

    def __init__(self):
        super().__init__()
        self.discovery_metrics: Dict[str, list] = {}

    def record_discovery_timing(self, activity: str, elapsed_time: float, insights_generated: int):
        """
        Record timing with scientific discovery metrics.

        Args:
            activity: Name of the activity
            elapsed_time: Time elapsed for the activity
            insights_generated: Number of insights generated during the activity
        """
        # Record standard timing
        self.record_timing(activity, elapsed_time)

        # Record discovery-specific metrics
        self.record_metric(f"discovery_insights_{activity}", insights_generated)
        self.record_metric(f"discovery_efficiency_{activity}",
                          insights_generated / elapsed_time if elapsed_time > 0 else 0)

        # Store in discovery metrics
        if f"discovery_{activity}" not in self.discovery_metrics:
            self.discovery_metrics[f"discovery_{activity}"] = []
        self.discovery_metrics[f"discovery_{activity}"].append({
            'elapsed_time': elapsed_time,
            'insights_generated': insights_generated,
            'timestamp': self.current_iteration
        })

    def predict_discovery_value(self, time_investment: float) -> Dict[str, Any]:
        """
        Predict scientific value of time investment.

        Args:
            time_investment: Amount of time to invest

        Returns:
            Dictionary with expected insights and value metrics
        """
        # This is a simplified prediction model
        # In practice, this would be based on historical data and machine learning

        # Calculate average insights per second from historical data
        avg_insights_per_second = 0.1  # Placeholder value

        # Get recent discovery efficiency
        recent_efficiency = self.get_metric_stats("discovery_efficiency")
        if recent_efficiency and recent_efficiency.get('avg'):
            avg_insights_per_second = recent_efficiency['avg']

        expected_insights = time_investment * avg_insights_per_second
        discovery_value = min(1.0, expected_insights / 5.0)  # Normalize to 0-1 scale

        return {
            'expected_insights': expected_insights,
            'discovery_value': discovery_value,
            'confidence': 0.7  # Simplified confidence measure
        }

    def get_discovery_stats(self) -> Dict[str, Any]:
        """Get statistics on discovery metrics."""
        stats = {}
        for metric_name in self.discovery_metrics.keys():
            if self.discovery_metrics[metric_name]:
                values = self.discovery_metrics[metric_name]
                insights = [v['insights_generated'] for v in values]
                times = [v['elapsed_time'] for v in values]

                stats[metric_name] = {
                    'total_insights': sum(insights),
                    'avg_insights_per_run': sum(insights) / len(insights) if insights else 0,
                    'total_time': sum(times),
                    'avg_time_per_run': sum(times) / len(times) if times else 0
                }

        return stats

import os
from typing import List, Dict, Any
from collections import defaultdict
from .insights import ScientificInsight, InsightType, Evidence

INSIGHT_EMOJIS = {
    InsightType.META: "🌟",
    InsightType.HYPOTHESIS: "🎯",
    InsightType.EFFICIENCY: "🚀",
    InsightType.SCALABILITY: "📈",
    InsightType.ROBUSTNESS: "🛡️",
    InsightType.CONVERGENCE: "📉",
    InsightType.GENERALIZATION: "🌍",
    InsightType.ADAPTABILITY: "🔧",
    InsightType.FAILURE: "💥",
}

class ScientificReportGenerator:
    """Generates a Markdown report from a list of scientific insights."""

    def __init__(self, challenge_name: str, algorithm_names: List[str], final_results: Dict[str, Any] = None, plot_path: str = None):
        """Initialize the report generator."""
        self.challenge_name = challenge_name
        self.algorithm_names = algorithm_names
        self.final_results = final_results
        self.plot_path = plot_path

    def generate_report(self, insights: List[ScientificInsight]) -> str:
        """Generate a Markdown report from a list of insights."""
        report_parts = [self._generate_header()]

        if self.final_results:
            report_parts.append(self._generate_summary_table())
            report_parts.append("---")

        if self.plot_path:
            report_parts.append(self._generate_plot_section())
            report_parts.append("---")

        if not insights:
            report_parts.append("## No significant insights were generated in this run.")
            return "\n".join(report_parts)

        meta_insights = [i for i in insights if i.type == InsightType.META]
        hypothesis_insights = [i for i in insights if i.type == InsightType.HYPOTHESIS]
        failure_insights = [i for i in insights if i.type == InsightType.FAILURE]
        other_insights = [i for i in insights if i.type not in [InsightType.META, InsightType.HYPOTHESIS, InsightType.FAILURE]]

        report_parts.append(self._generate_executive_summary(meta_insights, failure_insights))
        report_parts.append("---")

        if hypothesis_insights:
            report_parts.append(self._generate_hypothesis_section(hypothesis_insights))
            report_parts.append("---")

        if other_insights:
            report_parts.append(self._generate_detailed_findings(other_insights))
            report_parts.append("---")

        if failure_insights:
            report_parts.append(self._generate_failure_analysis_section(failure_insights))
            report_parts.append("---")

        return "\n".join(report_parts)

    def _generate_header(self) -> str:
        """Generate the report header."""
        return f"# Scientific Comparison Report: {self.challenge_name}\n\n**Algorithms Compared:** `{'`, `'.join(self.algorithm_names)}`\n"

    def _generate_summary_table(self) -> str:
        """Generates a markdown table of the final results."""
        section = "## 📈 Final Metrics Summary\n\n"
        if not self.final_results:
            return ""

        headers = ["Algorithm"]
        all_keys = set()
        for result in self.final_results.values():
            # The result for each algorithm is now a dictionary of metrics for each dataset split (e.g., 'all')
            for split_result in result.values():
                if isinstance(split_result, dict):
                    all_keys.update(split_result.keys())

        sorted_keys = sorted(list(all_keys), key=lambda x: ('loss' not in x, 'accuracy' not in x, x))
        headers.extend([key.replace('_', ' ').title() for key in sorted_keys])

        section += f"| {' | '.join(headers)} |\n"
        section += f"|{'|'.join(['---'] * len(headers))}|\n"

        for alg_name, results in self.final_results.items():
            row = [alg_name.replace('_optimized', ' (Optimized)')]
            # We will display the 'all' split results in the summary table
            summary_metrics = results.get('all', {})
            for key in sorted_keys:
                value = summary_metrics.get(key)
                if isinstance(value, float):
                    row.append(f"{value:.4f}")
                else:
                    row.append(str(value) if value is not None else "N/A")
            section += f"| {' | '.join(row)} |\n"

        return section + "\n"

    def _generate_plot_section(self) -> str:
        """Generates the markdown section for the performance plot."""
        section = "## 📊 Performance Plot\n\n"
        if not self.plot_path or not os.path.exists(self.plot_path):
            return section + "No performance plot was generated for this run.\n"

        relative_plot_path = os.path.relpath(self.plot_path, start=os.getcwd())
        section += f"![Performance Plot]({relative_plot_path})\n"
        return section + "\n"

    def _generate_executive_summary(self, meta_insights: List[ScientificInsight], failure_insights: List[ScientificInsight]) -> str:
        summary = "## 🌟 Executive Summary\n\n"
        if failure_insights:
            summary += "**Warning:** One or more models encountered failures during the run. See the Failure Analysis section for details.\n\n"
        if not meta_insights:
            summary += "No overarching meta-insights were identified. See detailed findings below.\n"
            return summary
        for insight in meta_insights:
            summary += f"### {insight.title}\n{insight.summary}\n\n"
            if insight.recommendations:
                summary += "**Top Recommendation:**\n"
                for rec in insight.recommendations:
                    summary += f"> {rec}\n"
        return summary

    def _generate_hypothesis_section(self, insights: List[ScientificInsight]) -> str:
        section = "## 🎯 Hypothesis Testing\n\n"
        for insight in sorted(insights, key=lambda i: i.title):
            section += self._format_insight(insight)
            section += "\n"
        return section

    def _generate_failure_analysis_section(self, insights: List[ScientificInsight]) -> str:
        section = "## 💥 Failure Analysis\n\n"
        for insight in insights:
            section += self._format_insight(insight)
            section += "\n"
        return section

    def _generate_detailed_findings(self, insights: List[ScientificInsight]) -> str:
        section = "## 📊 Detailed Findings\n\n"
        grouped_insights = defaultdict(list)
        for insight in insights:
            grouped_insights[insight.type].append(insight)
        for insight_type, insight_list in sorted(grouped_insights.items(), key=lambda item: item[0].value):
            emoji = INSIGHT_EMOJIS.get(insight_type, "🔹")
            section += f"### {emoji} {insight_type.value.replace('_', ' ').title()}\n\n"
            for insight in sorted(insight_list, key=lambda i: i.discovery_potential, reverse=True):
                section += self._format_insight(insight)
                section += "\n"
        return section

    def _format_insight(self, insight: ScientificInsight) -> str:
        emoji = INSIGHT_EMOJIS.get(insight.type, "🔹")
        formatted = f"#### {emoji} {insight.title}\n\n"
        if insight.type not in [InsightType.HYPOTHESIS, InsightType.META]:
             formatted += f"**Confidence:** `{insight.confidence:.1%}` | **Discovery Potential:** `{insight.discovery_potential:.1%}`\n\n"
        if insight.summary:
            formatted += f"_{insight.summary}_\n\n"
        if insight.implications:
            formatted += "**Implications:**\n"
            for implication in insight.implications:
                formatted += f"- {implication}\n"
            formatted += "\n"
        if insight.causal_attribution:
            formatted += f"**Causal Attribution:** {insight.causal_attribution}\n\n"
        if insight.recommendations:
            formatted += "**Recommendations:**\n"
            for rec in insight.recommendations:
                formatted += f"- {rec}\n"
            formatted += "\n"
        if insight.evidence:
            formatted += "**Evidence:**\n| Metric | Value | Effect Size | p-value | Description |\n|---|---|---|---|---|\n"
            for ev in insight.evidence:
                value_str = f"{ev.metric_value:.4f}" if isinstance(ev.metric_value, float) else str(ev.metric_value)
                effect_str = f"{ev.effect_size:.3f}" if ev.effect_size is not None else "N/A"
                pval_str = f"{ev.p_value:.3f}" if ev.p_value is not None else "N/A"
                formatted += f"| {ev.metric_name} | `{value_str}` | `{effect_str}` | `{pval_str}` | {ev.description or ''} |\n"
        return formatted

import math
import itertools
import numpy as np
from typing import List, Dict, Any, Tuple
from scipy import stats
from omegaconf import OmegaConf
from .config import ChallengeConfig, ChallengeLevel
from .insights import ScientificInsight, InsightType, Evidence
from .schemas import InsightConfigSchema

from .config import AlgorithmConfig

class ScientificInsightGenerator:
    """Extracts meaningful scientific insights from algorithm comparisons."""
    
    def __init__(self, challenge: ChallengeConfig, algorithms: List[AlgorithmConfig], config_path: str = "config/insight_config.yaml"):
        """Initialize with challenge context and configuration."""
        self.challenge = challenge
        self.algorithms = {alg.name: alg for alg in algorithms}
        raw_config = OmegaConf.load(config_path)
        self.config = InsightConfigSchema(**raw_config).model_dump()

        # Map insight types to their extraction methods
        self.comparison_results = None
        self.insight_method_map = {
            "efficiency": self._extract_efficiency_insights,
            "scalability": self._extract_scalability_insights,
            "convergence": self._extract_convergence_insights,
            "robustness": self._extract_robustness_insights,
            "generalization": self._extract_generalization_insights,
            "adaptability": self._extract_adaptability_insights,
            "failure": self._extract_failure_insights,
            "meta": self._synthesize_meta_insights,
            "test_hypothesis": self._test_hypotheses,
            "generated_hypothesis": self._generate_hypotheses,
        }
        
    def extract_insights(self, comparison_results: Dict[str, Any]) -> List[ScientificInsight]:
        """
        Extract scientific insights from comparison results based on the configured pipeline.
        
        Args:
            comparison_results: Results from algorithm comparison
            
        Returns:
            List of ScientificInsight objects
        """
        self.comparison_results = comparison_results
        insights = []
        pipeline = self.config.get("insight_pipeline", [])

        for insight_type in pipeline:
            method = self.insight_method_map.get(insight_type)
            if not method:
                print(f"Warning: Unknown insight type '{insight_type}' in pipeline.")
                continue

            # Methods that depend on prior insights vs. raw results
            if insight_type in ["meta", "test_hypothesis", "generated_hypothesis"]:
                new_insights = method(insights)
            else:
                new_insights = method(comparison_results)

            if new_insights:
                insights.extend(new_insights)
        
        return insights

    def _extract_failure_insights(self, comparison_results: Dict[str, Any]) -> List[ScientificInsight]:
        """Extract insights about model or training failures."""
        insights = []
        for alg_name, results in comparison_results.items():
            # Check for NaN loss, which indicates a training failure
            if 'all/lm_loss' in results and results['all/lm_loss'] is not None and math.isnan(results['all/lm_loss']):
                insights.append(ScientificInsight(
                    type=InsightType.FAILURE,
                    title=f"Training Failure: {alg_name}",
                    confidence=1.0, # We are certain about the failure
                    discovery_potential=0.5, # Moderately interesting, might indicate instability
                    implications=[f"{alg_name} failed to train, resulting in a NaN loss. This could be due to numerical instability, exploding gradients, or an inappropriate learning rate."],
                    recommendations=["Investigate the learning rate and gradient clipping parameters for this model. Check for potential sources of instability in the architecture."],
                    evidence=[Evidence(metric_name="final_loss", metric_value="NaN")]
                ))

            # Check for missing results, indicating a run failure
            if not results:
                 insights.append(ScientificInsight(
                    type=InsightType.FAILURE,
                    title=f"Run Failure: {alg_name}",
                    confidence=1.0,
                    discovery_potential=0.4,
                    implications=[f"{alg_name} failed to produce any results. The run may have crashed or been terminated prematurely."],
                    recommendations=["Check the logs for this run to diagnose the cause of the failure."],
                    evidence=[Evidence(metric_name="run_status", metric_value="failed")]
                ))

        return insights

    def _test_hypotheses(self, insights: List[ScientificInsight]) -> List[ScientificInsight]:
        """Test scientific hypotheses against the generated insights."""
        hypothesis_insights = []
        if not self.challenge.hypothesis_space:
            return hypothesis_insights

        # Map metric types to insight types
        metric_to_insight_type = {
            "efficiency": InsightType.EFFICIENCY,
            "scalability": InsightType.SCALABILITY,
            "robustness": InsightType.ROBUSTNESS,
            "convergence": InsightType.CONVERGENCE,
        }

        for hypothesis in self.challenge.hypothesis_space:
            insight_type = metric_to_insight_type.get(hypothesis.metric)
            if not insight_type:
                continue

            related_insights = [i for i in insights if i.type == insight_type]

            status = "Inconclusive"
            evidence_summary = "No direct evidence found to support or refute this hypothesis."
            confidence = 0.5

            if related_insights:
                # For now, assume the first relevant insight is the primary evidence
                primary_evidence = related_insights[0]
                actual_winner = self._get_winner_from_summary(primary_evidence.summary)

                if actual_winner:
                    if actual_winner == hypothesis.expected_winner:
                        status = "Confirmed"
                        evidence_summary = f"Hypothesis confirmed: {actual_winner} was found to be superior. Evidence: {primary_evidence.summary}"
                    else:
                        status = "Refuted"
                        evidence_summary = f"Hypothesis refuted: {actual_winner} was found to be superior, not {hypothesis.expected_winner}. Evidence: {primary_evidence.summary}"
                    confidence = primary_evidence.confidence

            hypothesis_insights.append(ScientificInsight(
                type=InsightType.HYPOTHESIS,
                title=f"Hypothesis ({status}): {hypothesis.description}",
                confidence=confidence,
                discovery_potential=0.8, # High potential as it directly addresses a research question
                implications=[evidence_summary],
                hypothesis=hypothesis.description,
                evidence=[Evidence(metric_name="hypothesis_status", metric_value=status)]
            ))

        return hypothesis_insights
    
    def _get_winner_from_summary(self, summary: str) -> str | None:
        """A simple heuristic to determine the winner from an insight summary."""
        summary_lower = summary.lower()
        for alg_name in self.algorithms.keys():
            # Check if the summary starts with the algorithm name, indicating it's the subject.
            if summary_lower.startswith(alg_name.lower()):
                return alg_name
        return None

    def _synthesize_meta_insights(self, insights: List[ScientificInsight]) -> List[ScientificInsight]:
        """Synthesize meta-insights from a list of individual insights."""
        meta_insights = []

        # --- 1. Dominance Analysis ---
        winner_counts = {alg_name: 0 for alg_name in self.algorithms}
        winning_categories = {alg_name: [] for alg_name in self.algorithms}

        for insight in insights:
            if insight.type in [InsightType.EFFICIENCY, InsightType.SCALABILITY, InsightType.ROBUSTNESS, InsightType.CONVERGENCE]:
                winner = self._get_winner_from_summary(insight.summary)
                if winner and winner in winner_counts:
                    winner_counts[winner] += 1
                    winning_categories[winner].append(insight.type.value)

        for alg, count in winner_counts.items():
            if count >= self.config["meta_insight_threshold"]:
                categories = list(set(winning_categories[alg]))
                meta_insights.append(ScientificInsight(
                    type=InsightType.META,
                    title=f"Dominant Performance: {alg}",
                    confidence=self._calculate_confidence_from_effect(count, k=1),
                    discovery_potential=0.95,
                    implications=[
                        f"{alg} demonstrates superior performance across multiple dimensions ({count} categories: {', '.join(categories)}).",
                        f"This suggests {alg} is a dominant architecture for the '{self.challenge.name}' challenge."
                    ],
                    recommendations=[f"Consider {alg} as the default choice for problems similar to this challenge."],
                    evidence=[
                        Evidence(metric_name="dominant_algorithm", metric_value=alg),
                        Evidence(metric_name="number_of_wins", metric_value=count),
                        Evidence(metric_name="winning_categories", metric_value=categories),
                    ]
                ))

        # --- 2. Pareto Front Analysis for Performance vs. Efficiency ---

        # Gather performance (loss) and efficiency (timing) data
        perf_data = {
            name: results.get('all/lm_loss')
            for name, results in self.comparison_results.items()
            if results.get('all/lm_loss') is not None and not math.isnan(results.get('all/lm_loss'))
        }

        time_data = {
            name: results.get('timing')
            for name, results in self.comparison_results.items()
            if results.get('timing') is not None
        }

        # Combine into a list of (name, loss, time) tuples
        combined_data = [
            (name, perf_data.get(name), time_data.get(name))
            for name in self.algorithms
            if name in perf_data and name in time_data
        ]

        if len(combined_data) >= 2:
            pareto_front = []
            for i, (name1, loss1, time1) in enumerate(combined_data):
                is_dominated = False
                for j, (name2, loss2, time2) in enumerate(combined_data):
                    if i == j:
                        continue
                    # Check if p2 dominates p1 (lower is better for both)
                    if loss2 <= loss1 and time2 <= time1 and (loss2 < loss1 or time2 < time1):
                        is_dominated = True
                        break
                if not is_dominated:
                    pareto_front.append((name1, loss1, time1))

            if len(pareto_front) > 1 or (len(pareto_front) == 1 and len(combined_data) > 1):
                # Generate an insight if there's a trade-off (multiple optimal points) or one algorithm dominates all others.

                # Sort front for consistent reporting: by loss, then by time
                pareto_front.sort(key=lambda x: (x[1], x[2]))

                implications = [
                    "A trade-off between performance (lower loss) and efficiency (lower time) was identified.",
                    f"The following {len(pareto_front)} algorithm(s) represent the optimal choices along this trade-off frontier (the Pareto Front):"
                ]

                recommendations = []
                evidence_list = []

                for name, loss, time in pareto_front:
                    implications.append(f"- **{name}**: Achieves a loss of {loss:.4f} in {time:.2f} seconds.")
                    recommendations.append(f"Consider **{name}** for its specific balance of performance and efficiency.")
                    evidence_list.append(Evidence(metric_name="pareto_optimal_point", metric_value=name, description=f"Loss: {loss:.4f}, Time: {time:.2f}s"))

                if len(pareto_front) == 1:
                    winner = pareto_front[0][0]
                    title = f"Dominant Performance & Efficiency: {winner}"
                    summary = f"{winner} is dominant, outperforming all other algorithms in either performance, efficiency, or both."
                else:
                    title = "Performance vs. Efficiency Trade-off (Pareto Front)"
                    summary = f"Found a Pareto front with {len(pareto_front)} optimal algorithms, revealing a trade-off between performance and efficiency."


                meta_insights.append(ScientificInsight(
                    type=InsightType.META,
                    title=title,
                    summary=summary,
                    confidence=0.9,
                    discovery_potential=0.85,
                    implications=implications,
                    recommendations=recommendations,
                    evidence=evidence_list
                ))

        return meta_insights

    def _extract_efficiency_insights(self, comparison_results: Dict[str, Any]) -> List[ScientificInsight]:
        """Extract efficiency-related insights by comparing all pairs of algorithms."""
        insights = []
        
        timing_data = {
            name: results['timing']
            for name, results in comparison_results.items()
            if 'timing' in results and results['timing'] is not None
        }

        if len(timing_data) < 2:
            return insights

        for (alg1_name, time1), (alg2_name, time2) in itertools.combinations(timing_data.items(), 2):
            if time1 == 0 or time2 == 0:
                continue

            if time1 < time2:
                faster_model_name, slower_model_name = alg1_name, alg2_name
                speedup = time2 / time1
            else:
                faster_model_name, slower_model_name = alg2_name, alg1_name
                speedup = time1 / time2

            if speedup > self.config["efficiency_speedup_threshold"]:
                faster_alg_config = self.algorithms.get(faster_model_name)
                slower_alg_config = self.algorithms.get(slower_model_name)

                causal_attribution = None
                if faster_alg_config and slower_alg_config:
                    # Primary causal attribution from complexity profile
                    if faster_alg_config.complexity_profile and slower_alg_config.complexity_profile:
                        shared_metrics = set(faster_alg_config.complexity_profile.keys()) & set(slower_alg_config.complexity_profile.keys())
                        for metric in shared_metrics:
                            val_faster = faster_alg_config.complexity_profile[metric]
                            val_slower = slower_alg_config.complexity_profile[metric]
                            if isinstance(val_faster, (int, float)) and isinstance(val_slower, (int, float)) and val_faster < val_slower:
                                causal_attribution = f"The superior efficiency of {faster_model_name} may be attributed to its lower {metric} ({val_faster} vs. {val_slower} for {slower_model_name})."
                                break # Stop after finding one explanation

                    is_surprising = faster_alg_config.complexity > slower_alg_config.complexity
                    base_potential = self.config["efficiency_surprising_potential"] if is_surprising else self.config["efficiency_expected_potential"]

                    if is_surprising:
                        implication_text = f"{faster_model_name} is surprisingly more computationally efficient than {slower_model_name} despite its higher complexity."
                        # Secondary attribution if primary is missing
                        if not causal_attribution and faster_alg_config.theoretical_advantages:
                            advantages = ', '.join(faster_alg_config.theoretical_advantages)
                            causal_attribution = f"The surprising efficiency of {faster_model_name} could be attributed to its {advantages}, which may overcome its inherent complexity on this specific challenge."
                    else:
                        implication_text = f"{faster_model_name} is more computationally efficient than {slower_model_name}, as expected for a simpler model."
                        # Secondary attribution if primary is missing
                        if not causal_attribution and slower_alg_config.theoretical_limitations:
                            limitations = ', '.join(slower_alg_config.theoretical_limitations)
                            causal_attribution = f"The slower performance of {slower_model_name} aligns with its theoretical limitations, such as {limitations}."
                else:
                    base_potential = self.config["efficiency_expected_potential"]
                    implication_text = f"{faster_model_name} is more computationally efficient than {slower_model_name}."

                boost_factor = self.config["efficiency_potential_boost"] if speedup > self.config["efficiency_high_speedup_threshold"] else 0.0
                discovery_potential = self.classify_discovery_potential(base_potential, boost_factor)
                confidence = self._calculate_confidence_from_effect(speedup - 1, k=2)

                insights.append(ScientificInsight(
                    type=InsightType.EFFICIENCY,
                    title=f"Efficiency: {faster_model_name} vs {slower_model_name}",
                    confidence=confidence,
                    discovery_potential=discovery_potential,
                    summary=f"{faster_model_name} is {speedup:.2f}x faster than {slower_model_name}.",
                    implications=[
                        implication_text,
                        f"The {speedup:.2f}x speedup could be critical for resource-constrained environments."
                    ],
                    evidence=[
                        Evidence(
                            metric_name="speedup_factor",
                            metric_value=round(speedup, 2),
                            description=f"{faster_model_name} ({time1:.2f}s) vs {slower_model_name} ({time2:.2f}s)",
                            effect_size=round(speedup, 2)
                        )
                    ],
                    causal_attribution=causal_attribution
                ))
        
        return insights
    
    def _extract_scalability_insights(self, comparison_results: Dict[str, Any]) -> List[ScientificInsight]:
        """Extract scalability-related insights by comparing all pairs of algorithms."""
        insights = []

        performance_data = {
            name: results['all/lm_loss']
            for name, results in comparison_results.items()
            if 'all/lm_loss' in results and results['all/lm_loss'] is not None and not math.isnan(results['all/lm_loss'])
        }

        if len(performance_data) < 2:
            return insights

        for (alg1_name, loss1), (alg2_name, loss2) in itertools.combinations(performance_data.items(), 2):
            if loss1 < loss2:
                winner_name, loser_name = alg1_name, alg2_name
                winner_loss, loser_loss = loss1, loss2
            else:
                winner_name, loser_name = alg2_name, alg1_name
                winner_loss, loser_loss = loss2, loss1

            performance_gap = (loser_loss - winner_loss) / loser_loss if loser_loss > 0 else 0

            if performance_gap < self.config["scalability_performance_gap_threshold"]:
                continue

            implication = f"{winner_name} outperforms {loser_name} on the '{self.challenge.name}' challenge, suggesting better scalability with task complexity."
            recommendation = f"For tasks similar to '{self.challenge.name}', {winner_name} is the recommended architecture due to its superior scalability."

            base_potential = 0.5
            causal_attribution = None
            winner_config = self.algorithms.get(winner_name)
            loser_config = self.algorithms.get(loser_name)

            if winner_config and loser_config:
                # Primary causal attribution from complexity profile
                if winner_config.complexity_profile and loser_config.complexity_profile:
                    shared_metrics = set(winner_config.complexity_profile.keys()) & set(loser_config.complexity_profile.keys())
                    for metric in shared_metrics:
                        val_winner = winner_config.complexity_profile[metric]
                        val_loser = loser_config.complexity_profile[metric]
                        if isinstance(val_winner, (int, float)) and isinstance(val_loser, (int, float)) and val_winner > val_loser:
                            causal_attribution = f"The superior performance of {winner_name} may be due to its higher {metric} ({val_winner} vs. {val_loser} for {loser_name}), allowing for greater capacity."
                            break

                is_winner_less_complex = winner_config.complexity < loser_config.complexity
                is_hard_task = self.challenge.difficulty in [ChallengeLevel.ADVANCED, ChallengeLevel.INTERMEDIATE]

                # Secondary attribution if primary is missing
                if not causal_attribution:
                    if winner_config.theoretical_advantages:
                        advantages = ', '.join(winner_config.theoretical_advantages)
                        causal_attribution = f"{winner_name}'s superior scalability may be due to its {advantages}."
                    elif loser_config.theoretical_limitations:
                        limitations = ', '.join(loser_config.theoretical_limitations)
                        causal_attribution = f"{loser_name}'s difficulty in scaling could be linked to its {limitations}."

                if is_winner_less_complex and is_hard_task:
                    base_potential = self.config["scalability_surprising_potential_hard"]
                elif not is_winner_less_complex and not is_hard_task and performance_gap > self.config["scalability_performance_gap_threshold"]:
                    base_potential = self.config["scalability_surprising_potential_easy"]
                elif not is_winner_less_complex and is_hard_task:
                    base_potential = self.config["scalability_expected_potential_hard"]
                elif is_winner_less_complex and not is_hard_task:
                    base_potential = self.config["scalability_expected_potential_easy"]

            boost_factor = self.config["scalability_potential_boost"] if performance_gap > self.config["scalability_high_performance_gap_threshold"] else 0.0
            discovery_potential = self.classify_discovery_potential(base_potential, boost_factor)
            confidence = self._calculate_confidence_from_effect(performance_gap, k=5)

            insights.append(ScientificInsight(
                type=InsightType.SCALABILITY,
                title=f"Scalability: {winner_name} vs {loser_name}",
                confidence=confidence,
                discovery_potential=discovery_potential,
                summary=f"{winner_name} outperforms {loser_name} by {performance_gap:.2%}.",
                implications=[implication],
                recommendations=[recommendation],
                causal_attribution=causal_attribution,
                evidence=[
                    Evidence(
                        metric_name="performance_gap",
                        metric_value=f"{performance_gap:.2%}",
                        description=f"Final loss comparison: {winner_name} ({winner_loss:.4f}) vs {loser_name} ({loser_loss:.4f})",
                        effect_size=performance_gap
                    )
                ]
            ))

        return insights

    def _extract_convergence_insights(self, comparison_results: Dict[str, Any]) -> List[ScientificInsight]:
        """Extract convergence-related insights by analyzing loss history."""
        insights = []
        
        # Get loss history data
        convergence_data = {}
        for alg_name, results in comparison_results.items():
            if 'loss_history' in results and results['loss_history']:
                convergence_data[alg_name] = results['loss_history']

        # --- Insight 1: Final Loss Comparison ---
        final_losses = {}
        for alg_name, results in comparison_results.items():
            if 'all/lm_loss' in results:
                final_losses[alg_name] = results['all/lm_loss']
        
        if len(final_losses) >= 2:
            best_alg, best_loss = min(final_losses.items(), key=lambda item: item[1])
            worst_alg, worst_loss = max(final_losses.items(), key=lambda item: item[1])

            best_alg_config = self.algorithms.get(best_alg)
            causal_attribution = None
            if best_alg_config and best_alg_config.theoretical_advantages:
                advantages = ', '.join(best_alg_config.theoretical_advantages)
                causal_attribution = f"{best_alg}'s ability to reach a lower loss may stem from its {advantages}."

            loss_diff = (worst_loss - best_loss) / worst_loss if worst_loss > 0 else 0
            confidence = self._calculate_confidence_from_effect(loss_diff, k=10)

            insights.append(ScientificInsight(
                type=InsightType.CONVERGENCE,
                title="Final Loss Comparison",
                confidence=confidence,
                discovery_potential=0.6, # Placeholder - potential could be linked to magnitude of improvement
                implications=[
                    f"{best_alg} achieves a lower final loss, indicating more optimal convergence.",
                    "This suggests its architecture is better suited to this problem's loss landscape."
                ],
                causal_attribution=causal_attribution,
                evidence=[
                    Evidence(metric_name="best_final_loss", metric_value=best_loss, description=f"Algorithm: {best_alg}", effect_size=(worst_loss - best_loss)),
                    Evidence(metric_name="worst_final_loss", metric_value=worst_loss, description=f"Algorithm: {worst_alg}")
                ]
            ))

        # --- Insight 2: Convergence Speed Analysis ---
        if len(convergence_data) >= 2:
            convergence_speed = {}
            for alg_name, loss_history in convergence_data.items():
                if not loss_history: continue
                initial_loss = loss_history[0]
                final_loss = loss_history[-1]
                # Threshold: point at which 90% of the learning is done
                threshold = final_loss + self.config["convergence_threshold_percent"] * (initial_loss - final_loss)

                steps_to_converge = next((i for i, loss in enumerate(loss_history) if loss <= threshold), len(loss_history))
                convergence_speed[alg_name] = steps_to_converge

            for (alg1_name, steps1), (alg2_name, steps2) in itertools.combinations(convergence_speed.items(), 2):
                if steps1 == 0 or steps2 == 0: continue

                if steps1 < steps2:
                    faster_alg, slower_alg = alg1_name, alg2_name
                    faster_steps, slower_steps = steps1, steps2
                else:
                    faster_alg, slower_alg = alg2_name, alg1_name
                    faster_steps, slower_steps = steps2, steps1

                speed_ratio = slower_steps / faster_steps
                if speed_ratio > self.config["convergence_speed_ratio_threshold"]:
                    faster_alg_config = self.algorithms.get(faster_alg)
                    slower_alg_config = self.algorithms.get(slower_alg)

                    causal_attribution = None
                    if faster_alg_config and faster_alg_config.theoretical_advantages:
                        advantages = ', '.join(faster_alg_config.theoretical_advantages)
                        causal_attribution = f"The faster convergence of {faster_alg} might be explained by its {advantages}."

                    is_surprising = faster_alg_config and slower_alg_config and faster_alg_config.complexity > slower_alg_config.complexity
                    base_potential = self.config["convergence_potential_surprising"] if is_surprising else self.config["convergence_potential_expected"]
                    discovery_potential = self.classify_discovery_potential(base_potential)
                    confidence = self._calculate_confidence_from_effect(speed_ratio - 1, k=2)

                    insights.append(ScientificInsight(
                        type=InsightType.CONVERGENCE,
                        title=f"Convergence Speed: {faster_alg} vs {slower_alg}",
                        confidence=confidence,
                        discovery_potential=discovery_potential,
                        summary=f"{faster_alg} converges {speed_ratio:.2f}x faster than {slower_alg}.",
                        implications=[
                            f"{faster_alg} converges significantly faster than {slower_alg}, which can reduce training time and cost.",
                            "This suggests a more efficient learning process or a better-suited architecture for the problem's loss landscape."
                        ],
                        causal_attribution=causal_attribution,
                        evidence=[
                            Evidence(metric_name="convergence_speed_ratio", metric_value=round(speed_ratio, 2), effect_size=round(speed_ratio, 2)),
                            Evidence(metric_name="steps_to_converge", metric_value=faster_steps, description=f"Algorithm: {faster_alg}"),
                            Evidence(metric_name="steps_to_converge", metric_value=slower_steps, description=f"Algorithm: {slower_alg}")
                        ]
                    ))
        
        return insights
    
    def _extract_robustness_insights(self, comparison_results: Dict[str, Any]) -> List[ScientificInsight]:
        """Extract robustness-related insights by analyzing performance variance."""
        insights = []
        
        # Get performance data, expecting lists of values for robustness analysis
        robustness_data = {}
        for alg_name, results in comparison_results.items():
            # Check for list-like data for a specific metric, e.g., from multiple runs
            if 'all/lm_loss_runs' in results and isinstance(results['all/lm_loss_runs'], list) and len(results['all/lm_loss_runs']) > 1:
                robustness_data[alg_name] = results['all/lm_loss_runs']

        if len(robustness_data) < 2:
            return insights

        # Calculate standard deviation for each model
        stdevs = {alg_name: np.std(losses) for alg_name, losses in robustness_data.items()}
        
        most_robust_alg = min(stdevs, key=stdevs.get)
        least_robust_alg = max(stdevs, key=stdevs.get)

        # Generate insight if the difference in robustness is significant
        if stdevs[least_robust_alg] > stdevs[most_robust_alg] * self.config["robustness_variance_threshold"]:
            p_value = self.calculate_statistical_significance(
                robustness_data[most_robust_alg],
                robustness_data[least_robust_alg]
            )
            confidence = 1.0 - p_value

            if confidence > (1.0 - self.config["statistical_significance_threshold"]):
                most_robust_config = self.algorithms.get(most_robust_alg)
                least_robust_config = self.algorithms.get(least_robust_alg)

                causal_attribution = None
                if most_robust_config and most_robust_config.theoretical_advantages:
                    advantages = ', '.join(most_robust_config.theoretical_advantages)
                    causal_attribution = f"The superior robustness of {most_robust_alg} could be due to its {advantages}."

                is_surprising = most_robust_config and least_robust_config and most_robust_config.complexity > least_robust_config.complexity
                base_potential = self.config["robustness_surprising_potential"] if is_surprising else self.config["robustness_expected_potential"]
                boost_factor = self.config["robustness_potential_boost"] if stdevs[least_robust_alg] > stdevs[most_robust_alg] * self.config["robustness_high_variance_threshold"] else 0.0
                discovery_potential = self.classify_discovery_potential(base_potential, boost_factor)

                effect_size = stdevs[least_robust_alg] / stdevs[most_robust_alg] if stdevs[most_robust_alg] > 0 else float('inf')

                insights.append(ScientificInsight(
                    type=InsightType.ROBUSTNESS,
                    confidence=confidence,
                    discovery_potential=discovery_potential,
                    implications=[
                        f"{most_robust_alg} demonstrates statistically significant higher robustness (lower performance variance) than {least_robust_alg}.",
                        "This reliability is crucial for production environments."
                    ],
                    causal_attribution=causal_attribution,
                    evidence=[
                        Evidence(metric_name="stdev_of_loss", metric_value=round(stdevs[most_robust_alg], 4), description=f"Algorithm: {most_robust_alg}"),
                        Evidence(metric_name="stdev_of_loss", metric_value=round(stdevs[least_robust_alg], 4), description=f"Algorithm: {least_robust_alg}"),
                        Evidence(metric_name="p_value", metric_value=p_value, description="Welch's t-test for variance", p_value=p_value, effect_size=effect_size)
                    ]
                ))

        return insights
    
    @staticmethod
    def calculate_statistical_significance(data1: List[float], data2: List[float]) -> float:
        """
        Calculate statistical significance between two datasets.
        
        Args:
            data1: First dataset
            data2: Second dataset
            
        Returns:
            p-value from a t-test
        """
        if len(data1) < 2 or len(data2) < 2:
            return 1.0
        try:
            # Perform t-test
            _, p_value = stats.ttest_ind(data1, data2, equal_var=False) # Welch's t-test
            return p_value
        except:
            # If statistical test fails, return a non-significant p-value
            return 1.0
    
    def classify_discovery_potential(self, base_potential: float, boost_factor: float = 0.0) -> float:
        """
        Classify the potential of an insight for future research.
        
        Args:
            base_potential: The base discovery potential.
            boost_factor: A factor to boost the potential.
            
        Returns:
            Discovery potential score (0.0-1.0)
        """
        # This is a simplified classification
        # In practice, this would be more sophisticated
        return min(1.0, base_potential + boost_factor)

    def _calculate_confidence_from_effect(self, effect_size: float, k: float = 1.0) -> float:
        """
        Calculate a confidence score from an effect size using a logistic function.
        This ensures confidence is between 0.5 and 1.0.

        Args:
            effect_size: The magnitude of the observed effect.
            k: Steepness of the curve.

        Returns:
            A confidence score between 0.5 and 1.0.
        """
        # Logistic function scaled to [0.5, 1.0]
        return 0.5 + 0.5 * (1 - math.exp(-k * abs(effect_size)))

    def _extract_generalization_insights(self, comparison_results: Dict[str, Any]) -> List[ScientificInsight]:
        """Extract generalization-related insights."""
        insights = []
        for alg_name, results in comparison_results.items():
            if 'all/lm_loss' in results and 'generalization_loss' in results:
                train_loss = results['all/lm_loss']
                gen_loss = results['generalization_loss']

                if train_loss > 0 and gen_loss > (train_loss * (1 + self.config["generalization_drop_off_threshold"])):
                    drop_off = (gen_loss - train_loss) / train_loss
                    confidence = 1.0 - self._calculate_confidence_from_effect(drop_off) # Inverse relationship: higher drop-off is "bad", but we are confident in the finding

                    insights.append(ScientificInsight(
                        type=InsightType.GENERALIZATION,
                        confidence=confidence,
                        discovery_potential=self.config["generalization_potential"],
                        implications=[
                            f"{alg_name} shows a {drop_off:.2%} performance drop on out-of-distribution data, suggesting overfitting."
                        ],
                        recommendations=["Investigating regularization techniques is recommended to improve generalization."],
                        evidence=[Evidence(metric_name="generalization_drop_off", metric_value=f"{drop_off:.2%}", effect_size=drop_off)]
                    ))
        return insights

    def _extract_adaptability_insights(self, comparison_results: Dict[str, Any]) -> List[ScientificInsight]:
        """Extract adaptability-related insights from fine-tuning performance."""
        insights = []
        for alg_name, results in comparison_results.items():
            if 'pre_finetune_loss' in results and 'post_finetune_loss' in results:
                pre_loss = results['pre_finetune_loss']
                post_loss = results['post_finetune_loss']

                improvement = (pre_loss - post_loss) / pre_loss if pre_loss > 0 else 0

                if improvement > self.config["adaptability_improvement_threshold"]:
                    confidence = self._calculate_confidence_from_effect(improvement, k=5)
                    insights.append(ScientificInsight(
                        type=InsightType.ADAPTABILITY,
                        confidence=confidence,
                        discovery_potential=self.config["adaptability_potential"],
                        implications=[
                            f"{alg_name} demonstrates strong adaptability, improving performance by {improvement:.2%} with fine-tuning.",
                            "This suggests the model learns transferable representations."
                        ],
                        recommendations=[f"The strong adaptability of {alg_name} makes it a good candidate for transfer learning scenarios."],
                        evidence=[Evidence(metric_name="finetuning_improvement", metric_value=f"{improvement:.2%}", effect_size=improvement)]
                    ))
        return insights

    def _generate_hypotheses(self, insights: List[ScientificInsight]) -> List[ScientificInsight]:
        """Generate new hypotheses based on existing insights."""
        generated_hypotheses = []

        # Pattern 1: Surprising Performance
        for insight in insights:
            full_text = insight.summary + " ".join(insight.implications)
            if insight.type in [InsightType.EFFICIENCY, InsightType.SCALABILITY] and "surprisingly" in full_text:
                winner_name = self._get_winner_from_summary(insight.summary)
                if winner_name:
                    hypothesis_text = f"The established complexity score for {winner_name} may not accurately reflect its practical performance on this problem class. Further investigation into its architectural efficiencies is warranted."
                    generated_hypotheses.append(ScientificInsight(
                        type=InsightType.GENERATED_HYPOTHESIS,
                        title="Generated Hypothesis: Surprising Performance",
                        summary=hypothesis_text,
                        confidence=0.7, # This is a generated hypothesis, so confidence is moderate
                        discovery_potential=0.9,
                        implications=[hypothesis_text],
                        recommendations=[f"Ablation studies on {winner_name}'s architecture could reveal the source of its surprising performance."]
                    ))

        # Pattern 2: Dominant Algorithm
        for insight in insights:
            if insight.type == InsightType.META and "Dominant Performance" in insight.title:
                winner = insight.evidence[0].metric_value
                categories = [e.metric_value for e in insight.evidence if e.metric_name == 'winning_categories']
                untested_areas = [t for t in ["robustness", "adaptability", "generalization"] if t not in categories]

                if untested_areas:
                    hypothesis_text = f"Given its dominant performance in multiple categories, {winner} is likely to also excel in related areas such as {untested_areas[0]}."
                    generated_hypotheses.append(ScientificInsight(
                        type=InsightType.GENERATED_HYPOTHESIS,
                        title=f"Generated Hypothesis: Extended Dominance of {winner}",
                        summary=hypothesis_text,
                        confidence=0.65,
                        discovery_potential=0.8,
                        implications=[hypothesis_text],
                        recommendations=[f"Run further experiments to test the performance of {winner} on {untested_areas[0]}."]
                    ))

        # Pattern 3: Clear Trade-off on Pareto Front
        for insight in insights:
            if insight.type == InsightType.META and "Pareto Front" in insight.title:
                pareto_points = [e.metric_value for e in insight.evidence if e.metric_name == 'pareto_optimal_point']
                if len(pareto_points) > 1:
                    alg1, alg2 = pareto_points[0], pareto_points[-1] # Simplification: compare the extremes of the front
                    hypothesis_text = f"A fundamental trade-off exists between the architectural approaches of {alg1} and {alg2}. The former appears to prioritize efficiency, while the latter excels in performance."
                    generated_hypotheses.append(ScientificInsight(
                        type=InsightType.GENERATED_HYPOTHESIS,
                        title="Generated Hypothesis: Architectural Trade-off",
                        summary=hypothesis_text,
                        confidence=0.75,
                        discovery_potential=0.85,
                        implications=[hypothesis_text],
                        recommendations=[f"Investigate the architectural differences between {alg1} and {alg2} to understand the root cause of the performance/efficiency trade-off."]
                    ))

        return generated_hypotheses
