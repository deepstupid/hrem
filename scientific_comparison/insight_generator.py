"""Scientific insight generation from algorithm comparisons."""

import math
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
        
    def extract_insights(self, comparison_results: Dict[str, Any]) -> List[ScientificInsight]:
        """
        Extract scientific insights from comparison results.
        
        Args:
            comparison_results: Results from algorithm comparison
            
        Returns:
            List of ScientificInsight objects
        """
        insights = []
        
        # Extract efficiency insights
        insights.extend(self._extract_efficiency_insights(comparison_results))
        
        # Extract scalability insights
        insights.extend(self._extract_scalability_insights(comparison_results))
        
        # Extract convergence insights
        insights.extend(self._extract_convergence_insights(comparison_results))
        
        # Extract robustness insights
        insights.extend(self._extract_robustness_insights(comparison_results))

        # Extract generalization insights
        insights.extend(self._extract_generalization_insights(comparison_results))

        # Extract adaptability insights
        insights.extend(self._extract_adaptability_insights(comparison_results))

        # Synthesize meta-insights from the collected individual insights
        insights.extend(self._synthesize_meta_insights(insights))

        # Test hypotheses against the generated insights
        insights.extend(self._test_hypotheses(insights))

        # Extract failure insights
        insights.extend(self._extract_failure_insights(comparison_results))
        
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

        # --- 2. Trade-off Analysis ---
        efficiency_winner = None
        scalability_winner = None
        for insight in insights:
            if insight.type == InsightType.EFFICIENCY:
                efficiency_winner = self._get_winner_from_summary(insight.summary)
            if insight.type == InsightType.SCALABILITY:
                scalability_winner = self._get_winner_from_summary(insight.summary)

        if efficiency_winner and scalability_winner and efficiency_winner != scalability_winner:
            meta_insights.append(ScientificInsight(
                type=InsightType.META,
                title="Performance vs. Efficiency Trade-off",
                confidence=0.9, # High confidence as it's based on two conflicting insights
                discovery_potential=0.8,
                implications=[
                    f"A trade-off between performance and efficiency was observed.",
                    f"{scalability_winner} delivers better performance (scalability), while {efficiency_winner} is more computationally efficient."
                ],
                recommendations=[
                    f"Choose {scalability_winner} for maximum performance if resources are not a constraint.",
                    f"Choose {efficiency_winner} for resource-constrained environments where speed is critical."
                ],
                evidence=[
                    Evidence(metric_name="performance_winner", metric_value=scalability_winner),
                    Evidence(metric_name="efficiency_winner", metric_value=efficiency_winner),
                ]
            ))

        return meta_insights

    def _extract_efficiency_insights(self, comparison_results: Dict[str, Any]) -> List[ScientificInsight]:
        """Extract efficiency-related insights."""
        insights = []
        
        # Get timing data
        timing_data = {}
        for alg_name, results in comparison_results.items():
            if 'timing' in results:
                timing_data[alg_name] = results['timing']

        if len(timing_data) < 2:
            return insights

        alg_names = list(timing_data.keys())
        times = list(timing_data.values())
        
        # Compare efficiency
        if len(times) >= 2:
            # Calculate relative efficiency
            faster_idx = 0 if times[0] < times[1] else 1
            slower_idx = 1 - faster_idx

            # Avoid division by zero
            if times[faster_idx] == 0:
                return insights

            speedup = times[slower_idx] / times[faster_idx]
            
            if speedup > self.config["efficiency_speedup_threshold"]:
                faster_model_name = alg_names[faster_idx]
                slower_model_name = alg_names[slower_idx]
                
                # Base discovery potential on surprise
                faster_alg_config = self.algorithms.get(faster_model_name)
                slower_alg_config = self.algorithms.get(slower_model_name)

                causal_attribution = None
                if faster_alg_config and slower_alg_config:
                    is_surprising = faster_alg_config.complexity > slower_alg_config.complexity
                    if is_surprising:
                        # Surprising result: more complex model is faster
                        base_potential = self.config["efficiency_surprising_potential"]
                        implication_text = f"{faster_model_name} is surprisingly more computationally efficient than {slower_model_name} despite its higher complexity."
                        if faster_alg_config.theoretical_advantages:
                            advantages = ', '.join(faster_alg_config.theoretical_advantages)
                            causal_attribution = f"The surprising efficiency of {faster_model_name} could be attributed to its {advantages}, which may overcome its inherent complexity on this specific challenge."
                    else:
                        # Expected result: simpler model is faster
                        base_potential = self.config["efficiency_expected_potential"]
                        implication_text = f"{faster_model_name} is more computationally efficient than {slower_model_name}, as expected for a simpler model."
                        if slower_alg_config.theoretical_limitations:
                            limitations = ', '.join(slower_alg_config.theoretical_limitations)
                            causal_attribution = f"The slower performance of {slower_model_name} aligns with its theoretical limitations, such as {limitations}."
                else:
                    base_potential = self.config["efficiency_expected_potential"]
                    implication_text = f"{faster_model_name} is more computationally efficient than {slower_model_name}."


                # Boost potential based on magnitude of speedup
                boost_factor = self.config["efficiency_potential_boost"] if speedup > self.config["efficiency_high_speedup_threshold"] else 0.0
                discovery_potential = self.classify_discovery_potential(base_potential, boost_factor)
                confidence = self._calculate_confidence_from_effect(speedup - 1, k=2)

                insights.append(ScientificInsight(
                    type=InsightType.EFFICIENCY,
                    confidence=confidence,
                    discovery_potential=discovery_potential,
                    implications=[
                        implication_text,
                        f"The {speedup:.2f}x speedup could be critical for resource-constrained environments."
                    ],
                    evidence=[
                        Evidence(
                            metric_name="speedup_factor",
                            metric_value=round(speedup, 2),
                            description=f"{faster_model_name} vs {slower_model_name}",
                            effect_size=round(speedup, 2)
                        )
                    ],
                    causal_attribution=causal_attribution
                ))
        
        return insights
    
    def _extract_scalability_insights(self, comparison_results: Dict[str, Any]) -> List[ScientificInsight]:
        """Extract scalability-related insights based on actual performance."""
        insights = []

        # Get performance data (final loss)
        performance_data = {}
        for alg_name, results in comparison_results.items():
            if 'all/lm_loss' in results:
                performance_data[alg_name] = results['all/lm_loss']

        if len(performance_data) < 2:
            return insights

        alg_names = list(performance_data.keys())
        losses = list(performance_data.values())

        # This method assumes a comparison between two algorithms.
        # For a more general implementation, this logic would need to be extended.
        if len(alg_names) != 2:
            return insights

        alg1_name, alg2_name = alg_names[0], alg_names[1]
        alg1_loss, alg2_loss = losses[0], losses[1]

        # Determine winner and loser
        if alg1_loss < alg2_loss:
            winner, loser = alg1_name, alg2_name
            performance_gap = (alg2_loss - alg1_loss) / alg2_loss if alg2_loss > 0 else 0
        else:
            winner, loser = alg2_name, alg1_name
            performance_gap = (alg1_loss - alg2_loss) / alg1_loss if alg1_loss > 0 else 0

        implication = f"{winner} outperforms {loser} on the '{self.challenge.name}' challenge, suggesting better scalability with task complexity."
        recommendation = f"For tasks similar to '{self.challenge.name}', {winner} is the recommended architecture due to its superior scalability."

        # Determine discovery potential and causal attribution based on surprise
        base_potential = 0.5  # Default
        causal_attribution = None
        winner_config = self.algorithms.get(winner)
        loser_config = self.algorithms.get(loser)

        if winner_config and loser_config:
            is_winner_less_complex = winner_config.complexity < loser_config.complexity
            is_hard_task = self.challenge.difficulty in [ChallengeLevel.ADVANCED, ChallengeLevel.INTERMEDIATE]

            # Causal attribution based on theoretical advantages/limitations
            if winner_config.theoretical_advantages:
                advantages = ', '.join(winner_config.theoretical_advantages)
                causal_attribution = f"{winner}'s superior scalability may be due to its {advantages}."
            elif loser_config.theoretical_limitations:
                limitations = ', '.join(loser_config.theoretical_limitations)
                causal_attribution = f"{loser}'s difficulty in scaling could be linked to its {limitations}."

            # Surprising if a less complex model wins on a hard task
            if is_winner_less_complex and is_hard_task:
                base_potential = self.config["scalability_surprising_potential_hard"]
            # Surprising if a more complex model wins decisively on an easy task
            elif not is_winner_less_complex and not is_hard_task and performance_gap > self.config["scalability_performance_gap_threshold"]:
                base_potential = self.config["scalability_surprising_potential_easy"]
            # Expected for more complex model to win on hard tasks
            elif not is_winner_less_complex and is_hard_task:
                base_potential = self.config["scalability_expected_potential_hard"]
            # Expected for less complex model to win on easy tasks
            elif is_winner_less_complex and not is_hard_task:
                base_potential = self.config["scalability_expected_potential_easy"]

        # Boost potential if the performance gap is large
        boost_factor = self.config["scalability_potential_boost"] if performance_gap > self.config["scalability_high_performance_gap_threshold"] else 0.0
        discovery_potential = self.classify_discovery_potential(base_potential, boost_factor)
        confidence = self._calculate_confidence_from_effect(performance_gap, k=5)

        insights.append(ScientificInsight(
            type=InsightType.SCALABILITY,
            confidence=confidence,
            discovery_potential=discovery_potential,
            implications=[implication],
            recommendations=[recommendation],
            causal_attribution=causal_attribution,
            evidence=[
                Evidence(
                    metric_name="performance_gap",
                    metric_value=f"{performance_gap:.2%}",
                    description=f"Final loss comparison between {winner} ({list(performance_data.values())[0]:.4f}) and {loser} ({list(performance_data.values())[1]:.4f})",
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
                initial_loss = loss_history[0]
                final_loss = loss_history[-1]
                # Threshold: point at which 90% of the learning is done
                threshold = final_loss + self.config["convergence_threshold_percent"] * (initial_loss - final_loss)

                steps_to_converge = next((i for i, loss in enumerate(loss_history) if loss <= threshold), len(loss_history))
                convergence_speed[alg_name] = steps_to_converge

            if len(convergence_speed) >= 2:
                faster_alg, faster_steps = min(convergence_speed.items(), key=lambda item: item[1])
                slower_alg, slower_steps = max(convergence_speed.items(), key=lambda item: item[1])

                if faster_steps > 0:
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
                            title="Convergence Speed Analysis",
                            confidence=confidence,
                            discovery_potential=discovery_potential,
                            implications=[
                                f"{faster_alg} converges {speed_ratio:.2f}x faster than {slower_alg}, reducing training time.",
                                "This suggests a more efficient learning process."
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
                base_potential = self.config["robustness_hrem_potential"] if is_surprising else self.config["robustness_hrm_potential"]
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