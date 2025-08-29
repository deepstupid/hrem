"""Scientific insight generation from algorithm comparisons."""

import math
import numpy as np
from typing import List, Dict, Any, Tuple
from scipy import stats
from .config import ChallengeConfig, ChallengeLevel
from .patience_manager import ScientificInsight

class InsightType:
    """Types of scientific insights."""
    EFFICIENCY = "efficiency"
    ROBUSTNESS = "robustness"
    SCALABILITY = "scalability"
    GENERALIZATION = "generalization"
    CONVERGENCE = "convergence"
    ADAPTABILITY = "adaptability"
    META = "meta"

class InsightConstants:
    """Constants and thresholds for insight generation."""
    # Efficiency Insights
    EFFICIENCY_SPEEDUP_THRESHOLD = 1.2
    EFFICIENCY_HIGH_SPEEDUP_THRESHOLD = 2.0
    EFFICIENCY_POTENTIAL_BOOST = 0.2
    EFFICIENCY_SURPRISING_POTENTIAL = 0.8
    EFFICIENCY_EXPECTED_POTENTIAL = 0.4

    # Scalability Insights
    SCALABILITY_PERFORMANCE_GAP_THRESHOLD = 0.1
    SCALABILITY_HIGH_PERFORMANCE_GAP_THRESHOLD = 0.25
    SCALABILITY_POTENTIAL_BOOST = 0.2
    SCALABILITY_SURPRISING_POTENTIAL_HARD = 0.8
    SCALABILITY_SURPRISING_POTENTIAL_EASY = 0.7
    SCALABILITY_EXPECTED_POTENTIAL_HARD = 0.6
    SCALABILITY_EXPECTED_POTENTIAL_EASY = 0.4

    # Convergence Insights
    CONVERGENCE_THRESHOLD_PERCENT = 0.1
    CONVERGENCE_SPEED_RATIO_THRESHOLD = 1.5
    CONVERGENCE_POTENTIAL_SURPRISING = 0.85
    CONVERGENCE_POTENTIAL_EXPECTED = 0.5

    # Robustness Insights
    ROBUSTNESS_VARIANCE_THRESHOLD = 1.5
    ROBUSTNESS_HIGH_VARIANCE_THRESHOLD = 3.0
    ROBUSTNESS_POTENTIAL_BOOST = 0.25
    ROBUSTNESS_HREM_POTENTIAL = 0.7
    ROBUSTNESS_HRM_POTENTIAL = 0.6

    # Generalization Insights
    GENERALIZATION_DROP_OFF_THRESHOLD = 0.2
    GENERALIZATION_POTENTIAL = 0.7

    # Adaptability Insights
    ADAPTABILITY_IMPROVEMENT_THRESHOLD = 0.1
    ADAPTABILITY_POTENTIAL = 0.75

    # Meta Insights
    META_INSIGHT_THRESHOLD = 3

from .config import AlgorithmConfig

class ScientificInsightGenerator:
    """Extracts meaningful scientific insights from algorithm comparisons."""
    
    def __init__(self, challenge: ChallengeConfig, algorithms: List[AlgorithmConfig], constants: InsightConstants = InsightConstants()):
        """Initialize with challenge context and constants."""
        self.challenge = challenge
        self.algorithms = {alg.name: alg for alg in algorithms}
        self.constants = constants
        
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
        
        return insights
    
    def _synthesize_meta_insights(self, insights: List[ScientificInsight]) -> List[ScientificInsight]:
        """Synthesize meta-insights from a list of individual insights."""
        meta_insights = []
        winner_counts = {}

        for insight in insights:
            if insight.type == InsightType.EFFICIENCY:
                winner = insight.evidence[0].get('faster_algorithm')
                if winner: winner_counts[winner] = winner_counts.get(winner, 0) + 1
            elif insight.type == InsightType.SCALABILITY:
                winner = insight.evidence[0].get('winning_algorithm')
                if winner: winner_counts[winner] = winner_counts.get(winner, 0) + 1
            elif insight.type == InsightType.ROBUSTNESS:
                winner = insight.evidence[0].get('most_robust_algorithm')
                if winner: winner_counts[winner] = winner_counts.get(winner, 0) + 1
            elif insight.type == InsightType.CONVERGENCE:
                winner = insight.evidence[0].get('best_algorithm')
                if winner: winner_counts[winner] = winner_counts.get(winner, 0) + 1
                winner = insight.evidence[0].get('faster_converging_algorithm')
                if winner: winner_counts[winner] = winner_counts.get(winner, 0) + 1

        for alg, count in winner_counts.items():
            if count >= self.constants.META_INSIGHT_THRESHOLD:
                meta_insights.append(ScientificInsight(
                    type=InsightType.META,
                    confidence=0.9, # High confidence as it's based on multiple sources of evidence
                    evidence=[{
                        'algorithm': alg,
                        'number_of_wins': count,
                        'winning_categories': [
                            i.type for i in insights if i.evidence[0].get('faster_algorithm') == alg or
                                                     i.evidence[0].get('winning_algorithm') == alg or
                                                     i.evidence[0].get('most_robust_algorithm') == alg or
                                                     i.evidence[0].get('best_algorithm') == alg or
                                                     i.evidence[0].get('faster_converging_algorithm') == alg
                        ]
                    }],
                    implications=[
                        f"{alg} demonstrates superior performance across multiple dimensions ({count} categories).",
                        f"This suggests {alg} is a dominant architecture for the '{self.challenge.name}' challenge."
                    ],
                    discovery_potential=0.95 # Very high discovery potential
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
            
            if speedup > self.constants.EFFICIENCY_SPEEDUP_THRESHOLD:
                faster_model_name = alg_names[faster_idx]
                
                # Base discovery potential on surprise
                faster_alg_config = self.algorithms.get(faster_model_name)
                slower_alg_config = self.algorithms.get(alg_names[slower_idx])

                if faster_alg_config and slower_alg_config and faster_alg_config.complexity > slower_alg_config.complexity:
                    # Surprising result: more complex model is faster
                    base_potential = self.constants.EFFICIENCY_SURPRISING_POTENTIAL
                    implication_text = f"{faster_model_name} is surprisingly more computationally efficient than {alg_names[slower_idx]} despite its higher complexity."
                else:
                    # Expected result: simpler model is faster
                    base_potential = self.constants.EFFICIENCY_EXPECTED_POTENTIAL
                    implication_text = f"{faster_model_name} is more computationally efficient than {alg_names[slower_idx]}, as expected for a simpler model."

                # Boost potential based on magnitude of speedup
                boost_factor = self.constants.EFFICIENCY_POTENTIAL_BOOST if speedup > self.constants.EFFICIENCY_HIGH_SPEEDUP_THRESHOLD else 0.0
                discovery_potential = self.classify_discovery_potential(base_potential, boost_factor)

                insights.append(ScientificInsight(
                    type=InsightType.EFFICIENCY,
                    confidence=0.9, # Placeholder
                    evidence=[{
                        'faster_algorithm': faster_model_name,
                        'slower_algorithm': alg_names[slower_idx],
                        'speedup_factor': round(speedup, 2),
                    }],
                    implications=[
                        implication_text,
                        f"This {speedup:.2f}x speedup could be critical for deployment in resource-constrained environments."
                    ],
                    discovery_potential=discovery_potential
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

        # Determine discovery potential based on surprise
        base_potential = 0.5  # Default
        winner_config = self.algorithms.get(winner)
        loser_config = self.algorithms.get(loser)

        if winner_config and loser_config:
            is_winner_less_complex = winner_config.complexity < loser_config.complexity
            is_hard_task = self.challenge.difficulty in [ChallengeLevel.ADVANCED, ChallengeLevel.INTERMEDIATE]

            # Surprising if a less complex model wins on a hard task
            if is_winner_less_complex and is_hard_task:
                base_potential = self.constants.SCALABILITY_SURPRISING_POTENTIAL_HARD
            # Surprising if a more complex model wins decisively on an easy task
            elif not is_winner_less_complex and not is_hard_task and performance_gap > self.constants.SCALABILITY_PERFORMANCE_GAP_THRESHOLD:
                base_potential = self.constants.SCALABILITY_SURPRISING_POTENTIAL_EASY
            # Expected for more complex model to win on hard tasks
            elif not is_winner_less_complex and is_hard_task:
                base_potential = self.constants.SCALABILITY_EXPECTED_POTENTIAL_HARD
            # Expected for less complex model to win on easy tasks
            elif is_winner_less_complex and not is_hard_task:
                base_potential = self.constants.SCALABILITY_EXPECTED_POTENTIAL_EASY

        # Boost potential if the performance gap is large
        boost_factor = self.constants.SCALABILITY_POTENTIAL_BOOST if performance_gap > self.constants.SCALABILITY_HIGH_PERFORMANCE_GAP_THRESHOLD else 0.0
        discovery_potential = self.classify_discovery_potential(base_potential, boost_factor)

        insights.append(ScientificInsight(
            type=InsightType.SCALABILITY,
            confidence=0.85, # Placeholder
            evidence=[{
                'winning_algorithm': winner,
                'losing_algorithm': loser,
                'performance_metric': 'final_loss',
                'performance_gap': f"{performance_gap:.2%}",
                'task_difficulty': self.challenge.difficulty.name
            }],
            implications=[
                implication,
                f"For tasks similar to '{self.challenge.name}', {winner} is the recommended architecture due to its superior performance."
            ],
            discovery_potential=discovery_potential
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
            best_alg = min(final_losses, key=final_losses.get)
            insights.append(ScientificInsight(
                type=InsightType.CONVERGENCE,
                confidence=0.75, # Placeholder
                evidence=[{
                    'best_algorithm': best_alg,
                    'final_loss': final_losses[best_alg],
                    'comparison': final_losses
                }],
                implications=[
                    f"{best_alg} achieves a lower final loss, indicating a more optimal convergence on the given task.",
                    "This suggests its architecture is better suited to navigating the specific loss landscape of this problem."
                ],
                discovery_potential=0.6 # Placeholder
            ))

        # --- Insight 2: Convergence Speed Analysis ---
        if len(convergence_data) >= 2:
            convergence_speed = {}
            for alg_name, loss_history in convergence_data.items():
                initial_loss = loss_history[0]
                final_loss = loss_history[-1]
                # Threshold: point at which 90% of the learning is done
                threshold = final_loss + self.constants.CONVERGENCE_THRESHOLD_PERCENT * (initial_loss - final_loss)

                steps_to_converge = next((i for i, loss in enumerate(loss_history) if loss <= threshold), len(loss_history))
                convergence_speed[alg_name] = steps_to_converge

            if len(convergence_speed) >= 2:
                faster_alg = min(convergence_speed, key=convergence_speed.get)
                slower_alg = max(convergence_speed, key=convergence_speed.get)

                # Avoid division by zero
                if convergence_speed[faster_alg] == 0:
                    speed_ratio = float('inf')
                else:
                    speed_ratio = convergence_speed[slower_alg] / convergence_speed[faster_alg]

                if speed_ratio > self.constants.CONVERGENCE_SPEED_RATIO_THRESHOLD:
                    # Base discovery potential on surprise
                    faster_alg_config = self.algorithms.get(faster_alg)
                    slower_alg_config = self.algorithms.get(slower_alg)

                    if faster_alg_config and slower_alg_config and faster_alg_config.complexity > slower_alg_config.complexity:
                        # Surprising: more complex model converges faster
                        base_potential = self.constants.CONVERGENCE_POTENTIAL_SURPRISING
                    else:
                        # Expected: simpler model converges faster
                        base_potential = self.constants.CONVERGENCE_POTENTIAL_EXPECTED

                    discovery_potential = self.classify_discovery_potential(base_potential)

                    insights.append(ScientificInsight(
                        type=InsightType.CONVERGENCE,
                        confidence=0.8, # Placeholder
                        evidence=[{
                            'faster_converging_algorithm': faster_alg,
                            'slower_converging_algorithm': slower_alg,
                            'convergence_speed_ratio': round(speed_ratio, 2),
                            'metric': 'steps_to_reach_90%_of_learning'
                        }],
                        implications=[
                            f"{faster_alg} converges {speed_ratio:.2f}x faster than {slower_alg}, which can significantly reduce training time and cost.",
                            "This rapid convergence suggests a more efficient learning process, enabling faster iteration during model development."
                        ],
                        discovery_potential=discovery_potential
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
        if stdevs[least_robust_alg] > stdevs[most_robust_alg] * self.constants.ROBUSTNESS_VARIANCE_THRESHOLD:

            most_robust_config = self.algorithms.get(most_robust_alg)
            least_robust_config = self.algorithms.get(least_robust_alg)

            if most_robust_config and least_robust_config and most_robust_config.complexity > least_robust_config.complexity:
                # Surprising: more complex model is more robust
                base_potential = self.constants.ROBUSTNESS_HREM_POTENTIAL # Re-using HREM potential for "surprising"
            else:
                # Expected: simpler model is more robust
                base_potential = self.constants.ROBUSTNESS_HRM_POTENTIAL # Re-using HRM potential for "expected"

            boost_factor = self.constants.ROBUSTNESS_POTENTIAL_BOOST if stdevs[least_robust_alg] > stdevs[most_robust_alg] * self.constants.ROBUSTNESS_HIGH_VARIANCE_THRESHOLD else 0.0
            discovery_potential = self.classify_discovery_potential(base_potential, boost_factor)

            # Calculate statistical significance
            p_value = self.calculate_statistical_significance(
                robustness_data[most_robust_alg],
                robustness_data[least_robust_alg]
            )
            confidence = 1.0 - p_value

            insights.append(ScientificInsight(
                type=InsightType.ROBUSTNESS,
                confidence=confidence,
                evidence=[{
                    'most_robust_algorithm': most_robust_alg,
                    'least_robust_algorithm': least_robust_alg,
                    'metric': 'standard_deviation_of_loss',
                    'comparison': {k: round(v, 4) for k, v in stdevs.items()}
                }],
                implications=[
                    f"{most_robust_alg} demonstrates statistically significant higher robustness (lower performance variance) than {least_robust_alg}.",
                    "This reliability is crucial for deployment in production environments where predictable performance is key."
                ],
                discovery_potential=discovery_potential
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

    def _extract_generalization_insights(self, comparison_results: Dict[str, Any]) -> List[ScientificInsight]:
        """Extract generalization-related insights."""
        insights = []
        for alg_name, results in comparison_results.items():
            if 'all/lm_loss' in results and 'generalization_loss' in results:
                train_loss = results['all/lm_loss']
                gen_loss = results['generalization_loss']

                if train_loss > 0 and gen_loss > (train_loss * (1 + self.constants.GENERALIZATION_DROP_OFF_THRESHOLD)):
                    insights.append(ScientificInsight(
                        type=InsightType.GENERALIZATION,
                        confidence=0.8, # Placeholder
                        evidence=[{
                            'algorithm': alg_name,
                            'training_loss': train_loss,
                            'generalization_loss': gen_loss,
                            'drop_off_percentage': (gen_loss - train_loss) / train_loss
                        }],
                        implications=[
                            f"{alg_name} shows a {((gen_loss - train_loss) / train_loss):.2%} performance drop on out-of-distribution data, suggesting overfitting.",
                            "Investigating regularization techniques (e.g., dropout, weight decay) is recommended to improve this model's generalization."
                        ],
                        discovery_potential=self.constants.GENERALIZATION_POTENTIAL
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

                if improvement > self.constants.ADAPTABILITY_IMPROVEMENT_THRESHOLD:
                    insights.append(ScientificInsight(
                        type=InsightType.ADAPTABILITY,
                        confidence=0.85, # Placeholder
                        evidence=[{
                            'algorithm': alg_name,
                            'pre_finetune_loss': pre_loss,
                            'post_finetune_loss': post_loss,
                            'improvement_percentage': improvement
                        }],
                        implications=[
                            f"{alg_name} demonstrates strong adaptability, improving its performance by {improvement:.2%} with minimal fine-tuning.",
                            "This suggests the model learns transferable representations and can be effectively repurposed for new, related tasks."
                        ],
                        discovery_potential=self.constants.ADAPTABILITY_POTENTIAL
                    ))
        return insights