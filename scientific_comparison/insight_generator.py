"""Scientific insight generation from algorithm comparisons."""

import math
import numpy as np
from typing import List, Dict, Any, Tuple
from scipy import stats
from .config import ChallengeConfig
from .patience_manager import ScientificInsight

class InsightType:
    """Types of scientific insights."""
    EFFICIENCY = "efficiency"
    ROBUSTNESS = "robustness"
    SCALABILITY = "scalability"
    GENERALIZATION = "generalization"
    CONVERGENCE = "convergence"
    ADAPTABILITY = "adaptability"

class ScientificInsightGenerator:
    """Extracts meaningful scientific insights from algorithm comparisons."""
    
    def __init__(self, challenge: ChallengeConfig):
        """Initialize with challenge context."""
        self.challenge = challenge
        
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
        efficiency_insights = self._extract_efficiency_insights(comparison_results)
        insights.extend(efficiency_insights)
        
        # Extract scalability insights
        scalability_insights = self._extract_scalability_insights(comparison_results)
        insights.extend(scalability_insights)
        
        # Extract convergence insights
        convergence_insights = self._extract_convergence_insights(comparison_results)
        insights.extend(convergence_insights)
        
        # Extract robustness insights
        robustness_insights = self._extract_robustness_insights(comparison_results)
        insights.extend(robustness_insights)
        
        return insights
    
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
            
            if speedup > 1.2:  # At least 20% faster
                faster_model_name = alg_names[faster_idx]
                
                # Base discovery potential on surprise
                if 'HREM' in faster_model_name:
                    # Surprising result: complex model is faster
                    discovery_potential = 0.8
                    implication_text = f"{faster_model_name} is surprisingly more efficient than {alg_names[slower_idx]}."
                else:
                    # Expected result: simpler model is faster
                    discovery_potential = 0.4
                    implication_text = f"{faster_model_name} is more efficient than {alg_names[slower_idx]}, as expected for a simpler model."

                # Boost potential based on magnitude of speedup
                if speedup > 2.0: # More than 2x faster
                    discovery_potential = min(1.0, discovery_potential + 0.2)

                confidence = 0.9 # High confidence as it's based on direct timing metrics

                insights.append(ScientificInsight(
                    type=InsightType.EFFICIENCY,
                    confidence=confidence,
                    evidence=[{
                        'faster_algorithm': faster_model_name,
                        'slower_algorithm': alg_names[slower_idx],
                        'speedup_factor': round(speedup, 2),
                    }],
                    implications=[
                        implication_text,
                        "This efficiency difference could be critical for deployment in resource-constrained environments."
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

        # Find HREM and HRM in the results
        hrem_idx = -1
        hrm_idx = -1
        for i, name in enumerate(alg_names):
            if 'HREM' in name:
                hrem_idx = i
            if 'HRM' in name:
                hrm_idx = i

        if hrem_idx == -1 or hrm_idx == -1:
            return insights

        hrem_loss = losses[hrem_idx]
        hrm_loss = losses[hrm_idx]

        # Determine which model performed better (lower loss is better)
        if hrem_loss < hrm_loss:
            winner = alg_names[hrem_idx]
            loser = alg_names[hrm_idx]
            performance_gap = (hrm_loss - hrem_loss) / hrm_loss if hrm_loss > 0 else 0
            implication = f"{winner} outperforms {loser} on this task, suggesting better scalability with task complexity."
        else:
            winner = alg_names[hrm_idx]
            loser = alg_names[hrem_idx]
            performance_gap = (hrem_loss - hrm_loss) / hrem_loss if hrem_loss > 0 else 0
            implication = f"{winner} outperforms {loser}, suggesting it's more efficient for this level of task complexity."

        # Determine discovery potential based on surprise
        task_difficulty_str = str(self.challenge.difficulty).split('.')[-1].upper()
        discovery_potential = 0.5  # Default

        is_hrem_winner = 'HREM' in winner

        # Surprising if simpler model (HRM) wins on a hard task
        if not is_hrem_winner and task_difficulty_str in ["ADVANCED", "INTERMEDIATE"]:
            discovery_potential = 0.8
        # Surprising if complex model (HREM) wins decisively on an easy task
        elif is_hrem_winner and task_difficulty_str == "BEGINNER" and performance_gap > 0.1:
            discovery_potential = 0.7
        # Expected for HREM to win on complex tasks
        elif is_hrem_winner and task_difficulty_str in ["ADVANCED", "INTERMEDIATE"]:
            discovery_potential = 0.6
        # Expected for HRM to win on simple tasks
        elif not is_hrem_winner and task_difficulty_str == "BEGINNER":
            discovery_potential = 0.4

        # Boost potential if the performance gap is large
        if performance_gap > 0.25:
            discovery_potential = min(1.0, discovery_potential + 0.2)

        insights.append(ScientificInsight(
            type=InsightType.SCALABILITY,
            confidence=0.85,  # High confidence as it's based on direct metrics
            evidence=[{
                'winning_algorithm': winner,
                'losing_algorithm': loser,
                'performance_metric': 'final_loss',
                'performance_gap': f"{performance_gap:.2%}",
                'task_difficulty': task_difficulty_str
            }],
            implications=[
                implication,
                "The choice of model architecture is critical for performance on tasks of this nature."
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
                confidence=0.75,
                evidence=[{
                    'best_algorithm': best_alg,
                    'final_loss': final_losses[best_alg],
                    'comparison': final_losses
                }],
                implications=[
                    f"{best_alg} achieves a lower final loss, indicating better overall performance.",
                    "The model's architecture directly impacts its ability to minimize the objective function."
                ],
                discovery_potential=0.6
            ))

        # --- Insight 2: Convergence Speed Analysis ---
        if len(convergence_data) >= 2:
            convergence_speed = {}
            for alg_name, loss_history in convergence_data.items():
                initial_loss = loss_history[0]
                final_loss = loss_history[-1]
                # Threshold: point at which 90% of the learning is done
                threshold = final_loss + 0.1 * (initial_loss - final_loss)

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

                if speed_ratio > 1.5:  # At least 50% faster to converge
                    # Base discovery potential on surprise
                    if 'HREM' in faster_alg:
                        # Surprising: complex model converges faster
                        discovery_potential = 0.85
                    else:
                        # Expected: simpler model converges faster
                        discovery_potential = 0.5

                    insights.append(ScientificInsight(
                        type=InsightType.CONVERGENCE,
                        confidence=0.8,
                        evidence=[{
                            'faster_converging_algorithm': faster_alg,
                            'slower_converging_algorithm': slower_alg,
                            'convergence_speed_ratio': round(speed_ratio, 2),
                            'metric': 'steps_to_reach_90%_of_learning'
                        }],
                        implications=[
                            f"{faster_alg} converges significantly faster than {slower_alg}.",
                            "Faster convergence can lead to reduced training time and costs."
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
        if stdevs[least_robust_alg] > stdevs[most_robust_alg] * 1.5:  # 50% more variance

            if 'HREM' in most_robust_alg:
                discovery_potential = 0.7
            else:
                discovery_potential = 0.6

            if stdevs[least_robust_alg] > stdevs[most_robust_alg] * 3:
                discovery_potential = min(1.0, discovery_potential + 0.25)

            insights.append(ScientificInsight(
                type=InsightType.ROBUSTNESS,
                confidence=0.8,
                evidence=[{
                    'most_robust_algorithm': most_robust_alg,
                    'least_robust_algorithm': least_robust_alg,
                    'metric': 'standard_deviation_of_loss',
                    'comparison': {k: round(v, 4) for k, v in stdevs.items()}
                }],
                implications=[
                    f"{most_robust_alg} shows higher robustness (lower performance variance) than {least_robust_alg}.",
                    "High robustness is crucial for reliable performance in real-world applications."
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
        try:
            # Perform t-test
            _, p_value = stats.ttest_ind(data1, data2)
            return p_value
        except:
            # If statistical test fails, return a non-significant p-value
            return 1.0
    
    def classify_discovery_potential(self, insight: ScientificInsight) -> float:
        """
        Classify the potential of an insight for future research.
        
        Args:
            insight: ScientificInsight to classify
            
        Returns:
            Discovery potential score (0.0-1.0)
        """
        # This is a simplified classification
        # In practice, this would be more sophisticated
        return insight.discovery_potential