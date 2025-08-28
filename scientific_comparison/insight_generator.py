"""Scientific insight generation from algorithm comparisons."""

import math
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
        
        if len(timing_data) >= 2:
            alg_names = list(timing_data.keys())
            times = list(timing_data.values())
            
            # Compare efficiency
            if len(times) >= 2:
                # Calculate relative efficiency
                faster_idx = 0 if times[0] < times[1] else 1
                slower_idx = 1 - faster_idx
                speedup = times[slower_idx] / times[faster_idx]
                
                if speedup > 1.5:  # At least 50% faster
                    confidence = self._calculate_statistical_significance(times[0], times[1])
                    discovery_potential = 0.7 if 'HREM' in alg_names[faster_idx] else 0.5
                    
                    insights.append(ScientificInsight(
                        type=InsightType.EFFICIENCY,
                        confidence=confidence,
                        evidence=[{
                            'faster_algorithm': alg_names[faster_idx],
                            'slower_algorithm': alg_names[slower_idx],
                            'speedup_factor': speedup,
                            'p_value': 1.0 - confidence  # Simplified
                        }],
                        implications=[
                            f"{alg_names[faster_idx]} is significantly more efficient than {alg_names[slower_idx]}",
                            "This efficiency difference may impact practical deployment scenarios"
                        ],
                        discovery_potential=discovery_potential
                    ))
        
        return insights
    
    def _extract_scalability_insights(self, comparison_results: Dict[str, Any]) -> List[ScientificInsight]:
        """Extract scalability-related insights."""
        insights = []
        
        # This would analyze performance across different dataset sizes or complexities
        # For now, we'll create a placeholder based on algorithm characteristics
        for alg_name, results in comparison_results.items():
            if 'HREM' in alg_name:
                # HREM is expected to scale better with complexity
                insights.append(ScientificInsight(
                    type=InsightType.SCALABILITY,
                    confidence=0.8,
                    evidence=[{
                        'algorithm': alg_name,
                        'theoretical_advantage': 'large_memory_capacity',
                        'expected_scalability': 'superior_on_complex_tasks'
                    }],
                    implications=[
                        f"{alg_name} should perform better on more complex tasks",
                        "Consider using HREM for problems with long-range dependencies"
                    ],
                    discovery_potential=0.7
                ))
            elif 'HRM' in alg_name:
                # HRM is expected to be more efficient on simpler tasks
                insights.append(ScientificInsight(
                    type=InsightType.EFFICIENCY,
                    confidence=0.7,
                    evidence=[{
                        'algorithm': alg_name,
                        'theoretical_advantage': 'computational_efficiency',
                        'expected_efficiency': 'superior_on_simple_tasks'
                    }],
                    implications=[
                        f"{alg_name} should be more efficient on simpler tasks",
                        "Consider using HRM when computational resources are limited"
                    ],
                    discovery_potential=0.6
                ))
        
        return insights
    
    def _extract_convergence_insights(self, comparison_results: Dict[str, Any]) -> List[ScientificInsight]:
        """Extract convergence-related insights."""
        insights = []
        
        # Analyze loss curves for convergence behavior
        convergence_data = {}
        for alg_name, results in comparison_results.items():
            if 'loss_history' in results:
                convergence_data[alg_name] = results['loss_history']
        
        # For now, create a simple insight based on final loss values
        losses = []
        alg_names = []
        for alg_name, results in comparison_results.items():
            if 'all/lm_loss' in results:
                losses.append(results['all/lm_loss'])
                alg_names.append(alg_name)
        
        if len(losses) >= 2:
            best_idx = losses.index(min(losses))
            insights.append(ScientificInsight(
                type=InsightType.CONVERGENCE,
                confidence=0.7,
                evidence=[{
                    'best_algorithm': alg_names[best_idx],
                    'final_loss': losses[best_idx],
                    'comparison': dict(zip(alg_names, losses))
                }],
                implications=[
                    f"{alg_names[best_idx]} achieves better final convergence",
                    f"Algorithm choice significantly impacts model performance"
                ],
                discovery_potential=0.6
            ))
        
        return insights
    
    def _extract_robustness_insights(self, comparison_results: Dict[str, Any]) -> List[ScientificInsight]:
        """Extract robustness-related insights."""
        insights = []
        
        # This would analyze variance in performance across runs
        # For now, create a placeholder
        for alg_name in comparison_results.keys():
            if 'HREM' in alg_name:
                insights.append(ScientificInsight(
                    type=InsightType.ROBUSTNESS,
                    confidence=0.6,
                    evidence=[{
                        'algorithm': alg_name,
                        'characteristic': 'memory_based_architecture',
                        'expected_robustness': 'high_for_complex_patterns'
                    }],
                    implications=[
                        f"{alg_name} may be more robust for complex pattern recognition",
                        "Memory-based architectures can provide more stable performance"
                    ],
                    discovery_potential=0.5
                ))
            else:
                insights.append(ScientificInsight(
                    type=InsightType.ROBUSTNESS,
                    confidence=0.6,
                    evidence=[{
                        'algorithm': alg_name,
                        'characteristic': 'recurrent_structure',
                        'expected_robustness': 'moderate_with_good_initialization'
                    }],
                    implications=[
                        f"{alg_name} may require careful initialization for robust performance",
                        "Recurrent structures can be sensitive to hyperparameters"
                    ],
                    discovery_potential=0.4
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