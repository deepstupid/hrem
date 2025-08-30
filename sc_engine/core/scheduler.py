"""Discovery-aware scheduler for scientific algorithm comparison."""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from .config import AlgorithmConfig, ChallengeConfig

@dataclass
class TerminationCriteria:
    """Criteria for early stopping of algorithm evaluation."""
    max_trials: int
    min_improvement: float
    patience: int  # Number of trials without improvement before stopping

@dataclass
class ExplorationPriority:
    """Priority ranking for exploration activities."""
    algorithm_order: List[str]   # Priority order of algorithms
    resource_allocation: Dict[str, float]  # Resource percentage per algorithm
    termination_criteria: Dict[str, TerminationCriteria]  # Early stopping rules

@dataclass
class PerformanceSignal:
    """Performance signal from algorithm evaluation."""
    algorithm_name: str
    metric_name: str
    current_value: float
    trend: str  # 'improving', 'degrading', 'stable'
    confidence: float  # 0.0-1.0

@dataclass
class ExplorationDepth:
    """Depth of exploration for an algorithm."""
    max_trials: int
    search_space_exploration: float  # 0.0-1.0 percentage of search space to explore

class DiscoveryAwareScheduler:
    """Prioritizes exploration activities based on scientific discovery potential."""
    
    def __init__(self, algorithms: List[AlgorithmConfig], challenge: ChallengeConfig):
        """Initialize with algorithms and challenge context."""
        self.algorithms = {alg.name: alg for alg in algorithms}
        self.challenge = challenge
        self.performance_history: Dict[str, List[float]] = {alg.name: [] for alg in algorithms}
        
    def prioritize_exploration(self, current_results: Optional[Dict[str, Any]] = None) -> ExplorationPriority:
        """
        Rank exploration activities by discovery potential.
        
        Args:
            current_results: Current comparison metrics (if available)
            
        Returns:
            ExplorationPriority object with ranking and allocation
        """
        # Start with a default order based on theoretical advantages
        algorithm_names = list(self.algorithms.keys())
        
        # If we have current results, adjust priority based on performance
        if current_results:
            # Sort algorithms by current performance (lower loss is better)
            algorithm_performance = []
            for alg_name in algorithm_names:
                # Get performance metric (this is simplified)
                loss = current_results.get(alg_name, {}).get('all/lm_loss', float('inf'))
                algorithm_performance.append((alg_name, loss))
            
            # Sort by performance
            algorithm_performance.sort(key=lambda x: x[1])
            algorithm_names = [name for name, _ in algorithm_performance]
        
        # Allocate resources - initially equal distribution
        resource_allocation = {name: 1.0/len(algorithm_names) for name in algorithm_names}
        
        # Set termination criteria based on algorithm characteristics
        termination_criteria = {}
        for alg_name in algorithm_names:
            alg_config = self.algorithms[alg_name]
            
            # More complex algorithms get more trials but with early stopping
            max_trials = 10 if 'HREM' in alg_name else 5
            
            # Set minimum improvement based on algorithm sensitivity
            min_improvement = 0.01 if 'HREM' in alg_name else 0.005
            
            termination_criteria[alg_name] = TerminationCriteria(
                max_trials=max_trials,
                min_improvement=min_improvement,
                patience=3  # Stop after 3 trials without improvement
            )
        
        return ExplorationPriority(
            algorithm_order=algorithm_names,
            resource_allocation=resource_allocation,
            termination_criteria=termination_criteria
        )
    
    def adjust_algorithm_depth(self, algorithm_name: str, 
                             performance_signal: Optional[PerformanceSignal] = None) -> ExplorationDepth:
        """
        Adjust exploration depth based on performance signals.
        
        Args:
            algorithm_name: Name of the algorithm
            performance_signal: Performance signal (if available)
            
        Returns:
            ExplorationDepth object with adjusted depth
        """
        # Default exploration depth
        max_trials = 10
        search_space_exploration = 0.5  # Explore 50% of search space by default
        
        # Adjust based on performance signal
        if performance_signal:
            # If algorithm is performing well and showing improvement, explore more
            if performance_signal.trend == 'improving' and performance_signal.confidence > 0.8:
                max_trials = min(20, max_trials * 2)
                search_space_exploration = min(1.0, search_space_exploration * 1.5)
            # If algorithm is degrading or stable, reduce exploration
            elif performance_signal.trend == 'degrading':
                max_trials = max(3, max_trials // 2)
                search_space_exploration = max(0.2, search_space_exploration * 0.5)
        
        # Adjust based on algorithm type
        if 'HREM' in algorithm_name:
            # HREM has larger search space, so explore more carefully
            search_space_exploration = min(1.0, search_space_exploration * 0.8)
        else:
            # HRM is simpler, can explore more of its smaller search space
            search_space_exploration = min(1.0, search_space_exploration * 1.2)
        
        return ExplorationDepth(
            max_trials=max_trials,
            search_space_exploration=search_space_exploration
        )