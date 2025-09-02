from dataclasses import dataclass
from typing import Dict, Any, List

@dataclass
class DiscoveryResults:
    """Results from a scientific discovery session."""
    challenge: any
    algorithm_results: Dict[str, Any]
    insights: List[Any]
    timing_data: Dict[str, Any]
    metadata: Dict[str, Any]
    log_histories: Dict[str, List[Dict[str, Any]]] = None
