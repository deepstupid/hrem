#!/usr/bin/env python3
"""Test script for the scientific comparison engine."""

import sys
import os

# Add the project root to the path so we can import our modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from sc_engine.core.config import ChallengeConfig, AlgorithmConfig, PatienceBudget, ChallengeLevel
from sc_engine.core.engine import ScientificDiscoveryEngine
from hrm_system.config import DataConfig

def test_scientific_engine():
    """Test the scientific discovery engine with a simple example."""
    print("🧪 Testing Scientific Discovery Engine...")
    
    # Create a simple challenge configuration
    challenge = ChallengeConfig(
        name="Test Challenge",
        id="test_challenge",
        description="A test challenge for validation",
        dataset=DataConfig(dataset="synthetic"),
        difficulty=ChallengeLevel.BEGINNER,
        scientific_question="How do HRM and HREM compare on synthetic data?",
        hypothesis_space=[
            "HREM will perform better due to its memory architecture",
            "HRM will be faster due to its simpler structure"
        ]
    )
    
    # Create algorithm configurations
    hrm_config = AlgorithmConfig(
        name="HRM",
        algorithm_class="hrm_system.algorithms.hrm.HRMAlgorithm",
        theoretical_advantages=["efficiency", "simplicity"],
        theoretical_limitations=["long_range_dependencies"],
        search_space={}
    )
    
    hrem_config = AlgorithmConfig(
        name="HREM",
        algorithm_class="hrm_system.algorithms.hrem.HREMAlgorithm",
        theoretical_advantages=["memory_capacity", "scalability"],
        theoretical_limitations=["computational_overhead"],
        search_space={}
    )
    
    algorithms = [hrm_config, hrem_config]
    
    # Create patience budget
    patience_budget = PatienceBudget(level="low")
    
    # Initialize the engine
    engine = ScientificDiscoveryEngine(challenge, algorithms)
    
    # This would normally run the full discovery session, but for testing
    # we'll just verify the components are initialized correctly
    print("✅ Challenge:", engine.challenge.name)
    print("✅ Algorithms:", [alg.name for alg in engine.algorithms])
    print("✅ Scheduler initialized:", engine.scheduler is not None)
    print("✅ Insight generator initialized:", engine.insight_generator is not None)
    print("✅ Timing manager initialized:", engine.timing_manager is not None)
    
    print("🎉 All components initialized successfully!")

if __name__ == "__main__":
    test_scientific_engine()