#!/usr/bin/env python3
"""Comprehensive test for the scientific comparison engine."""

import sys
import os

# Add the project root to the path so we can import our modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from sc_engine.core.config import ChallengeConfig, AlgorithmConfig, PatienceBudget, ChallengeLevel, Hypothesis
from sc_engine.core.engine import ScientificDiscoveryEngine

def test_full_workflow():
    """Test the full scientific discovery workflow."""
    print("🧪 Testing Full Scientific Discovery Workflow...")
    
    # Create a challenge configuration
    challenge = ChallengeConfig(
        name="Sequence Duplication Challenge",
        id="sequence_duplication",
        description="Test algorithms on duplicating sequences which requires memory and pattern recognition",
        dataset={"dataset": "synthetic-duplicate"},
        difficulty=ChallengeLevel.BEGINNER,
        scientific_question="How do HRM and HREM differ in pattern duplication tasks?",
        hypothesis_space=[
            Hypothesis(description="HREM will be more efficient than HRM", metric="efficiency", expected_winner="HREM", expected_loser="HRM"),
            Hypothesis(description="HRM will be more robust than HREM", metric="robustness", expected_winner="HRM", expected_loser="HREM"),
        ]
    )
    
    # Create algorithm configurations
    hrm_config = AlgorithmConfig(
        name="HRM",
        algorithm_class="sc_engine.plugins.algorithms.hrm.HRMAlgorithm",
            config={
                    "name": "hrm.act@HierarchicalReasoningModel_ACTV1",
                "algorithm_class": "sc_engine.plugins.algorithms.hrm.HRMAlgorithm",
                    "loss": {"name": "models.losses@ACTLossHead", "loss_type": "softmax_cross_entropy"},
                    "H_cycles": 1, "L_cycles": 1, "H_layers": 1, "L_layers": 1,
                    "hidden_size": 128, "expansion": 2.0, "num_heads": 4,
                    "pos_encodings": "rope", "halt_max_steps": 1, "halt_exploration_prob": 0.0,
                        "puzzle_emb_ndim": 16,
            },
        theoretical_advantages=["computational_efficiency", "simplicity_of_architecture"],
        theoretical_limitations=["limited_memory_capacity", "difficulty_with_long_range_dependencies"],
        search_space={
            "hidden_size": {"type": "int", "low": 64, "high": 256},
            "num_layers": {"type": "int", "low": 1, "high": 3}
        }
    )
    
    hrem_config = AlgorithmConfig(
        name="HREM",
        algorithm_class="sc_engine.plugins.algorithms.hrem.HREMAlgorithm",
            config={
                    "name": "hrm.hrem@HREM",
                "algorithm_class": "sc_engine.plugins.algorithms.hrem.HREMAlgorithm",
                    "loss": {"name": "models.losses@ACTLossHead", "loss_type": "softmax_cross_entropy"},
                    "H_cycles": 1, "L_cycles": 1, "H_layers": 1, "L_layers": 1,
                    "hidden_size": 128, "expansion": 2.0, "num_heads": 4,
                    "pos_encodings": "rope", "halt_max_steps": 1, "halt_exploration_prob": 0.0,
                    "use_memory": True,
                        "puzzle_emb_ndim": 16,
            },
        theoretical_advantages=["large_memory_capacity", "efficient_information_retrieval", "scalable_to_long_sequences"],
        theoretical_limitations=["computational_overhead", "complexity_of_implementation"],
        search_space={
            "memory_size": {"type": "int", "low": 32, "high": 128},
            "top_k_retrieval": {"type": "int", "low": 3, "high": 10},
            "num_memory_layers": {"type": "int", "low": 2, "high": 5}
        }
    )
    
    algorithms = [hrm_config, hrem_config]
    
    # Create patience budget
    patience_budget = PatienceBudget(level="low")
    
    # Initialize the engine
    engine = ScientificDiscoveryEngine(challenge, algorithms, config={"n_trials": 1})
    engine.default_training_config['epochs'] = 1
    engine.default_training_config['smoke_test'] = True
    
    # Execute the discovery session
    results = engine.execute_discovery_session(patience_budget)
    
    # Verify results
    assert results.challenge.name == "Sequence Duplication Challenge"
    assert len(results.algorithm_results) > 0
    assert len(results.insights) > 0
    assert len(results.timing_data) > 0
    assert 'session_start_time' in results.metadata
    
    print("✅ Challenge:", results.challenge.name)
    print("✅ Algorithms tested:", list(results.algorithm_results.keys()))
    print("✅ Insights generated:", len(results.insights))
    print("✅ Timing data collected:", len(results.timing_data) > 0)
    
    # Display a sample insight
    if results.insights:
        sample_insight = results.insights[0]
        print(f"✅ Sample insight: {sample_insight.type} (Confidence: {sample_insight.confidence:.2f})")
    
    print("🎉 Full workflow test completed successfully!")

if __name__ == "__main__":
    test_full_workflow()