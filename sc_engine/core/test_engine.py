#!/usr/bin/env python3
"""Comprehensive test for the scientific comparison engine."""

import sys
import os
import unittest.mock as mock

# Add the project root to the path so we can import our modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from sc_engine.core.config import ChallengeConfig, AlgorithmConfig, PatienceBudget, ChallengeLevel, Hypothesis
from sc_engine.core.engine import ScientificDiscoveryEngine
from sc_engine.core.orchestrator import EngineOrchestrator
from dataset_manager import dataset_manager

@mock.patch('puzzle_dataset.PuzzleDataset')
def test_full_workflow(MockPuzzleDataset):
    """Test the full scientific discovery workflow."""
    # Mock the dataset loading
    mock_dataset_instance = MockPuzzleDataset.return_value
    mock_dataset_instance._load_metadata.return_value = mock.MagicMock()

    print("🧪 Testing Full Scientific Discovery Workflow...")
    
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
    
    dataset_manager.get_dataset_path("synthetic-duplicate", smoke_test=True)

    hrm_config = AlgorithmConfig(
        name="HRM",
        algorithm_class="sc_engine.plugins.algorithms.hrm.HRMAlgorithm",
        config={
            "base_arch_config": "hrm_v1", "name": "HRM", "algorithm_class": "sc_engine.plugins.algorithms.hrm.HRMAlgorithm",
        },
        theoretical_advantages=[], theoretical_limitations=[], search_space={}
    )
    
    hrem_config = AlgorithmConfig(
        name="HREM",
        algorithm_class="sc_engine.plugins.algorithms.hrem.HREMAlgorithm",
        config={
            "base_arch_config": "hrem_v1", "name": "HREM", "algorithm_class": "sc_engine.plugins.algorithms.hrem.HREMAlgorithm",
        },
        theoretical_advantages=[], theoretical_limitations=[], search_space={}
    )
    
    algorithms = [hrm_config, hrem_config]
    patience_budget = PatienceBudget(level="low")
    
    default_training_config = {
        'seed': 42, 'global_batch_size': 1, 'epochs': 1, 'eval_interval': 1, 'use_amp': False, 'eval_save_outputs': []
    }

    # Mock the engine's dependencies that perform heavy computation or require real data
    with mock.patch('sc_engine.core.engine.Trainer') as MockTrainer:
        mock_trainer_instance = MockTrainer.return_value
        mock_trainer_instance.run_sequential_training.return_value = ({'final_metric': 1.0}, [])
        mock_trainer_instance.train_and_evaluate.return_value = ({'all/lm_loss': 0.5}, [])

        engine = ScientificDiscoveryEngine(
            challenge,
            algorithms,
            default_training_config=default_training_config,
            config={"n_trials": 1, "smoke_test": True}
        )

        orchestrator = EngineOrchestrator(engine)
        results = orchestrator.run(patience_budget)
    
    assert results.challenge.name == "Sequence Duplication Challenge"
    assert len(results.algorithm_results) > 0
    assert 'HRM_optimized' in results.algorithm_results
    
    print("🎉 Full workflow test completed successfully!")

if __name__ == "__main__":
    test_full_workflow()