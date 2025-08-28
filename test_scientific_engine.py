import unittest
from unittest.mock import patch, MagicMock, call
import sys
import os

# Add the project root to the path so we can import our modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from scientific_comparison.config import ChallengeConfig, AlgorithmConfig, PatienceBudget, ChallengeLevel
from scientific_comparison.engine import ScientificDiscoveryEngine
from hrm_system.config import DataConfig, ModelConfig

class TestScientificDiscoveryEngine(unittest.TestCase):
    """Test suite for the ScientificDiscoveryEngine."""

    def setUp(self):
        """Set up a test instance of the engine."""
        self.challenge = ChallengeConfig(
            name="Test Challenge",
            id="test_challenge",
            description="A test challenge for validation",
            dataset=DataConfig(dataset="synthetic"),
            difficulty=ChallengeLevel.BEGINNER,
            scientific_question="How do HRM and HREM compare on synthetic data?",
            hypothesis_space=["HREM will perform better"]
        )

        self.hrm_config = AlgorithmConfig(
            name="HRM",
            algorithm_class="hrm_system.algorithms.hrm.HRMAlgorithm",
            search_space={'learning_rate': [0.001, 0.01]},
            theoretical_advantages=[],
            theoretical_limitations=[]
        )

        self.hrem_config = AlgorithmConfig(
            name="HREM",
            algorithm_class="hrm_system.algorithms.hrem.HREMAlgorithm",
            search_space={'embedding_dim': [32, 64]},
            theoretical_advantages=[],
            theoretical_limitations=[]
        )

        self.algorithms = [self.hrm_config, self.hrem_config]
        self.patience_budget = PatienceBudget(level="low")

        self.engine = ScientificDiscoveryEngine(
            challenge=self.challenge,
            algorithms=self.algorithms,
            smoke_test=True
        )

    @patch('scientific_comparison.engine.run_optimization')
    @patch('scientific_comparison.engine.run_model_with_fallback')
    @patch('scientific_comparison.engine.get_model_config')
    @patch('scientific_comparison.engine.get_model_search_space')
    def test_full_discovery_session_workflow(self, mock_get_search_space, mock_get_model_config, mock_run_model, mock_run_optimization):
        """Test the full workflow of a discovery session, ensuring optimized parameters are used."""
        print("🧪 Testing full discovery session workflow...")

        # --- Mock Setup ---

        # Mock model and search space getters
        mock_get_model_config.side_effect = lambda name: ModelConfig(name=name, algorithm_class=f"some.{name}")
        mock_get_search_space.side_effect = lambda name: {'lr': [0.1, 0.01]} if name == "HRM" else {'dim': [16, 32]}

        # Mock for run_model_with_fallback
        # Returns a simple metric dictionary
        mock_run_model.return_value = {'all/lm_loss': 0.5, 'timing': 1.0}

        # Mock for run_optimization
        # Returns a dictionary with best_params
        mock_run_optimization.side_effect = [
            {'best_params': {'learning_rate': 0.01}},  # For HRM
            {'best_params': {'embedding_dim': 64}},    # For HREM
        ]

        # --- Execute ---

        results = self.engine.execute_discovery_session(self.patience_budget)

        # --- Assertions ---

        # 1. Check if the session completed
        self.assertIsNotNone(results)
        self.assertIn("HRM_optimized", results.algorithm_results)
        self.assertIn("HREM_optimized", results.algorithm_results)

        # 2. Verify that run_optimization was called for each algorithm
        self.assertEqual(mock_run_optimization.call_count, 2)

        # 3. Verify that run_model_with_fallback was called for baseline and final evaluations
        # 2 baseline runs + 2 final baseline runs + 2 final optimized runs = 6 calls
        self.assertEqual(mock_run_model.call_count, 6)

        # 4. Crucially, verify that the optimized run for HREM used the correct hyperparameters
        # We need to inspect the calls to mock_run_model

        # Get the keyword arguments of the call for the optimized HREM run
        optimized_hrem_call = None
        for c in mock_run_model.call_args_list:
            kwargs = c.kwargs
            if kwargs.get('run_identifier') == 'HREM_final_optimized':
                optimized_hrem_call = c
                break

        self.assertIsNotNone(optimized_hrem_call, "Optimized run for HREM was not called.")

        # Check that the model_config passed to the optimized run has the correct overrides
        optimized_model_config = optimized_hrem_call.kwargs['model_config']
        self.assertEqual(optimized_model_config.arch_overrides, {'embedding_dim': 64})

        # 5. Verify the same for HRM
        optimized_hrm_call = None
        for c in mock_run_model.call_args_list:
            kwargs = c.kwargs
            if kwargs.get('run_identifier') == 'HRM_final_optimized':
                optimized_hrm_call = c
                break

        self.assertIsNotNone(optimized_hrm_call, "Optimized run for HRM was not called.")

        optimized_model_config_hrm = optimized_hrm_call.kwargs['model_config']
        self.assertEqual(optimized_model_config_hrm.arch_overrides, {'learning_rate': 0.01})

        print("🎉 Test completed successfully!")


if __name__ == "__main__":
    unittest.main()
