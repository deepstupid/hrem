import unittest
from unittest.mock import MagicMock, patch
from sc_engine.core.engine import ScientificDiscoveryEngine
from sc_engine.core.config import ChallengeConfig, AlgorithmConfig, PatienceBudget, ChallengeLevel
from sc_engine.core.optimization import HyperparameterOptimizer

class TestScientificDiscoveryEngine(unittest.TestCase):
    def setUp(self):
        self.challenge_config = ChallengeConfig(
            name="Test Challenge",
            id="test_challenge",
            description="A test challenge",
            dataset={},
            difficulty=ChallengeLevel.BEGINNER
        )
        self.algorithm_config = AlgorithmConfig(
            name="Test Algorithm",
            algorithm_class="sc_engine.plugins.algorithms.hrm.HRMAlgorithm",
            search_space={"lr": {"type": "float", "low": 1e-5, "high": 1e-3}},
            config={},
            theoretical_advantages=[],
            theoretical_limitations=[]
        )
        self.patience_budget = PatienceBudget(level="medium")

    @patch('sc_engine.core.engine.Trainer')
    def test_run_hyperparameter_optimization(self, MockTrainer):
        # Arrange
        mock_trainer_instance = MockTrainer.return_value
        mock_trainer_instance.train_and_evaluate.return_value = {'all/lm_loss': 0.1}

        engine = ScientificDiscoveryEngine(
            challenge=self.challenge_config,
            algorithms=[self.algorithm_config],
            config={"n_trials": 5}
        )
        engine.optimizer = MagicMock(spec=HyperparameterOptimizer)
        engine.patience_manager = MagicMock()

        # Act
        engine._run_hyperparameter_optimization(baseline_results={})

        # Assert
        engine.optimizer.optimize.assert_called_once()
        self.assertEqual(engine.optimizer.optimize.call_args[1]['n_trials'], 5)
        self.assertIn('objective', engine.optimizer.optimize.call_args[1])
        self.assertEqual(engine.optimizer.optimize.call_args[1]['search_space'], self.algorithm_config.search_space)

    @patch('sc_engine.core.engine.Trainer')
    def test_run_evaluation(self, MockTrainer):
        # Arrange
        mock_trainer_instance = MockTrainer.return_value
        mock_trainer_instance.train_and_evaluate.return_value = {'accuracy': 0.9}

        engine = ScientificDiscoveryEngine(
            challenge=self.challenge_config,
            algorithms=[self.algorithm_config]
        )
        engine.patience_manager = MagicMock()
        engine.timing_manager = MagicMock()

        # Act
        results = engine._run_evaluation(
            title="Test Evaluation",
            phase=MagicMock(),
            patience_allocation=0.5,
            run_suffix="test",
            model_config_fn=lambda alg: alg.config,
            result_key_fn=lambda alg: alg.name,
            timing_value=0
        )

        # Assert
        MockTrainer.assert_called_once()
        mock_trainer_instance.train_and_evaluate.assert_called_once()
        self.assertIn("Test Algorithm", results)
        self.assertEqual(results["Test Algorithm"]['accuracy'], 0.9)


if __name__ == '__main__':
    unittest.main()
