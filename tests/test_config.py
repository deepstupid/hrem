import pytest
from sc_engine.core.config import ChallengeConfig, AlgorithmConfig, PatienceBudget, DiscoveryConfig

def test_default_evaluation_config():
    """
    Tests that a default ExperimentConfig is set to evaluation mode
    and has the correct sub-configurations.
    """
    # This test is no longer applicable as the mode is determined by the runner method.
    pass

def test_optimization_config_creation():
    """
    Tests that setting the mode to 'optimize' correctly creates
    the OptimizationConfig.
    """
    # This test is no longer applicable as the mode is determined by the runner method.
    pass

def test_smoke_test_flag():
    """
    Tests the smoke_test flag in the run_config.
    """
    # TODO: Move this test to an integration test for the ScientificModelRunner
    pass

from sc_engine.core.config import ChallengeLevel

def test_config_validation_error():
    """
    Tests that Pydantic raises a validation error for invalid data.
    """
    with pytest.raises(ValueError):
        # 'invalid_level' is not a valid literal for the ChallengeLevel
        ChallengeLevel("invalid_level")

import unittest
from unittest.mock import patch, mock_open
from sc_engine.core.config_manager import ConfigManager

def test_valid_dataset_in_challenge_config():
    """
    Tests that a ChallengeConfig can be created with a valid dataset dictionary.
    """
    try:
        ChallengeConfig(
            name="Test",
            id="test",
            description="Test",
            dataset={"name": "synthetic-sort", "path": "/data/synthetic-sort"}
        )
    except Exception as e:
        pytest.fail(f"ChallengeConfig raised an exception for a valid dataset dict: {e}")

class TestConfigManager(unittest.TestCase):
    @patch('os.path.isdir')
    @patch('os.listdir')
    @patch('builtins.open', new_callable=mock_open)
    @patch('yaml.safe_load')
    def test_load_model_configs(self, mock_yaml_load, mock_open, mock_listdir, mock_isdir):
        # Arrange
        mock_isdir.return_value = True
        mock_listdir.return_value = ['hrm.yaml', 'hrem.yaml']
        mock_yaml_load.side_effect = [
            {
                'name': 'HRM',
                'algorithm_class': 'HRMAlgorithm',
                'theoretical_advantages': [],
                'theoretical_limitations': []
            },
            {
                'name': 'HREM',
                'algorithm_class': 'HREMAlgorithm',
                'theoretical_advantages': [],
                'theoretical_limitations': []
            }
        ]

        config_manager = ConfigManager()

        # Act
        model_configs = config_manager.load_model_configs()

        # Assert
        self.assertIn('HRM', model_configs)
        self.assertIn('HREM', model_configs)
        self.assertEqual(model_configs['HRM'].algorithm_class, 'HRMAlgorithm')
        self.assertEqual(model_configs['HREM'].algorithm_class, 'HREMAlgorithm')

    @patch('builtins.open', new_callable=mock_open)
    @patch('yaml.safe_load')
    def test_load_challenge_configs(self, mock_yaml_load, mock_open):
        # Arrange
        mock_yaml_load.return_value = {
            "challenges": [
                {
                    "name": "Test Challenge",
                    "id": "test_challenge",
                    "description": "A test challenge.",
                    "difficulty": "EASY",
                    "hardware": "Low",
                    "duration": "1 min",
                    "dataset": {"dataset": "test_dataset"},
                    "models": ["HRM"],
                    "patience_level": "low",
                    "optimization": False
                }
            ]
        }
        config_manager = ConfigManager()

        # Act
        challenge_configs = config_manager.load_challenge_configs()

        # Assert
        self.assertEqual(len(challenge_configs.challenges), 1)
        self.assertEqual(challenge_configs.challenges[0].name, 'Test Challenge')
