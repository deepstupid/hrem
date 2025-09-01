import unittest
from unittest.mock import MagicMock, patch
import yaml

from sc_engine.core.engine import ScientificDiscoveryEngine
from sc_engine.core.config_manager import ConfigManager
from dataset_manager import dataset_manager

class TestInteractiveAPI(unittest.TestCase):

    def setUp(self):
        """Set up a mock engine for testing."""
        self.config_manager = ConfigManager()
        self.challenge_configs = self.config_manager.load_challenge_configs()
        self.model_configs = self.config_manager.load_model_configs()
        self.search_spaces = self.config_manager.load_search_spaces()

        self.challenge_config = self.challenge_configs.challenges[0]

        algorithm_names = ["HRM", "HREM"]
        self.algorithm_configs = []
        for name in algorithm_names:
            model_config = self.model_configs.get(name)
            from types import SimpleNamespace
            self.algorithm_configs.append(SimpleNamespace(
                name=model_config.name,
                config=model_config.model_dump(),
                search_space=self.search_spaces.get(name)
            ))

        with open("config/demo_config.yaml", 'r') as f:
            demo_config = yaml.safe_load(f)

        with open("config/training/default.yaml", 'r') as f:
            default_training_config = yaml.safe_load(f)

        self.engine = ScientificDiscoveryEngine(
            challenge=self.challenge_config,
            algorithms=self.algorithm_configs,
            config=demo_config,
            default_training_config=default_training_config
        )

    def test_run_interactive_puzzle_structure(self):
        """Test the basic structure of the output from run_interactive_puzzle."""
        dataset_name = "synthetic-sort"
        puzzle_index = 0
        dataset_path = dataset_manager.get_dataset_path(dataset_name, smoke_test=True)

        results = self.engine.run_interactive_puzzle(
            dataset_path=dataset_path,
            puzzle_index=puzzle_index,
            algorithm_configs=self.algorithm_configs
        )

        self.assertIsInstance(results, dict)
        self.assertIn("HRM", results)
        self.assertIn("HREM", results)

        for model_results in results.values():
            self.assertIsInstance(model_results, list)
            for step_result in model_results:
                self.assertIsInstance(step_result, dict)
                self.assertIn("prediction", step_result)
                self.assertIn("correct", step_result)
                self.assertIn("metrics", step_result)

if __name__ == '__main__':
    unittest.main()
