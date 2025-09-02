import unittest
from unittest.mock import MagicMock, patch
import yaml

from sc_engine.core.model_runner import ScientificModelRunner
from dataset_manager import dataset_manager

class TestInteractiveAPI(unittest.TestCase):

    def setUp(self):
        """Set up a mock engine for testing."""
        self.runner = ScientificModelRunner()

    def test_run_interactive_puzzle_structure(self):
        """Test the basic structure of the output from run_interactive_puzzle."""
        dataset_name = "synthetic-sort"
        puzzle_index = 0
        dataset_path = dataset_manager.get_dataset_path(dataset_name, smoke_test=True)

        results = self.runner.run(
            run_type="interactive",
            dataset_path=dataset_path,
            puzzle_index=puzzle_index,
            challenge_id="synthetic_sort",
            models=["HRM", "HREM"]
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
