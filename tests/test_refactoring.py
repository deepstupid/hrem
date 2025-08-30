import unittest
from unittest.mock import MagicMock
import torch
from sc_engine.plugins.algorithms.hrem import HREMAlgorithm
from sc_engine.core.insight_generator import ScientificInsightGenerator
import itertools
import yaml

class RefactoringTests(unittest.TestCase):

    def test_hrem_optimizer_selection(self):
        """
        Test that the refactored HREMAlgorithm correctly selects the optimizer.
        """
        # Mock metadata
        train_metadata = MagicMock()
        train_metadata.vocab_size = 10
        train_metadata.seq_len = 10
        train_metadata.num_puzzle_identifiers = 1
        train_metadata.total_groups = 1
        train_metadata.mean_puzzle_examples = 1

        # Load base architecture config
        with open("config/arch/hrem_v1.yaml", 'r') as f:
            base_arch_config = yaml.safe_load(f)

        # Test with AdamW
        training_config_adamw = {"optimizer": "AdamW", "optimizer_eps": 1e-5, "puzzle_emb_weight_decay": 1.0, "weight_decay": 1.0, "beta1": 0.9, "beta2": 0.95, "puzzle_emb_lr": 1e-4, "lr": 1e-4, "epochs": 1, "global_batch_size": 32}
        model_config_adamw = base_arch_config.copy()
        model_config_adamw['algorithm_class'] = "sc_engine.plugins.algorithms.hrem.HREMAlgorithm"


        hrem_algo_adamw = HREMAlgorithm(model_config_adamw, training_config_adamw)
        hrem_algo_adamw.initialize_train_state(train_metadata, world_size=1, rank=0)

        self.assertIsInstance(hrem_algo_adamw.train_state.optimizers[1], torch.optim.AdamW)
        self.assertEqual(hrem_algo_adamw.train_state.optimizers[1].defaults['eps'], 1e-5)

        # Test with Adam
        training_config_adam = {"optimizer": "Adam", "optimizer_eps": 1e-8, "puzzle_emb_weight_decay": 1.0, "weight_decay": 1.0, "beta1": 0.9, "beta2": 0.95, "puzzle_emb_lr": 1e-4, "lr": 1e-4, "epochs": 1, "global_batch_size": 32}
        model_config_adam = base_arch_config.copy()
        model_config_adam['algorithm_class'] = "sc_engine.plugins.algorithms.hrem.HREMAlgorithm"

        hrem_algo_adam = HREMAlgorithm(model_config_adam, training_config_adam)
        hrem_algo_adam.initialize_train_state(train_metadata, world_size=1, rank=0)

        self.assertIsInstance(hrem_algo_adam.train_state.optimizers[1], torch.optim.Adam)
        self.assertEqual(hrem_algo_adam.train_state.optimizers[1].defaults['eps'], 1e-8)

    def test_statistical_significance(self):
        """
        Test the new statistical significance calculation.
        """
        raw_results = {
            "ModelA": [
                {"all/accuracy": 0.85},
                {"all/accuracy": 0.88},
                {"all/accuracy": 0.90},
                {"all/accuracy": 0.86},
                {"all/accuracy": 0.89},
            ],
            "ModelB": [
                {"all/accuracy": 0.75},
                {"all/accuracy": 0.78},
                {"all/accuracy": 0.80},
                {"all/accuracy": 0.76},
                {"all/accuracy": 0.79},
            ],
            "ModelC": [
                {"all/accuracy": 0.84},
                {"all/accuracy": 0.87},
                {"all/accuracy": 0.89},
                {"all/accuracy": 0.85},
                {"all/accuracy": 0.88},
            ]
        }
        metric='all/accuracy'

        # Helper to extract metric data
        def get_metric_data(results, model_name, metric_key):
            return [run[metric_key] for run in results[model_name]]

        model_names = list(raw_results.keys())
        p_values = {name: {} for name in model_names}

        # This part is a bit of a hack. The ScientificInsightGenerator is not a static class.
        # I will instantiate it with dummy data.
        from sc_engine.core.config import ChallengeConfig, AlgorithmConfig, ChallengeLevel
        challenge_config = ChallengeConfig(name="test", id="test", description="test", dataset={}, difficulty=ChallengeLevel.BEGINNER)
        algo_configs = [AlgorithmConfig(name=name, algorithm_class="", config={}, theoretical_advantages=[], theoretical_limitations=[], search_space={}) for name in model_names]
        insight_generator = ScientificInsightGenerator(challenge_config, algo_configs)


        for model1, model2 in itertools.combinations(model_names, 2):
            data1 = get_metric_data(raw_results, model1, metric)
            data2 = get_metric_data(raw_results, model2, metric)

            p_value = insight_generator.calculate_statistical_significance(data1, data2)
            p_values[model1][model2] = p_value
            p_values[model2][model1] = p_value


        # Test ModelA vs ModelB (should be significant)
        self.assertIn("ModelB", p_values["ModelA"])
        self.assertLess(p_values["ModelA"]["ModelB"], 0.05)

        # Test ModelA vs ModelC (should not be significant)
        self.assertIn("ModelC", p_values["ModelA"])
        self.assertGreater(p_values["ModelA"]["ModelC"], 0.05)

if __name__ == "__main__":
    unittest.main()
