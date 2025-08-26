#!/usr/bin/env python3
"""
Test suite for the enhanced HRM/HREM demo system.
"""

import sys
import os
import unittest
from unittest.mock import patch, MagicMock

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from demo_models import ModelRegistry, model_registry, register_custom_model, get_model_config, get_model_configs, list_available_models
from demo_config import AlgorithmConfigFactory, DemoMode, get_demo_config
from demo_validation import ConfigValidator
from hrm_system.config import ModelConfig

class TestDemoModels(unittest.TestCase):
    """Test cases for the demo models module."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Reset the global registry for each test
        global model_registry
        model_registry = ModelRegistry()
    
    def test_model_registry_default_models(self):
        """Test that the model registry has default models."""
        models = model_registry.list_models()
        self.assertIn("HRM", models)
        self.assertIn("HREM", models)
        self.assertIn("EnhancedHREM", models)
    
    def test_model_registry_get_model(self):
        """Test getting a model from the registry."""
        hrm_config = model_registry.get_model("HRM")
        self.assertIsInstance(hrm_config, ModelConfig)
        self.assertEqual(hrm_config.name, "HRM")
    
    def test_model_registry_register_model(self):
        """Test registering a new model."""
        new_model = ModelConfig(
            name="TestModel",
            algorithm_class="test.algorithm.TestAlgorithm",
            base_arch_config="test_config"
        )
        model_registry.register_model("TestModel", new_model)
        
        models = model_registry.list_models()
        self.assertIn("TestModel", models)
        
        retrieved = model_registry.get_model("TestModel")
        self.assertEqual(retrieved.name, "TestModel")
    
    def test_global_functions(self):
        """Test global model registry functions."""
        # Test listing models
        models = list_available_models()
        self.assertIn("HRM", models)
        self.assertIn("HREM", models)
        
        # Test getting a model
        hrm_config = get_model_config("HRM")
        self.assertEqual(hrm_config.name, "HRM")
        
        # Test getting multiple models
        configs = get_model_configs(["HRM", "HREM"])
        self.assertEqual(len(configs), 2)
        self.assertEqual(configs[0].name, "HRM")
        self.assertEqual(configs[1].name, "HREM")

class TestDemoConfig(unittest.TestCase):
    """Test cases for the demo configuration module."""
    
    def test_algorithm_config_factory_search_space(self):
        """Test the search space path determination."""
        # Test HRM algorithm
        hrm_path = AlgorithmConfigFactory.get_search_space_path("hrm_system.algorithms.hrm.HRMAlgorithm")
        self.assertEqual(hrm_path, "config/hrm_search_space.yaml")
        
        # Test HREM algorithm
        hrem_path = AlgorithmConfigFactory.get_search_space_path("hrm_system.algorithms.hrem.HREMAlgorithm")
        self.assertEqual(hrem_path, "config/hparam_search_space.yaml")
        
        # Test EnhancedHREM algorithm
        enhanced_path = AlgorithmConfigFactory.get_search_space_path("hrm_system.algorithms.enhanced_hrem.EnhancedHREMAlgorithm")
        self.assertEqual(enhanced_path, "config/hparam_search_space.yaml")
        
        # Test unknown algorithm (should default to HREM)
        unknown_path = AlgorithmConfigFactory.get_search_space_path("unknown.algorithm.UnknownAlgorithm")
        self.assertEqual(unknown_path, "config/hparam_search_space.yaml")
    
    def test_algorithm_config_factory_base_arch_config(self):
        """Test the base architecture config determination."""
        # Test HRM algorithm
        hrm_base = AlgorithmConfigFactory._get_base_arch_config("hrm_system.algorithms.hrm.HRMAlgorithm")
        self.assertEqual(hrm_base, "hrm_v1")
        
        # Test HREM algorithm
        hrem_base = AlgorithmConfigFactory._get_base_arch_config("hrm_system.algorithms.hrem.HREMAlgorithm")
        self.assertEqual(hrem_base, "hrem_v1")
        
        # Test EnhancedHREM algorithm
        enhanced_base = AlgorithmConfigFactory._get_base_arch_config("hrm_system.algorithms.enhanced_hrem.EnhancedHREMAlgorithm")
        self.assertEqual(enhanced_base, "enhanced_hrem_v1")
        
        # Test unknown algorithm (should default to hrem_v1)
        unknown_base = AlgorithmConfigFactory._get_base_arch_config("unknown.algorithm.UnknownAlgorithm")
        self.assertEqual(unknown_base, "hrem_v1")
    
    def test_algorithm_config_factory_optimization_config(self):
        """Test creating optimization configuration."""
        # Test HRM config
        hrm_config = AlgorithmConfigFactory.create_optimization_config(
            "hrm_system.algorithms.hrm.HRMAlgorithm",
            "HRM",
            10
        )
        self.assertEqual(hrm_config["n_trials"], 10)
        self.assertEqual(hrm_config["model_to_optimize"]["name"], "HRM_best")
        self.assertEqual(hrm_config["model_to_optimize"]["base_arch_config"], "hrm_v1")
        self.assertEqual(hrm_config["search_space"]["path"], "config/hrm_search_space.yaml")
        
        # Test HREM config
        hrem_config = AlgorithmConfigFactory.create_optimization_config(
            "hrm_system.algorithms.hrem.HREMAlgorithm",
            "HREM",
            15
        )
        self.assertEqual(hrem_config["n_trials"], 15)
        self.assertEqual(hrem_config["model_to_optimize"]["name"], "HREM_best")
        self.assertEqual(hrem_config["model_to_optimize"]["base_arch_config"], "hrem_v1")
        self.assertEqual(hrem_config["search_space"]["path"], "config/hparam_search_space.yaml")
    
    def test_demo_config_modes(self):
        """Test demo configuration modes."""
        # Test fast mode
        fast_config = get_demo_config(DemoMode.FAST)
        self.assertEqual(fast_config.mode, DemoMode.FAST)
        self.assertEqual(fast_config.experiment_settings.opt_trials, 3)
        
        # Test full mode
        full_config = get_demo_config(DemoMode.FULL)
        self.assertEqual(full_config.mode, DemoMode.FULL)
        self.assertEqual(full_config.experiment_settings.opt_trials, 20)
        
        # Test interactive mode
        interactive_config = get_demo_config(DemoMode.INTERACTIVE, interactive=True)
        self.assertEqual(interactive_config.mode, DemoMode.INTERACTIVE)
        self.assertTrue(interactive_config.interactive)

class TestDemoValidation(unittest.TestCase):
    """Test cases for the demo validation module."""
    
    def test_config_validator_model_names(self):
        """Test model name validation."""
        available = ["HRM", "HREM", "EnhancedHREM"]
        requested = ["HRM", "UnknownModel", "HREM"]
        
        valid = ConfigValidator.validate_model_names(requested, available)
        self.assertEqual(valid, ["HRM", "HREM"])
    
    def test_config_validator_model_config(self):
        """Test model configuration validation."""
        # Valid config
        valid_config = ModelConfig(
            name="TestModel",
            algorithm_class="test.algorithm.TestAlgorithm",
            base_arch_config="test_config"
        )
        self.assertTrue(ConfigValidator.validate_model_config(valid_config))
        
        # Invalid config - missing name
        invalid_config = ModelConfig(
            name="",
            algorithm_class="test.algorithm.TestAlgorithm",
            base_arch_config="test_config"
        )
        self.assertFalse(ConfigValidator.validate_model_config(invalid_config))
    
    def test_config_validator_model_configs(self):
        """Test model configurations validation."""
        # Valid configs
        valid_configs = [
            ModelConfig(name="Model1", algorithm_class="test.Test1", base_arch_config="config1"),
            ModelConfig(name="Model2", algorithm_class="test.Test2", base_arch_config="config2")
        ]
        self.assertTrue(ConfigValidator.validate_model_configs(valid_configs))
        
        # Invalid configs - duplicate names
        invalid_configs = [
            ModelConfig(name="Model1", algorithm_class="test.Test1", base_arch_config="config1"),
            ModelConfig(name="Model1", algorithm_class="test.Test2", base_arch_config="config2")
        ]
        self.assertFalse(ConfigValidator.validate_model_configs(invalid_configs))
        
        # Invalid configs - empty list
        self.assertFalse(ConfigValidator.validate_model_configs([]))
    
    def test_config_validator_challenge_key(self):
        """Test challenge key validation."""
        available = ["Copy Task", "Reverse Task", "ARC Challenge"]
        
        # Valid exact match
        self.assertTrue(ConfigValidator.validate_challenge_key("Copy Task", available))
        
        # Valid partial match
        self.assertTrue(ConfigValidator.validate_challenge_key("copy", available))
        
        # Invalid key
        self.assertFalse(ConfigValidator.validate_challenge_key("Unknown Challenge", available))

if __name__ == "__main__":
    unittest.main()