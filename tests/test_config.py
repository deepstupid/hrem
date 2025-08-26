import pytest
from hrm_system import ExperimentConfig, EvaluationConfig, OptimizationConfig

def test_default_evaluation_config():
    """
    Tests that a default ExperimentConfig is set to evaluation mode
    and has the correct sub-configurations.
    """
    config = ExperimentConfig()
    assert config.mode == "evaluate"
    assert isinstance(config.evaluation_config, EvaluationConfig)
    assert config.optimization_config is None
    assert config.evaluation_config.model_a is None
    assert config.evaluation_config.model_b is None

def test_optimization_config_creation():
    """
    Tests that setting the mode to 'optimize' correctly creates
    the OptimizationConfig.
    """
    config = ExperimentConfig(mode="optimize")
    assert config.mode == "optimize"
    assert isinstance(config.optimization_config, OptimizationConfig)
    assert config.evaluation_config is None # Should not be created
    assert config.optimization_config.baseline_model.name == "HRM"
    assert config.optimization_config.model_to_optimize.name == "HREM_best"

def test_smoke_test_flag():
    """
    Tests the smoke_test flag in the run_config.
    """
    config = ExperimentConfig()
    assert config.run_config.smoke_test is False

    config_smoke = ExperimentConfig(run_config={"smoke_test": True})
    assert config_smoke.run_config.smoke_test is True

def test_config_validation_error():
    """
    Tests that Pydantic raises a validation error for invalid data.
    """
    with pytest.raises(ValueError):
        # 'invalid_mode' is not a valid literal for the mode
        ExperimentConfig(mode="invalid_mode")

    with pytest.raises(ValueError):
        # 'invalid_dataset' is not a valid literal for the dataset
        ExperimentConfig(data_config={"dataset": "invalid_dataset"})
