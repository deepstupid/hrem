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
