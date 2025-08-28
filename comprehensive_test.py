"""Comprehensive test script to verify refactored code functionality."""

import time
from demo_timing_utils import EnhancedTimingManager, EnhancedTimingContext
from demo_model_runner import get_dataset_config
from demo_config_manager import EnhancedConfigManager

def test_timing_functionality():
    """Test the timing functionality."""
    tm = EnhancedTimingManager()

    # Record some metrics
    tm.record_metric("accuracy", 0.95)
    tm.record_metric("loss", 0.05)

    # Test timing recording
    with EnhancedTimingContext(tm, "test_operation") as timer:
        time.sleep(0.01)  # Sleep for 10ms

    # Check that timing was recorded
    stats = tm.get_timing_stats("test_operation")
    assert stats and stats['count'] == 1

def test_model_runner_functionality():
    """Test the model runner functionality."""
    # This test just checks if the function can be imported and called.
    get_dataset_config("synthetic", smoke_test=True)

def test_config_manager_functionality():
    """Test the config manager functionality."""
    # Test that we can call the methods
    ui_config = EnhancedConfigManager.load_ui_config()
    challenge_config = EnhancedConfigManager.load_challenge_config()
    assert isinstance(ui_config, dict)
    assert isinstance(challenge_config, list)

def test_integration():
    """Test integration between components."""
    # Create instances of each component
    timing_manager = EnhancedTimingManager()
    
    # Test that they can work together
    timing_manager.record_metric("integration_test", 1.0)
    ui_config = EnhancedConfigManager.load_ui_config()
    
    assert "integration_test" in timing_manager.metrics
    assert "default_models" in ui_config