"""Comprehensive test script to verify refactored code functionality."""

import time
from demo_timing_utils import TimingManager
from demo_config_manager import ConfigManager, DemoConfig

def test_timing_functionality():
    """Test the timing functionality."""
    tm = TimingManager()

    # Record some metrics
    tm.record_metric("accuracy", 0.95)
    tm.record_metric("loss", 0.05)

    # Test timing recording
    with tm.get_context("test_operation"):
        time.sleep(0.01)  # Sleep for 10ms

    # Check that timing was recorded
    stats = tm.get_timing_stats("test_operation")
    assert stats and stats['count'] == 1

def test_config_manager_functionality():
    """Test the config manager functionality."""
    # Test that we can create a config
    config = ConfigManager.get_adaptive_config()
    assert isinstance(config, DemoConfig)

def test_integration():
    """Test integration between components."""
    # Create instances of each component
    timing_manager = TimingManager()
    
    # Test that they can work together
    timing_manager.record_metric("integration_test", 1.0)
    config = ConfigManager.get_adaptive_config()
    
    assert "integration_test" in timing_manager.metrics
    assert config.patience_level == "low"

if __name__ == "__main__":
    test_timing_functionality()
    test_config_manager_functionality()
    test_integration()
    print("All comprehensive tests passed!")
