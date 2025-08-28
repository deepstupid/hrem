"""Comprehensive test script to verify refactored code functionality."""

import time
from demo_metrics import MetricsManager
from demo_config_manager import ConfigManager, DemoConfig

def test_metrics_functionality():
    """Test the metrics functionality."""
    mm = MetricsManager()

    # Record some metrics
    mm.record_metric("accuracy", 0.95)
    mm.record_metric("loss", 0.05)

    # Test timing recording
    with mm.get_context("test_operation"):
        time.sleep(0.01)  # Sleep for 10ms

    # Check that timing was recorded
    stats = mm.get_timing_stats("test_operation")
    assert stats and stats['count'] == 1

    # Check that metrics were recorded
    assert "accuracy" in mm.metrics
    assert "loss" in mm.metrics


def test_config_manager_functionality():
    """Test the config manager functionality."""
    # Test that we can create a config
    config = ConfigManager.get_adaptive_config()
    assert isinstance(config, DemoConfig)

def test_integration():
    """Test integration between components."""
    # Create instances of each component
    metrics_manager = MetricsManager()
    
    # Test that they can work together
    metrics_manager.record_metric("integration_test", 1.0)
    config = ConfigManager.get_adaptive_config()
    
    assert "integration_test" in metrics_manager.metrics
    assert config.patience_level == "low"

if __name__ == "__main__":
    test_metrics_functionality()
    test_config_manager_functionality()
    test_integration()
    print("All comprehensive tests passed!")
