"""Test script to verify the unified demo runner functionality."""

from demo_config import DemoConfig, DemoMode, DemoConfigManager
from unified_demo_runner import run_demo

def test_lightning_demo():
    """Test the lightning demo mode."""
    # Create a minimal configuration for testing
    config = DemoConfigManager.create_config(
        mode=DemoMode.LIGHTNING,
        models=["HRM", "HREM"],
        dataset="synthetic",
        task="copy",
        smoke_test=True,
        export_metrics=False
    )
    
    # Override loop control for faster testing
    config.loop_control.max_epochs = 1
    config.loop_control.max_trials = 1
    
    # Run the demo
    run_demo(config)

def test_demo_config():
    """Test the demo configuration functionality."""
    # Test creating configs for different modes
    for mode in [DemoMode.LIGHTNING, DemoMode.ADAPTIVE, DemoMode.COMPREHENSIVE]:
        config = DemoConfigManager.create_config(mode=mode)
        assert config.demo_mode == mode