#!/usr/bin/env python3
"""Test script to verify the unified demo runner functionality."""

import sys
import os

# Add the current directory to the path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from demo_config import DemoConfig, DemoMode, DemoConfigManager
from unified_demo_runner import run_demo

def test_lightning_demo():
    """Test the lightning demo mode."""
    print("Testing Lightning Demo Mode...")
    
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
    
    print(f"Demo Config: {config}")
    
    # Run the demo
    try:
        run_demo(config)
        print("Lightning demo test completed successfully!")
        return True
    except Exception as e:
        print(f"Error running lightning demo: {e}")
        return False

def test_demo_config():
    """Test the demo configuration functionality."""
    print("Testing Demo Configuration...")
    
    # Test creating configs for different modes
    for mode in [DemoMode.LIGHTNING, DemoMode.ADAPTIVE, DemoMode.COMPREHENSIVE]:
        config = DemoConfigManager.create_config(mode=mode)
        print(f"{mode.value} mode config: {config.demo_mode}")
    
    print("Demo configuration test completed!")
    return True

if __name__ == "__main__":
    print("Running Unified Demo Runner Tests...")
    
    # Run tests
    success = True
    success &= test_demo_config()
    # Note: We're not running the full demo test as it requires the full HRM system
    
    if success:
        print("\nAll tests passed!")
    else:
        print("\nSome tests failed!")
        sys.exit(1)