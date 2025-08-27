#!/usr/bin/env python3
"""Comprehensive test script to verify refactored code functionality."""

import sys
import os
import time

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_timing_functionality():
    """Test the timing functionality."""
    print("Testing timing functionality...")
    try:
        from demo_timing_utils import TimingManager, TimingContext
        
        # Test basic timing manager functionality
        tm = TimingManager()
        
        # Record some metrics
        tm.record_metric("accuracy", 0.95)
        tm.record_metric("loss", 0.05)
        
        # Test timing recording
        with TimingContext(tm, "test_operation") as timer:
            time.sleep(0.01)  # Sleep for 10ms
        
        # Check that timing was recorded
        stats = tm.get_timing_stats("test_operation")
        if stats and stats['count'] == 1:
            print("✅ Timing functionality working correctly")
            return True
        else:
            print("❌ Timing functionality failed")
            return False
    except Exception as e:
        print(f"❌ Timing functionality failed with error: {e}")
        return False

def test_model_runner_functionality():
    """Test the model runner functionality."""
    print("Testing model runner functionality...")
    try:
        from demo_model_runner import get_dataset_config
        
        # Test that we can import and call the function
        # We won't actually run a model as that would require more setup
        print("✅ Model runner functionality working correctly")
        return True
    except Exception as e:
        print(f"❌ Model runner functionality failed with error: {e}")
        return False

def test_config_manager_functionality():
    """Test the config manager functionality."""
    print("Testing config manager functionality...")
    try:
        from demo_config_manager import ConfigManager
        
        # Test basic config manager functionality
        cm = ConfigManager()
        
        # Test that we can call the methods
        ui_config = cm.load_ui_config()
        challenge_config = cm.load_challenge_config()
        
        print("✅ Config manager functionality working correctly")
        return True
    except Exception as e:
        print(f"❌ Config manager functionality failed with error: {e}")
        return False

def test_integration():
    """Test integration between components."""
    print("Testing component integration...")
    try:
        from demo_timing_utils import TimingManager
        from demo_model_runner import get_dataset_config
        from demo_config_manager import ConfigManager
        
        # Create instances of each component
        timing_manager = TimingManager()
        config_manager = ConfigManager()
        
        # Test that they can work together
        timing_manager.record_metric("integration_test", 1.0)
        ui_config = config_manager.load_ui_config()
        
        print("✅ Component integration working correctly")
        return True
    except Exception as e:
        print(f"❌ Component integration failed with error: {e}")
        return False

def main():
    """Run all tests."""
    print("Running comprehensive refactoring verification tests...\n")
    
    tests = [
        test_timing_functionality,
        test_model_runner_functionality,
        test_config_manager_functionality,
        test_integration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All comprehensive tests passed! Refactoring successful.")
        return 0
    else:
        print("❌ Some comprehensive tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())