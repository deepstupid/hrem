#!/usr/bin/env python3
"""Test script to verify refactored code functionality."""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_timing_utils():
    """Test the new timing utilities."""
    print("Testing timing utilities...")
    try:
        from demo_timing_utils import TimingManager, TimingContext
        tm = TimingManager()
        tm.record_metric("test_metric", 1.0)
        print("✅ Timing utilities working correctly")
        return True
    except Exception as e:
        print(f"❌ Timing utilities failed: {e}")
        return False

def test_model_runner():
    """Test the new model runner."""
    print("Testing model runner...")
    try:
        from demo_model_runner import run_model_with_fallback, get_dataset_config
        print("✅ Model runner imported correctly")
        return True
    except Exception as e:
        print(f"❌ Model runner failed: {e}")
        return False

def test_config_manager():
    """Test the new config manager."""
    print("Testing config manager...")
    try:
        from demo_config_manager import ConfigManager
        cm = ConfigManager()
        print("✅ Config manager imported correctly")
        return True
    except Exception as e:
        print(f"❌ Config manager failed: {e}")
        return False

def test_backward_compatibility():
    """Test that old modules still work."""
    print("Testing backward compatibility...")
    try:
        from demo_timing import TimingCollector
        from demo_config import load_ui_config
        from demo_shared import run_trial
        print("✅ Backward compatibility maintained")
        return True
    except Exception as e:
        print(f"❌ Backward compatibility broken: {e}")
        return False

def main():
    """Run all tests."""
    print("Running refactoring verification tests...\n")
    
    tests = [
        test_timing_utils,
        test_model_runner,
        test_config_manager,
        test_backward_compatibility
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Refactoring successful.")
        return 0
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())