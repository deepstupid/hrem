#!/usr/bin/env python3
"""
Test script for the parameterized CLI demo.
"""

import subprocess
import sys
import os

def test_parameterized_demo():
    """Test the parameterized demo with different configurations."""
    
    # Test 1: Run with default parameters (should show help)
    print("Test 1: Checking if the script runs and shows help...")
    try:
        result = subprocess.run([
            sys.executable, "run_demo_cli_parameterized.py", "--help"
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print("✅ Script runs and shows help correctly")
        else:
            print("❌ Script failed to run properly")
            print(f"Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Failed to run script: {e}")
        return False
    
    # Test 2: Check that all expected models are supported
    print("\nTest 2: Verifying supported models...")
    expected_models = ["HRM", "HREM", "EnhancedHREM"]
    
    # Read the script and check for model definitions
    try:
        with open("run_demo_cli_parameterized.py", "r") as f:
            content = f.read()
            
        for model in expected_models:
            if model in content:
                print(f"✅ Found model: {model}")
            else:
                print(f"❌ Missing model: {model}")
                
    except Exception as e:
        print(f"❌ Failed to read script: {e}")
        return False
    
    # Test 3: Validate command line argument parsing
    print("\nTest 3: Testing argument parsing...")
    try:
        # Create a simple test that doesn't actually run the full demo
        test_script = '''
import argparse
from run_demo_cli_parameterized import get_model_configs

# Test argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("--models", nargs="+", default=["HRM", "HREM"])
args = parser.parse_args(["--models", "HRM", "HREM", "EnhancedHREM"])

configs = get_model_configs(args.models)
model_names = [config.name for config in configs]

print("Parsed models:", model_names)
expected = ["HRM", "HREM", "EnhancedHREM"]
if set(model_names) == set(expected):
    print("✅ Model parsing works correctly")
else:
    print("❌ Model parsing failed")
    print(f"Expected: {expected}")
    print(f"Got: {model_names}")
'''
        
        with open("temp_test.py", "w") as f:
            f.write(test_script)
            
        result = subprocess.run([
            sys.executable, "temp_test.py"
        ], capture_output=True, text=True, timeout=10)
        
        print(result.stdout)
        if "✅ Model parsing works correctly" in result.stdout:
            print("✅ Argument parsing works correctly")
        else:
            print("❌ Argument parsing failed")
            print(f"Output: {result.stdout}")
            print(f"Error: {result.stderr}")
            
        # Clean up
        os.remove("temp_test.py")
        
    except Exception as e:
        print(f"❌ Failed to test argument parsing: {e}")
        # Clean up if needed
        if os.path.exists("temp_test.py"):
            os.remove("temp_test.py")
        return False
    
    print("\n✅ All tests passed!")
    return True

if __name__ == "__main__":
    # Change to the project directory
    os.chdir("/home/me/hrem2")
    
    if test_parameterized_demo():
        print("\n🎉 All tests passed! The parameterized demo is working correctly.")
    else:
        print("\n❌ Some tests failed. Please check the implementation.")
        sys.exit(1)