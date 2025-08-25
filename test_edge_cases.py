#!/usr/bin/env python3
"""
Edge case tests for the parameterized CLI demo.
"""

import subprocess
import sys
import os

def test_edge_cases():
    """Test edge cases for the parameterized demo."""
    
    print("Testing edge cases for the parameterized demo...")
    
    # Test 1: Invalid challenge key
    print("\nTest 1: Invalid challenge key")
    try:
        result = subprocess.run([
            sys.executable, "run_demo_cli_parameterized.py", "--challenge-key", "NonExistentChallenge"
        ], capture_output=True, text=True, timeout=10)
        
        if "not found" in result.stdout or result.returncode != 0:
            print("✅ Correctly handled invalid challenge key")
        else:
            print("❌ Did not handle invalid challenge key properly")
            print(f"Output: {result.stdout}")
    except Exception as e:
        print(f"❌ Failed to test invalid challenge key: {e}")
    
    # Test 2: Invalid model name
    print("\nTest 2: Invalid model name")
    try:
        # We'll test the get_model_configs function directly
        test_script = '''
from run_demo_cli_parameterized import get_model_configs

# Test with invalid model name
configs = get_model_configs(["HRM", "InvalidModel", "HREM"])
model_names = [config.name for config in configs]

print("Models with invalid entry:", model_names)
if "InvalidModel" not in model_names and "HRM" in model_names and "HREM" in model_names:
    print("✅ Correctly handled invalid model name")
else:
    print("❌ Did not handle invalid model name properly")
'''
        
        with open("temp_test.py", "w") as f:
            f.write(test_script)
            
        result = subprocess.run([
            sys.executable, "temp_test.py"
        ], capture_output=True, text=True, timeout=10)
        
        print(result.stdout.strip())
        if "✅ Correctly handled invalid model name" in result.stdout:
            print("✅ Invalid model name handling works correctly")
        else:
            print("❌ Invalid model name handling failed")
            print(f"Output: {result.stdout}")
            
        # Clean up
        os.remove("temp_test.py")
        
    except Exception as e:
        print(f"❌ Failed to test invalid model name: {e}")
        # Clean up if needed
        if os.path.exists("temp_test.py"):
            os.remove("temp_test.py")
    
    # Test 3: Empty model list
    print("\nTest 3: Empty model list")
    try:
        # We'll test the get_model_configs function directly
        test_script = '''
from run_demo_cli_parameterized import get_model_configs

# Test with empty model list
configs = get_model_configs([])
model_names = [config.name for config in configs]

print("Models with empty list:", model_names)
if len(model_names) == 0:
    print("✅ Correctly handled empty model list")
else:
    print("❌ Did not handle empty model list properly")
'''
        
        with open("temp_test.py", "w") as f:
            f.write(test_script)
            
        result = subprocess.run([
            sys.executable, "temp_test.py"
        ], capture_output=True, text=True, timeout=10)
        
        print(result.stdout.strip())
        if "✅ Correctly handled empty model list" in result.stdout:
            print("✅ Empty model list handling works correctly")
        else:
            print("❌ Empty model list handling failed")
            print(f"Output: {result.stdout}")
            
        # Clean up
        os.remove("temp_test.py")
        
    except Exception as e:
        print(f"❌ Failed to test empty model list: {e}")
        # Clean up if needed
        if os.path.exists("temp_test.py"):
            os.remove("temp_test.py")
    
    print("\n✅ Edge case testing completed!")

if __name__ == "__main__":
    # Change to the project directory
    os.chdir("/home/me/hrem2")
    
    test_edge_cases()