#!/usr/bin/env python3
"""
Example usage scripts for the parameterized CLI demo.
"""

import subprocess
import sys

def run_example_commands():
    """Run example commands to demonstrate the parameterized demo."""
    
    examples = [
        {
            "name": "Basic usage with default models",
            "command": ["python", "run_demo_cli_parameterized.py", "--help"]
        },
        {
            "name": "Run with specific challenge and models",
            "command": ["python", "run_demo_cli_parameterized.py", "--challenge-key", "Copy", "--models", "HRM", "HREM"]
        },
        {
            "name": "Run with all models in fast mode",
            "command": ["python", "run_demo_cli_parameterized.py", "--fast", "--challenge-key", "Reverse", "--models", "HRM", "HREM", "EnhancedHREM"]
        }
    ]
    
    for example in examples:
        print(f"\n{'='*60}")
        print(f"Example: {example['name']}")
        print(f"Command: {' '.join(example['command'])}")
        print('='*60)
        
        # For demonstration purposes, we'll just show the command
        # In a real scenario, you might want to actually run these
        print("(This is a demonstration - commands are not executed in this script)")

if __name__ == "__main__":
    print("Parameterized CLI Demo - Example Usage")
    print("=====================================")
    
    run_example_commands()
    
    print("\nTo run any of these commands, execute them directly in your terminal.")
    print("Note: Some commands may require specific datasets to be available.")