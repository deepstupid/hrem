#!/usr/bin/env python3
"""
Smoke test for the TUI to ensure all components can be imported and run.
"""

import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_tui_imports():
    """Test that all TUI components can be imported."""
    try:
        from tui.app import HRMApp
        print("✓ Main app imports successfully")
        
        from tui.screens.evaluation import EvaluationScreen
        print("✓ Evaluation screen imports successfully")
        
        from tui.screens.optimization import OptimizationScreen
        print("✓ Optimization screen imports successfully")
        
        from tui.screens.visualizer import DatasetVisualizer
        print("✓ Visualizer screen imports successfully")
        
        from tui.screens.testing import TestingScreen
        print("✓ Testing screen imports successfully")
        
        from tui.screens.dataset_management import DatasetManagementScreen
        print("✓ Dataset management screen imports successfully")
        
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_textual_available():
    """Test that Textual is available."""
    try:
        import textual
        print("✓ Textual is available")
        return True
    except ImportError:
        print("✗ Textual is not installed")
        return False

if __name__ == "__main__":
    print("Running TUI smoke test...")
    
    if not test_textual_available():
        print("Please install Textual: pip install textual")
        sys.exit(1)
        
    if not test_tui_imports():
        print("TUI import test failed")
        sys.exit(1)
        
    print("All TUI smoke tests passed!")
    print("You can now run the TUI with: python -m tui.app")