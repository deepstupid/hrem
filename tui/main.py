#!/usr/bin/env python3
"""
Entry point for the HRM System TUI.
"""

import sys
import os
import traceback

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    """Run the HRM System TUI."""
    try:
        from tui.app import HRMApp
        app = HRMApp()
        app.run()
    except ImportError as e:
        print(f"Error importing TUI: {e}")
        print("Make sure you have installed the required dependencies:")
        print("pip install textual")
        sys.exit(1)
    except Exception as e:
        print(f"Error running TUI: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()