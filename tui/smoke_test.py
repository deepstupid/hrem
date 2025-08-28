"""
Smoke test for the TUI to ensure all components can be imported and run.
"""

def test_tui_imports():
    """Test that all TUI components can be imported."""
    from tui.app import HRMApp
    from tui.screens.evaluation import EvaluationScreen
    from tui.screens.optimization import OptimizationScreen
    from tui.screens.visualizer import DatasetVisualizer
    from tui.screens.testing import TestingScreen
    from tui.screens.dataset_management import DatasetManagementScreen

    # If all imports succeed, this test passes.
    assert True

def test_textual_available():
    """Test that Textual is available."""
    import textual
    assert textual is not None