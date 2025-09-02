import pytest
from unittest.mock import MagicMock, patch

from textual.pilot import Pilot
from textual.widgets import Select, Button
from tui.app import DiscoveryTUI
from tui.engine import ExperimentRunner

@pytest.fixture
def mock_model_runner():
    """Fixture to mock the ScientificModelRunner."""
    with patch('tui.engine.ScientificModelRunner') as mock:
        yield mock

@pytest.fixture
def mock_config_manager():
    """Fixture to mock the ConfigManager and its dependencies."""
    with patch('tui.widgets.setup_view.ConfigManager') as mock_cm:
        mock_instance = mock_cm.return_value
        mock_instance.load_model_configs.return_value = {"HRM": {}, "HREM": {}}

        with patch('tui.widgets.setup_view.ChallengeRegistry') as mock_cr:
            mock_cr_instance = mock_cr.return_value
            mock_challenge = MagicMock()
            mock_challenge.name = "Test Challenge"
            mock_challenge.id = "test_challenge"
            mock_cr_instance.get_all_challenges.return_value = [mock_challenge]
            yield mock_cm, mock_cr

@patch('tui.engine.ExperimentRunner.run_experiment')
async def test_start_and_cancel_experiment(mock_run_experiment, mock_model_runner, mock_config_manager):
    """Test starting and then cancelling an experiment."""

    async with DiscoveryTUI().run_test(size=(120, 80)) as pilot:
        # Get to the discovery screen
        await pilot.click("#proceed")
        await pilot.pause()

        # Start the experiment
        await pilot.click("#start-button")
        await pilot.pause()

        # Check that the run view is now visible
        run_view = pilot.app.query_one("RunView")
        assert run_view.has_class("hidden") is False

        # Cancel the experiment
        await pilot.click("#cancel-button")
        await pilot.pause()

        log = pilot.app.query_one("#live-log")
        assert "Cancellation request sent" in log.render()

@patch('tui.engine.ExperimentRunner.run_experiment')
async def test_pause_and_resume_experiment(mock_run_experiment, mock_model_runner, mock_config_manager):
    """Test pausing and resuming an experiment."""

    async with DiscoveryTUI().run_test(size=(120, 80)) as pilot:
        # Get to the discovery screen and start the experiment
        await pilot.click("#proceed")
        await pilot.pause()
        await pilot.click("#start-button")
        await pilot.pause()

        runner = pilot.app.query_one(ExperimentRunner)
        control = runner.experiment_control

        # Pause the experiment
        await pilot.click("#pause-button")
        await pilot.pause()
        assert control.is_paused is True
        status_tracker = pilot.app.query_one("#status-tracker")
        assert "Paused" in status_tracker.renderable

        # Resume the experiment
        await pilot.click("#pause-button")
        await pilot.pause()
        assert control.is_paused is False
        assert "Running" in status_tracker.renderable

@patch('tui.engine.ExperimentRunner.cancel_experiment')
async def test_graceful_quit(mock_cancel_experiment, mock_model_runner, mock_config_manager):
    """Test that quitting gracefully cancels a running experiment."""
    async with DiscoveryTUI().run_test(size=(120, 80)) as pilot:
        # Get to the discovery screen and start the experiment
        await pilot.click("#proceed")
        await pilot.pause()
        await pilot.click("#start-button")
        await pilot.pause()

        # Press quit
        await pilot.press("q")
        await pilot.pause()

        mock_cancel_experiment.assert_called_once()
