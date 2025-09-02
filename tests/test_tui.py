import pytest
from unittest.mock import MagicMock, patch

from textual.pilot import Pilot
from textual.widgets import Select
from tui.main import DiscoveryTUI, ExperimentState

@pytest.fixture
def mock_model_runner():
    """Fixture to mock the ScientificModelRunner."""
    with patch('tui.main.ScientificModelRunner') as mock:
        yield mock

@pytest.fixture
def mock_config_manager():
    """Fixture to mock the ConfigManager and its dependencies."""
    with patch('tui.main.ConfigManager') as mock_cm:
        mock_instance = mock_cm.return_value
        mock_instance.load_model_configs.return_value = {"HRM": {}, "HREM": {}}

        with patch('tui.main.ChallengeRegistry') as mock_cr:
            mock_cr_instance = mock_cr.return_value
            mock_challenge = MagicMock()
            mock_challenge.name = "Test Challenge"
            mock_challenge.id = "test_challenge"
            mock_cr_instance.get_all_challenges.return_value = [mock_challenge]
            yield mock_cm, mock_cr

@patch('tui.main.DiscoveryTUI.run_experiment')
async def test_start_and_cancel_experiment(mock_run_experiment, mock_model_runner, mock_config_manager):
    """Test starting and then cancelling an experiment."""
    mock_worker = MagicMock()
    mock_worker.cancel = MagicMock()
    mock_run_experiment.return_value = mock_worker

    async with DiscoveryTUI().run_test(size=(120, 40)) as pilot:
        app = pilot.app

        await pilot.click("#start-button")
        await pilot.pause()

        assert app.state == ExperimentState.RUNNING
        assert app.query_one("#start-button").label == "🔁 Cancel"

        await pilot.click("#start-button")
        await pilot.pause()

        mock_worker.cancel.assert_called_once()

@patch('tui.main.DiscoveryTUI.run_experiment')
async def test_pause_and_resume_experiment(mock_run_experiment, mock_model_runner, mock_config_manager):
    """Test pausing and resuming an experiment."""
    mock_run_experiment.return_value = MagicMock()

    async with DiscoveryTUI().run_test(size=(120, 40)) as pilot:
        app = pilot.app
        await pilot.click("#start-button")
        await pilot.pause()
        assert app.state == ExperimentState.RUNNING

        control = app.experiment_control

        await pilot.click("#pause-button")
        await pilot.pause()

        assert app.state == ExperimentState.PAUSED
        assert control.is_paused() is True

        await pilot.click("#pause-button")
        await pilot.pause()

        assert app.state == ExperimentState.RUNNING
        assert control.is_paused() is False

@patch('tui.main.DiscoveryTUI.run_experiment')
async def test_graceful_quit(mock_run_experiment, mock_model_runner, mock_config_manager):
    """Test that quitting gracefully cancels a running experiment."""
    mock_worker = MagicMock()
    mock_worker.cancel = MagicMock()
    mock_run_experiment.return_value = mock_worker

    async with DiscoveryTUI().run_test(size=(120, 40)) as pilot:
        app = pilot.app
        await pilot.click("#start-button")
        await pilot.pause()
        assert app.state == ExperimentState.RUNNING

        await pilot.press("q")
        await pilot.pause()

        mock_worker.cancel.assert_called_once()
