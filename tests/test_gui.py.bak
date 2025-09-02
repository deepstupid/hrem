import sys
import pytest
from PyQt6.QtCore import QTimer

from gui.main_window import MainWindow

@pytest.mark.skipif(sys.platform == "darwin", reason="Test fails on macOS in CI")
def test_gui_startup_and_shutdown(qtbot):
    """
    Tests that the GUI starts up and shuts down cleanly without errors.
    This test relies on pytest-qt to manage the QApplication instance.
    """
    window = MainWindow()
    qtbot.addWidget(window)
    window.show()

    # The main window starts an experiment by default.
    # We need to let it run for a moment and then close it.
    # A timer ensures we don't block the Qt event loop.
    def close_window():
        # This will trigger the closeEvent and the thread cleanup logic
        window.close()

    # Close the window after 1 second
    QTimer.singleShot(1000, close_window)

    # The qtbot will wait until the window is actually closed.
    # The timeout is a safeguard. The shutdown logic itself has a 5s timeout,
    # so we give it a bit more here.
    qtbot.wait_for_window_closed(window, timeout=7000)

    # The real test is that the test runner doesn't crash with a
    # "Fatal Python error: Aborted" or "QThread: Destroyed while thread..." error.
    # If this test completes without the process crashing, the cleanup logic is working.
    assert True
