import threading

class ExperimentControl:
    """
    A thread-safe class to control the state of a running experiment.
    This object is passed from the TUI thread to the worker thread to allow
    the UI to pause, resume, and cancel the experiment.
    """
    def __init__(self):
        self._pause_event = threading.Event()
        self._cancel_event = threading.Event()
        self._pause_event.set()  # A set event means "not paused"

    def pause(self) -> None:
        """Signal the experiment to pause execution."""
        self._pause_event.clear()  # Clearing the event will cause threads calling wait() to block.

    def resume(self) -> None:
        """Signal the experiment to resume execution."""
        self._pause_event.set()  # Setting the event will cause threads calling wait() to unblock.

    def cancel(self) -> None:
        """Signal the experiment to cancel."""
        self._cancel_event.set()

    def wait_if_paused(self) -> None:
        """
        Called from within the worker thread. This method will block
        until the resume() method is called from the main thread.
        """
        self._pause_event.wait()

    def is_cancelled(self) -> bool:
        """
        Called from within the worker thread to check if cancellation
        has been requested.
        """
        return self._cancel_event.is_set()

    @property
    def is_paused(self) -> bool:
        """Check if the experiment is currently paused."""
        return not self._pause_event.is_set()
