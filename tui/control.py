import threading
from dataclasses import dataclass, field

@dataclass
class ExperimentControl:
    """A thread-safe object to control the state of a running experiment."""
    cancel_event: threading.Event = field(default_factory=threading.Event)
    pause_event: threading.Event = field(default_factory=threading.Event)
    pause_lock: threading.Lock = field(default_factory=threading.Lock)

    def is_cancelled(self) -> bool:
        """Check if the experiment has been cancelled."""
        return self.cancel_event.is_set()

    def is_paused(self) -> bool:
        """Check if the experiment is paused."""
        return self.pause_event.is_set()

    def pause(self):
        """Signal the experiment to pause."""
        with self.pause_lock:
            self.pause_event.set()

    def resume(self):
        """Signal the experiment to resume."""
        with self.pause_lock:
            self.pause_event.clear()

    def cancel(self):
        """Signal the experiment to cancel."""
        # If the experiment is paused, we need to resume it first
        # so that it can check for the cancellation signal and exit gracefully.
        if self.is_paused():
            self.resume()
        self.cancel_event.set()

    def check_pause(self):
        """If paused, block until resumed. Called from the worker thread."""
        if self.is_paused():
            self.pause_event.wait()
