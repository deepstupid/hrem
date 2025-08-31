from typing import Dict, Any
from textual.types import MessageTarget

from sc_engine.core.progress_handler import ProgressHandler
from tui.messages import EngineEvent

class TUIProgressHandler(ProgressHandler):
    """
    A progress handler that bridges the scientific engine with the Textual UI
    by posting messages.
    """

    def __init__(self, owner: MessageTarget):
        self.owner = owner

    def on_progress(self, event_type: str, data: Dict[str, Any]):
        """
        Receives a progress update from the engine and posts it as a message.
        """
        self.owner.post_message(EngineEvent(event_type, data))
