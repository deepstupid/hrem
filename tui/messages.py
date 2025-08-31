from typing import Dict, Any
from textual.message import Message

class EngineEvent(Message):
    """A message to relay events from the scientific engine to the TUI."""

    def __init__(self, event_type: str, data: Dict[str, Any]):
        self.event_type = event_type
        self.data = data
        super().__init__()
