"""Metrics collection and timing utilities for the HRM/HREM demo system."""

from typing import Dict, Any, Optional
from demo_timing_utils import TimingManager

class MetricsCollector(TimingManager):
    """Collects and manages timing and performance metrics for the demo."""
    
    def __init__(self):
        super().__init__()