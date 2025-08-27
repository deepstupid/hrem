"""Utility functions for the HRM System."""

import time
from typing import Dict, Any, List
from collections import defaultdict

def logger_callback(message: str):
    """A simple logger callback that prints to the console."""
    from rich.console import Console
    console = Console()
    if ("it/s" not in message and "%" not in message and
        "TensorFloat32" not in message and "Online softmax" not in message and
        "torch._prims_common.check" not in message and
        "FutureWarning" not in message and "UserWarning" not in message):
        console.print(message)