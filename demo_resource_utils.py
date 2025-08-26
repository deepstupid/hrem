"""Resource detection utilities for the HRM/HREM demo system."""

import os
import torch
from typing import Literal

def get_cpu_count() -> int:
    """Get the number of CPU cores."""
    return os.cpu_count() or 1

def get_gpu_memory_gb() -> int:
    """Get the total GPU memory in GB. Returns 0 if no GPU is available."""
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        return round(properties.total_memory / (1024**3))
    return 0

def get_resource_profile() -> Literal["low", "medium", "high"]:
    """Determine the resource profile based on system specs."""
    cpu_cores = get_cpu_count()
    gpu_mem_gb = get_gpu_memory_gb()

    if cpu_cores > 16 and gpu_mem_gb > 16:
        return "high"
    elif cpu_cores >= 8 and gpu_mem_gb >= 8:
        return "medium"
    else:
        return "low"
