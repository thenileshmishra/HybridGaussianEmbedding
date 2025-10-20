"""CPU memory monitoring utilities."""

import psutil


def get_memory_usage():
    """Get current process memory usage in bytes.

    Returns:
        Resident set size (RSS) in bytes.
    """
    process = psutil.Process()
    return process.memory_info().rss
