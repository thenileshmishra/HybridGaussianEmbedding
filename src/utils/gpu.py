"""GPU monitoring and metrics logging utilities."""

import os

import pandas as pd

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False


def monitor_gpu_usage():
    """Print current GPU memory usage and utilization."""
    if not PYNVML_AVAILABLE:
        print("GPU monitoring unavailable (pynvml not installed)")
        return

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    util_rate = pynvml.nvmlDeviceGetUtilizationRates(handle)
    print(f"GPU Memory Used: {mem_info.used / 1024**2:.2f} MiB / {mem_info.total / 1024**2:.2f} MiB")
    print(f"GPU Utilization: {util_rate.gpu}%")


def log_gpu_metrics(epoch: int, doc_idx: int, csv_path: str = "metrics/gpu_usage.csv"):
    """Log GPU metrics to a CSV file.

    Args:
        epoch: Current training epoch.
        doc_idx: Current document index.
        csv_path: Path to the output CSV file.
    """
    if not PYNVML_AVAILABLE:
        return

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    util = pynvml.nvmlDeviceGetUtilizationRates(handle)

    df = pd.DataFrame([{
        "Epoch": epoch,
        "DocIdx": doc_idx,
        "GPU_Memory_Used_MB": mem_info.used / 1024**2,
        "GPU_Utilization_Percent": util.gpu,
    }])

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    if not os.path.exists(csv_path):
        df.to_csv(csv_path, index=False)
    else:
        df.to_csv(csv_path, mode="a", header=False, index=False)
