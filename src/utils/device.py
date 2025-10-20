"""Device management utilities."""

import torch


def move_to_device(tensor, device):
    """Move a tensor or list of tensors to the specified device.

    Args:
        tensor: A torch.Tensor or list of tensors.
        device: Target device (e.g., 'cuda' or 'cpu').

    Returns:
        Tensor(s) moved to the target device.
    """
    if isinstance(tensor, torch.Tensor):
        return tensor.to(device)
    elif isinstance(tensor, list):
        return [t.to(device) if isinstance(t, torch.Tensor) else t for t in tensor]
    return tensor


def get_device(prefer_cuda: bool = True) -> torch.device:
    """Get the best available device.

    Args:
        prefer_cuda: Whether to prefer CUDA if available.

    Returns:
        torch.device instance.
    """
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
