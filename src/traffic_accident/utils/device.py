import torch


def get_device(preferred: str | None = None) -> torch.device:
    """Return the requested torch device, falling back to CUDA/CPU auto-detection."""
    if preferred:
        return torch.device(preferred)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

