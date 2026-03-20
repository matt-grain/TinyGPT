"""Device detection for GPU/CPU/MPS auto-selection (Colab-ready)."""

from __future__ import annotations
import torch


def get_device() -> torch.device:
    """Auto-detect best available device: CUDA > MPS > CPU.

    Google Colab provides free GPU (CUDA). Without this detection,
    the model runs on CPU even when GPU is available — 10-50x slower.
    MPS is Apple Silicon GPU (M1/M2 Macs).
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple MPS GPU")
    else:
        device = torch.device("cpu")
        print(f"Using CPU ({torch.get_num_threads()} threads)")
    return device
