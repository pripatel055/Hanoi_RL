"""Random seed helpers."""

from __future__ import annotations

import random

import numpy as np


def set_global_seeds(seed: int) -> None:
    """Set seeds for standard Python, NumPy, and PyTorch if available."""
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
