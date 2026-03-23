"""Observation encoding utilities for Tower of Hanoi states."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def validate_state(state: Sequence[int] | np.ndarray, n_pegs: int = 3) -> np.ndarray:
    """Validate a symbolic state vector and return it as an integer array."""
    state_array = np.asarray(state, dtype=np.int64)
    if state_array.ndim != 1:
        raise ValueError("state must be a one-dimensional sequence of peg indices")
    if np.any(state_array < 0) or np.any(state_array >= n_pegs):
        raise ValueError(f"state entries must be between 0 and {n_pegs - 1}")
    return state_array


def one_hot_encode_state(
    state: Sequence[int] | np.ndarray,
    n_max: int,
    n_pegs: int = 3,
) -> np.ndarray:
    """Encode a symbolic state as a padded one-hot vector of length n_max * n_pegs."""
    state_array = validate_state(state, n_pegs=n_pegs)
    if state_array.size > n_max:
        raise ValueError("state length cannot exceed n_max")

    encoded = np.zeros((n_max, n_pegs), dtype=np.float32)
    for disk_index, peg_index in enumerate(state_array):
        encoded[disk_index, peg_index] = 1.0

    return encoded.reshape(-1)
