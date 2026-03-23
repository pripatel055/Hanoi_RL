"""Metrics used by the Tower of Hanoi experiments."""

from __future__ import annotations


def efficiency_gap(moves_to_solve: int, n_disks: int) -> int:
    """Return the gap between an observed trajectory and the optimum."""
    if n_disks <= 0:
        raise ValueError("n_disks must be positive")
    return moves_to_solve - ((2**n_disks) - 1)
