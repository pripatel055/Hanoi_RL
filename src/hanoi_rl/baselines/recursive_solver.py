"""Reference recursive solver for the Tower of Hanoi puzzle."""

from __future__ import annotations

Move = tuple[int, int]


def optimal_move_count(n_disks: int) -> int:
    """Return the optimal number of moves for the standard puzzle."""
    if n_disks <= 0:
        raise ValueError("n_disks must be positive")
    return (2**n_disks) - 1


def solve_tower_of_hanoi(
    n_disks: int,
    source: int = 0,
    auxiliary: int = 1,
    target: int = 2,
) -> list[Move]:
    """Return the optimal sequence of moves for n_disks."""
    if n_disks <= 0:
        raise ValueError("n_disks must be positive")

    moves: list[Move] = []

    def _solve(count: int, start: int, spare: int, finish: int) -> None:
        if count == 1:
            moves.append((start, finish))
            return
        _solve(count - 1, start, finish, spare)
        moves.append((start, finish))
        _solve(count - 1, spare, start, finish)

    _solve(n_disks, source, auxiliary, target)
    return moves
