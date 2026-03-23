"""Reward configuration for the Tower of Hanoi environment."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RewardConfig:
    """Reward values used by the environment."""

    goal_reward: float = 20.0
    step_penalty: float = -0.1
    invalid_move_penalty: float = -1.0


def compute_reward(
    *,
    goal_reached: bool,
    invalid_move: bool,
    config: RewardConfig,
) -> float:
    """Combine the per-step, invalid-move, and goal rewards."""
    reward = config.step_penalty
    if invalid_move:
        reward += config.invalid_move_penalty
    if goal_reached:
        reward += config.goal_reward
    return reward
