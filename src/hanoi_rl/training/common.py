"""Shared config and evaluation helpers for training scripts."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Any, Protocol

import numpy as np

from hanoi_rl.analysis.metrics import efficiency_gap
from hanoi_rl.envs import EnvConfig, RewardConfig, TowerOfHanoiEnv


class PredictablePolicy(Protocol):
    """Protocol for policies that expose an SB3-like predict method."""

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> tuple[Any, Any]:
        """Predict an action for the provided observation."""


@dataclass
class PolicyEpisodeResult:
    """Per-episode metrics for policy evaluation."""

    episode: int
    reward: float
    steps: int
    success: bool
    truncated: bool
    invalid_moves: int
    invalid_action_rate: float
    efficiency_gap: int | None


def build_env_config(config_data: dict[str, Any]) -> EnvConfig:
    """Create an EnvConfig from YAML data."""
    reward_data = config_data.get("reward", {})
    reward_config = RewardConfig(
        goal_reward=float(reward_data.get("goal_reward", RewardConfig.goal_reward)),
        step_penalty=float(reward_data.get("step_penalty", RewardConfig.step_penalty)),
        invalid_move_penalty=float(
            reward_data.get("invalid_move_penalty", RewardConfig.invalid_move_penalty)
        ),
    )
    return EnvConfig(
        n_disks=int(config_data.get("n_disks", 3)),
        n_max=int(config_data.get("n_max", 6)),
        source_peg=int(config_data.get("source_peg", 0)),
        auxiliary_peg=int(config_data.get("auxiliary_peg", 1)),
        target_peg=int(config_data.get("target_peg", 2)),
        step_limit_multiplier=int(config_data.get("step_limit_multiplier", 3)),
        reward=reward_config,
    )


def summarise_policy_results(
    episode_results: list[PolicyEpisodeResult],
    *,
    n_disks: int,
) -> dict[str, Any]:
    """Aggregate episode-level metrics into a compact summary."""
    if not episode_results:
        raise ValueError("episode_results cannot be empty")

    successes = [result for result in episode_results if result.success]
    first_success_episode = next(
        (result.episode for result in episode_results if result.success),
        None,
    )
    summary: dict[str, Any] = {
        "episodes": len(episode_results),
        "success_count": len(successes),
        "success_rate": len(successes) / len(episode_results),
        "mean_reward": mean(result.reward for result in episode_results),
        "mean_steps": mean(result.steps for result in episode_results),
        "mean_invalid_action_rate": mean(
            result.invalid_action_rate for result in episode_results
        ),
        "first_success_episode": first_success_episode,
        "optimal_move_count": TowerOfHanoiEnv.optimal_move_count(n_disks),
    }

    if successes:
        summary["mean_successful_steps"] = mean(result.steps for result in successes)
        summary["mean_efficiency_gap"] = mean(
            result.efficiency_gap for result in successes if result.efficiency_gap is not None
        )
    else:
        summary["mean_successful_steps"] = None
        summary["mean_efficiency_gap"] = None

    return summary


def evaluate_policy(
    policy: PredictablePolicy,
    env_config: EnvConfig,
    *,
    episodes: int,
    deterministic: bool = True,
    base_seed: int | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate a policy on the configured environment."""
    if episodes <= 0:
        raise ValueError("episodes must be positive")

    env = TowerOfHanoiEnv(env_config)
    episode_results: list[PolicyEpisodeResult] = []

    for episode_index in range(1, episodes + 1):
        observation, info = env.reset(
            seed=None if base_seed is None else base_seed + episode_index
        )
        terminated = False
        truncated = False
        total_reward = 0.0
        invalid_moves = 0

        while not (terminated or truncated):
            action, _ = policy.predict(observation, deterministic=deterministic)
            observation, reward, terminated, truncated, info = env.step(int(action))
            total_reward += reward
            invalid_moves += int(info["invalid_move"])

        steps = int(info["step_count"])
        success = bool(terminated)
        episode_results.append(
            PolicyEpisodeResult(
                episode=episode_index,
                reward=total_reward,
                steps=steps,
                success=success,
                truncated=bool(truncated),
                invalid_moves=invalid_moves,
                invalid_action_rate=(invalid_moves / steps) if steps else 0.0,
                efficiency_gap=efficiency_gap(steps, env_config.n_disks) if success else None,
            )
        )

    env.close()
    return (
        summarise_policy_results(episode_results, n_disks=env_config.n_disks),
        [
            {
                "episode": result.episode,
                "reward": result.reward,
                "steps": result.steps,
                "success": result.success,
                "truncated": result.truncated,
                "invalid_moves": result.invalid_moves,
                "invalid_action_rate": result.invalid_action_rate,
                "efficiency_gap": result.efficiency_gap,
            }
            for result in episode_results
        ],
    )
