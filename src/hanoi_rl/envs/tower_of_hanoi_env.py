"""Gymnasium environment for the three-peg Tower of Hanoi puzzle."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .encoding import one_hot_encode_state
from .reward import RewardConfig, compute_reward

ACTIONS: tuple[tuple[int, int], ...] = (
    (0, 1),
    (0, 2),
    (1, 0),
    (1, 2),
    (2, 0),
    (2, 1),
)


@dataclass
class EnvConfig:
    """Configuration values for the Tower of Hanoi environment."""

    n_disks: int = 3
    n_max: int = 6
    source_peg: int = 0
    auxiliary_peg: int = 1
    target_peg: int = 2
    step_limit_multiplier: int = 3
    reward: RewardConfig = field(default_factory=RewardConfig)

    def __post_init__(self) -> None:
        if self.n_disks <= 0:
            raise ValueError("n_disks must be positive")
        if self.n_max < self.n_disks:
            raise ValueError("n_max must be at least n_disks")
        if self.step_limit_multiplier <= 0:
            raise ValueError("step_limit_multiplier must be positive")


class TowerOfHanoiEnv(gym.Env):
    """A deterministic environment for the standard three-peg puzzle."""

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}

    def __init__(self, config: EnvConfig | None = None, render_mode: str | None = None) -> None:
        super().__init__()
        self.config = config or EnvConfig()
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(len(ACTIONS))
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.config.n_max * 3,),
            dtype=np.float32,
        )

        self.state: np.ndarray | None = None
        self.step_count = 0
        self._terminated = False
        self._truncated = False
        self._trajectory: list[dict[str, Any]] = []

    @staticmethod
    def optimal_move_count(n_disks: int) -> int:
        """Return the mathematically optimal move count for n disks."""
        if n_disks <= 0:
            raise ValueError("n_disks must be positive")
        return (2**n_disks) - 1

    @property
    def max_steps(self) -> int:
        """Maximum episode length for the current configuration."""
        return self.config.step_limit_multiplier * self.optimal_move_count(self.config.n_disks)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the environment to the starting state."""
        super().reset(seed=seed)
        if options:
            raise NotImplementedError("reset options are not used in the current scaffold")

        self.state = np.full(self.config.n_disks, self.config.source_peg, dtype=np.int64)
        self.step_count = 0
        self._terminated = False
        self._truncated = False
        self._trajectory = [self._snapshot(action=None, reward=0.0, invalid_move=False)]

        observation = self._get_observation()
        info = self._build_info(invalid_move=False)
        return observation, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Apply one of the six directed peg moves."""
        if self.state is None:
            raise RuntimeError("environment must be reset before step is called")
        if self._terminated or self._truncated:
            raise RuntimeError("episode has ended; call reset before stepping again")
        if not self.action_space.contains(action):
            raise ValueError(f"action must be in range [0, {self.action_space.n - 1}]")

        source_peg, destination_peg = ACTIONS[int(action)]
        invalid_move = True
        invalid_reason = "empty_source"

        moving_disk = self._top_disk(source_peg)
        if moving_disk is not None:
            destination_top = self._top_disk(destination_peg)
            if destination_top is None or moving_disk < destination_top:
                self.state[moving_disk] = destination_peg
                invalid_move = False
                invalid_reason = None
            else:
                invalid_reason = "smaller_disk_on_destination"

        self.step_count += 1
        terminated = self.is_goal_state()
        truncated = self.step_count >= self.max_steps and not terminated
        self._terminated = terminated
        self._truncated = truncated
        reward = compute_reward(
            goal_reached=terminated,
            invalid_move=invalid_move,
            config=self.config.reward,
        )

        observation = self._get_observation()
        info = self._build_info(invalid_move=invalid_move, invalid_reason=invalid_reason)
        self._trajectory.append(
            self._snapshot(action=action, reward=reward, invalid_move=invalid_move)
        )

        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, truncated, info

    def get_valid_actions(self) -> list[int]:
        """Return the subset of actions that correspond to legal moves."""
        if self.state is None:
            return []

        valid_actions: list[int] = []
        for action_index, (source_peg, destination_peg) in enumerate(ACTIONS):
            moving_disk = self._top_disk(source_peg)
            if moving_disk is None:
                continue
            destination_top = self._top_disk(destination_peg)
            if destination_top is None or moving_disk < destination_top:
                valid_actions.append(action_index)
        return valid_actions

    def is_goal_state(self, state: np.ndarray | None = None) -> bool:
        """Check whether all disks are on the target peg."""
        active_state = self.state if state is None else state
        if active_state is None:
            return False
        return bool(np.all(active_state == self.config.target_peg))

    def export_trajectory(self) -> list[dict[str, Any]]:
        """Return a copy of the tracked trajectory for offline analysis."""
        return [
            {
                key: list(value) if isinstance(value, list) else value
                for key, value in snapshot.items()
            }
            for snapshot in self._trajectory
        ]

    def render(self) -> str | None:
        """Render the current state as text."""
        if self.state is None:
            text = "<uninitialised TowerOfHanoiEnv>"
        else:
            pegs = {peg_index: [] for peg_index in range(3)}
            for disk_index in range(self.config.n_disks - 1, -1, -1):
                pegs[int(self.state[disk_index])].append(disk_index)
            text = (
                f"step={self.step_count} "
                f"peg0={pegs[0]} peg1={pegs[1]} peg2={pegs[2]} "
                f"goal={self.is_goal_state()}"
            )

        if self.render_mode == "human":
            print(text)
            return None
        return text

    def close(self) -> None:
        """Close the environment."""
        return None

    def _top_disk(self, peg_index: int) -> int | None:
        if self.state is None:
            return None
        disks = np.where(self.state == peg_index)[0]
        if disks.size == 0:
            return None
        return int(disks.min())

    def _get_observation(self) -> np.ndarray:
        if self.state is None:
            raise RuntimeError("environment state is not initialised")
        return one_hot_encode_state(self.state, n_max=self.config.n_max)

    def _build_info(
        self,
        *,
        invalid_move: bool,
        invalid_reason: str | None = None,
    ) -> dict[str, Any]:
        if self.state is None:
            raise RuntimeError("environment state is not initialised")
        return {
            "state": tuple(int(peg) for peg in self.state),
            "valid_actions": self.get_valid_actions(),
            "invalid_move": invalid_move,
            "invalid_reason": invalid_reason,
            "step_count": self.step_count,
            "terminated": self._terminated,
            "truncated": self._truncated,
            "optimal_move_count": self.optimal_move_count(self.config.n_disks),
        }

    def _snapshot(
        self,
        *,
        action: int | None,
        reward: float,
        invalid_move: bool,
    ) -> dict[str, Any]:
        if self.state is None:
            raise RuntimeError("environment state is not initialised")
        return {
            "step": self.step_count,
            "action": action,
            "state": [int(peg) for peg in self.state.tolist()],
            "reward": reward,
            "invalid_move": invalid_move,
            "terminated": self._terminated,
            "truncated": self._truncated,
        }
