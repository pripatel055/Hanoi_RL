"""Simple tabular Q-learning agent used as a baseline."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import DefaultDict

import numpy as np


@dataclass
class TabularQConfig:
    """Configuration for tabular Q-learning."""

    alpha: float = 0.1
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.995
    action_size: int = 6
    seed: int | None = None


class TabularQAgent:
    """A lightweight tabular Q-learning implementation."""

    def __init__(self, config: TabularQConfig | None = None) -> None:
        self.config = config or TabularQConfig()
        self.epsilon = self.config.epsilon_start
        self.q_values: DefaultDict[tuple[int, ...], np.ndarray] = defaultdict(
            lambda: np.zeros(self.config.action_size, dtype=np.float64)
        )
        self.rng = np.random.default_rng(self.config.seed)

    def state_key(self, state: np.ndarray | list[int] | tuple[int, ...]) -> tuple[int, ...]:
        """Convert a symbolic state into a hashable key."""
        return tuple(int(value) for value in state)

    def select_action(self, state: np.ndarray | list[int] | tuple[int, ...], greedy: bool = False) -> int:
        """Select an action using epsilon-greedy exploration."""
        key = self.state_key(state)
        if not greedy and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.config.action_size))
        return self._argmax_with_random_tie_break(self.q_values[key])

    def update(
        self,
        state: np.ndarray | list[int] | tuple[int, ...],
        action: int,
        reward: float,
        next_state: np.ndarray | list[int] | tuple[int, ...],
        done: bool,
    ) -> None:
        """Apply the standard Q-learning update."""
        state_key = self.state_key(state)
        next_key = self.state_key(next_state)

        best_next = 0.0 if done else float(np.max(self.q_values[next_key]))
        td_target = reward + (self.config.gamma * best_next)
        td_error = td_target - self.q_values[state_key][action]
        self.q_values[state_key][action] += self.config.alpha * td_error

    def decay_epsilon(self) -> None:
        """Decay epsilon while respecting the minimum value."""
        self.epsilon = max(self.config.epsilon_end, self.epsilon * self.config.epsilon_decay)

    def greedy_action(self, state: np.ndarray | list[int] | tuple[int, ...]) -> int:
        """Return the current greedy action."""
        return self.select_action(state, greedy=True)

    def _argmax_with_random_tie_break(self, values: np.ndarray) -> int:
        max_value = np.max(values)
        candidates = np.flatnonzero(values == max_value)
        return int(self.rng.choice(candidates))
