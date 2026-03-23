# Tower of Hanoi with Reinforcement Learning

This repository contains the implementation scaffold for a dissertation project on solving the Tower of Hanoi with reinforcement learning.

The current structure is designed around:

- a custom `gymnasium` environment for the three-peg puzzle;
- a formal comparison across `Tabular Q-learning`, `DQN`, `A2C`, and `PPO`;
- zero-shot evaluation from `n` disks to `n + 1`;
- reproducible experiment configuration, plotting, and report-ready outputs.

## Project layout

```text
configs/     Experiment, environment, and algorithm settings.
src/         Python package code.
tests/       Unit and smoke tests.
outputs/     Saved models, logs, figures, tables, and trajectories.
```

## Getting started

Use `uv` with Python `3.14`:

```bash
uv python pin 3.14
UV_CACHE_DIR=.uv-cache uv venv --python 3.14 .venv
UV_CACHE_DIR=.uv-cache uv sync --dev
```

Run commands through `uv`:

```bash
UV_CACHE_DIR=.uv-cache uv run pytest
UV_CACHE_DIR=.uv-cache uv run python -m hanoi_rl.training.train_tabular
```

## Immediate implementation order

1. Finish the Tower of Hanoi environment logic and validation tests.
2. Implement the full tabular Q-learning training loop.
3. Integrate `stable-baselines3` experiments for `DQN`, `A2C`, and `PPO`.
4. Add evaluation, plotting, and zero-shot comparison scripts.

## Planning documents

- `IMPLEMENTATION_PLAN.md`
- `DISSERTATION_PROJECT_PLAN.md`
