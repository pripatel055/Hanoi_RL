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
