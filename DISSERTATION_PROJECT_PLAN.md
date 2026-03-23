# Tower of Hanoi RL Dissertation Plan

## Research Questions & Aims

1. Can RL algorithms learn efficient policies for the three-peg Tower of Hanoi when they share the same environment, observation encoding, reward structure and compute budget?
2. Which algorithm among Tabular Q-learning, DQN, A2C and PPO delivers the best trade-off between efficiency, robustness and scaling as `n` increases?
3. How far beyond small `n` can each algorithm maintain high success rates before encountering a breaking point, and what does the efficiency gap look like near that limit?
4. Does zero-shot generalisation from `n` to `n + 1` hold for the neural baselines when a fixed maximum disk count and padded observation encoding are enforced?

The aim is to produce a fair, reproducible comparison across these algorithms, document their learning curves and transfer behaviour, and situate the findings in the broader dissertation narrative.

## Methodological Rationale

- **Gymnasium plus Stable-Baselines3** keeps the neural baselines standardised and reduces engineering drift. A shared padded one-hot observation of size `N_max * 3` and `Discrete(6)` action space guarantee comparability.
- **Tabular Q-learning** is coded in-house to serve as a classical reference, while DQN, A2C and PPO leverage SB3 to capture replay-based, on-policy and robust policy-gradient dynamics.
- **Zero-shot evaluation** (no fine-tuning) is the sole transfer setting to keep the study focused and technically consistent with fixed-size observations.
- **Local CPU first, Colab as overflow**: the study is scoped so core experiments finish on an M2 Max MacBook Pro; Colab accelerates repeated neural trials or additional `N_max = 7` explorations.
- **Reward simplicity** (`+20` for completion, `-0.1` per step, `-1` for invalid moves) keeps algorithmic differences interpretable, with shaping left as an optional extension.

## Dissertation Chapter Mapping

1. **Introduction**: Define the Tower of Hanoi, `2^n - 1` optimum, research questions, motivation for RL versus recursion, and the vision of zero-shot transfer.
2. **Methods**: Detail the environment (state encoding, action set, transition rules, helper APIs), the reward scheme, the observation padding, and the configuration-driven experiment protocol, plus algorithm selection logic for Tabular Q-learning, DQN, A2C and PPO.
3. **Results**: Present environment validation, within-size comparisons, breaking-point data, zero-shot `n -> n + 1` outcomes, and key visualisations or figures (learning curves, efficiency gap plots, breaking-point summary).
4. **Evaluation**: Reflect on which algorithm family performed best, where each failed, the limits of model-free RL for structured puzzles, the zero-shot findings, compute or reward limitations, and optional extension insights if available.
5. **Conclusion & Contributions**: Summarise empirical findings, discuss the dissertation claims (e.g., RL works for small `n` but degrades sharply, neural methods scale further than tabular, zero-shot transfer remains hard), and outline future work.

## Experiment Design Overview

- **Phase 1 – Environment validation**: Implement Gymnasium env, recursive solver, and helpers; verify legal/illegal moves and reward behaviour; save benchmark trajectories.
- **Phase 2 – Pilot calibration**: Run random policy, tabular Q-learning on small `n`, and short DQN/A2C/PPO pilots to set budgets, seeds, and reward scaling.
- **Phase 3 – Within-size comparison**: Train each algorithm on `n = 2, 3, 4, 5, 6` (optional `7` on Colab) with fixed budgets; record success rates, moves, efficiency gaps, invalid action rates, returns, steps to threshold, and wall-clock time.
- **Phase 4 – Breaking-point analysis**: Identify where each algorithm falls below a 90% success threshold or shows high efficiency gaps despite the budget; discuss the degradation patterns.
- **Phase 5 – Zero-shot `n -> n + 1`**: Freeze neural models trained on `n` and evaluate directly on `n + 1` (pairs `2->3`, `3->4`, `4->5`, `5->6`, optionally `6->7`); present these results separately.
- **Phase 6 – Optional extension**: Only if time permits, explore four-peg variants, reward shaping, action masking, or other rule variants in a confined appendix.

## Risks and Mitigations

- **Reward design bias**: Keep reward constants fixed across algorithms; treat any shaping experiment as optional.
- **Observation mismatch breaking zero-shot**: Build the fixed `N_max` padded encoding before training any neural agent.
- **Local CPU limitations**: Regularly monitor runtime, cap experiments at `N_max = 6`, and optionally move extra runs to Colab.
- **Uneven tuning leads to unfair comparison**: Freeze tuned configs after a small pilot stage rather than retuning per algorithm or per disk count.
- **Tabular Q-learning dominating write-up**: Frame it as a small-`n` baseline; emphasise neural baselines in the narrative and zero-shot sections.
- **Manim takes excessive time**: Keep animations post-processing and limit scenes to a few high-impact trajectories.

## Timeline (Weeks)

1. **Week 1**: Repository and dependency setup; recursive solver; environment spec discussion.
2. **Week 2**: Environment implementation, reward logic, padded observations; unit tests.
3. **Week 3**: Random and tabular baselines; metrics validation and trajectory export.
4. **Week 4**: SB3 training integration for DQN, A2C, PPO; smoke tests; checkpoint/logging structure.
5. **Week 5**: Pilot tuning on `n = 3/4`; finalise budgets, seeds, reward constants; freeze protocol.
6. **Week 6**: Within-size comparison for lower `n`; early learning curves and plots.
7. **Week 7**: Extended scaling to `N_max`; breaking-point analysis documentation.
8. **Week 8**: Zero-shot `n -> n + 1` evaluations; transfer plots and tables.
9. **Week 9**: Finalise figures, tables, and selected Manim animation from saved trajectories.
10. **Week 10**: Re-run noisy experiments if needed; collate results for the paper; draft results narrative.
11-12. **Weeks 11-12**: Write final chapters, polish evaluation, and add optional extension only if core work is stable.

## Expected Contributions

- A validated custom Gymnasium Tower of Hanoi environment with helper APIs and padded observation encoding.
- A reproducible comparison across Tabular Q-learning, DQN, A2C, and PPO with shared configs, metrics, and logging.
- Empirical evidence of where each algorithm breaks down as `n` increases on an M2 Max laptop budget, plus zero-shot `n -> n + 1` assessments.
- Visual assets (plots, tables, Manim scenes) that highlight learning curves, efficiency gaps, breaking points, and transfer outcomes.
- A dissertation narrative that positions the contributions within reinforcement learning’s ability to tackle structured combinatorial puzzles.
