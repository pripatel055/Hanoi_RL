# Tower of Hanoi RL Implementation Plan

## Status

Revised on `17 March 2026` after scope decisions were confirmed.

This version assumes:

- a formal comparison across RL algorithms;
- use of standard Gymnasium-compatible RL libraries;
- training on a local machine first, with Google Colab available if longer runs are needed;
- `n -> n + 1` assessed as **zero-shot evaluation only**;
- four-peg and rule-variant work kept optional;
- British spelling throughout the report and project materials.

## 1. Project Framing

### 1.1 Core aim

Build a reproducible Python project that tests whether reinforcement learning agents can learn efficient Tower of Hanoi policies and how well those policies scale as the puzzle grows.

### 1.2 Main research questions

1. Can RL agents solve the standard three-peg Tower of Hanoi reliably for small and medium disk counts?
2. Which RL algorithm performs best under the same environment, reward design, and training budget?
3. How close do learned policies get to the optimal `2^n - 1` move solution?
4. At what disk count does each algorithm begin to break down?
5. Do policies trained on `n` disks generalise zero-shot to `n + 1` disks?

### 1.3 Success criteria

The project is successful if it delivers:

- a correct custom Gymnasium environment;
- a fair, reproducible algorithm comparison;
- clear plots and tables for learning, efficiency, and scaling;
- a defensible zero-shot evaluation design;
- enough evidence to discuss both strengths and limitations of RL on a structured puzzle.

## 2. Confirmed Scope and Design Decisions

### 2.1 Comparison set

The comparison should include:

- **Random policy** as a weak baseline;
- **Recursive oracle solver** as the optimal reference;
- **Tabular Q-learning** as a classical RL baseline;
- **DQN** as the main value-based neural baseline;
- **A2C** as a lighter on-policy baseline;
- **PPO** as a stronger policy-gradient baseline.

This gives a useful spread across:

- non-learning versus learning baselines;
- tabular versus function approximation;
- value-based versus policy-gradient methods.

### 2.2 Library choice

Recommended libraries:

- `gymnasium` for the environment;
- `stable-baselines3` for `DQN`, `A2C`, and `PPO`;
- `torch` as the underlying deep learning framework;
- `numpy` and `pandas` for data handling;
- `matplotlib` and optionally `seaborn` for plots;
- `manim` for animations;
- `pytest` for tests.

Notes:

- `stable-baselines3` keeps the neural baselines standard and well documented.
- `Tabular Q-learning` should be implemented directly in the project because SB3 does not cover it.
- `sb3-contrib` should be avoided in the core comparison unless an action-masking extension is added later, because masking changes the problem difficulty.

### 2.3 Compute assumptions

Primary compute target:

- local CPU on an `M2 Max MacBook Pro`.

Secondary compute option:

- Google Colab for repeated neural runs or a larger `N_max`.

Practical implication:

- the core study should be designed to run credibly on local hardware;
- Colab should improve throughput, not rescue an over-ambitious experimental design.

### 2.4 Transfer scope

The transfer study should be:

- **zero-shot only**.

That means:

- train on size `n`;
- freeze the trained policy;
- evaluate directly on size `n + 1`;
- do not fine-tune on `n + 1`.

### 2.5 Extension scope

The following should remain optional:

- four-peg Tower of Hanoi;
- action masking;
- reward shaping ablations;
- rule variants.

The dissertation should not depend on these to be complete.

## 3. Recommended Technical Strategy

### 3.1 Fixed-size observation design

The main technical challenge is zero-shot evaluation from `n` to `n + 1`.
Standard neural policies require a fixed observation size, so the project should use a shared maximum disk count `N_max`.

Recommended default:

- set `N_max = 6` for the main local study;
- consider `N_max = 7` only if pilot runs on Colab look practical.

Why this matters:

- it makes zero-shot testing technically valid for neural agents;
- it allows the same policy architecture to be reused across all training sizes;
- it prevents the transfer experiment from collapsing into a shape-mismatch problem.

### 3.2 Formal comparison principle

The algorithm comparison should be fair in the following sense:

- same environment rules;
- same observation encoding;
- same action space;
- same reward design;
- same termination criteria;
- same evaluation schedule;
- same random seed protocol;
- similar hyperparameter tuning effort for each neural algorithm.

### 3.3 Narrative structure

The project should be written in two layers at once:

- an **engineering layer** that explains how the system is built and evaluated;
- a **dissertation layer** that frames the same work as a methods/results/evaluation study.

This avoids rewriting the project logic later.

## 4. System Architecture

### 4.1 Recommended repository structure

```text
project_root/
  README.md
  pyproject.toml
  requirements.txt
  configs/
    env/
      base.yaml
    algorithms/
      q_learning.yaml
      dqn.yaml
      a2c.yaml
      ppo.yaml
    experiments/
      scaling.yaml
      zero_shot.yaml
  src/
    hanoi_rl/
      envs/
        tower_of_hanoi_env.py
        encoding.py
        reward.py
      baselines/
        recursive_solver.py
        random_policy.py
      agents/
        tabular_q_learning.py
      training/
        train_tabular.py
        train_sb3.py
        evaluate.py
      analysis/
        metrics.py
        plots.py
        tables.py
      visualisation/
        manim_scenes.py
      utils/
        seeding.py
        io.py
        logging.py
  tests/
    test_env.py
    test_encoding.py
    test_reward.py
    test_solver.py
    test_tabular.py
    test_training_smoke.py
  outputs/
    models/
    logs/
    figures/
    tables/
    trajectories/
```

### 4.2 Configuration strategy

All experiments should be config-driven.

Each config should specify:

- disk count `n`;
- maximum disk count `N_max`;
- reward constants;
- episode step limit;
- algorithm name;
- algorithm hyperparameters;
- random seed;
- training budget;
- evaluation interval;
- output directory.

This is important for both reproducibility and dissertation reporting.

## 5. Environment Design

### 5.1 Internal state representation

Use a symbolic state vector of length `n` where entry `i` stores the peg index of disk `i`.

Recommended convention:

- disk `0` is the smallest disk;
- disk indices increase with disk size.

Example for `n = 4`:

- `[0, 0, 2, 1]`

Meaning:

- disk `0` is on peg `0`;
- disk `1` is on peg `0`;
- disk `2` is on peg `2`;
- disk `3` is on peg `1`.

This representation is compact, deterministic, and easy to convert into either tabular keys or neural observations.

### 5.2 Observation representation

Use a padded one-hot encoding for learning.

Recommended format:

- one-hot encode each disk across `3` peg positions;
- pad missing disk slots up to `N_max`;
- flatten to shape `N_max * 3`.

Example:

- if `N_max = 6` and the current puzzle size is `n = 4`, the last two disk slots are padding values rather than active disks.

Why this is preferable:

- fixed shape for SB3 policies;
- avoids treating peg indices as ordinal values;
- supports zero-shot evaluation on `n + 1`.

### 5.3 Action space

Use `Discrete(6)` corresponding to directed peg-to-peg moves:

- `0 -> 1`
- `0 -> 2`
- `1 -> 0`
- `1 -> 2`
- `2 -> 0`
- `2 -> 1`

This keeps the action space fixed for every disk count and for every algorithm.

### 5.4 Transition rules

For each action:

- identify the top disk on the source peg;
- reject the action if the source peg is empty;
- reject the action if the destination peg contains a smaller top disk;
- otherwise move the source top disk to the destination peg.

Invalid moves should leave the state unchanged and receive a penalty.

### 5.5 Episode termination

An episode ends when:

- all disks are on the target peg; or
- the agent reaches a step limit.

Recommended step limit:

- `max_steps = c * (2^n - 1)`

Recommended starting value:

- `c = 3`

Reason:

- it allows exploration;
- it still keeps failure measurable;
- it stops very poor policies from running indefinitely.

### 5.6 Reward design

The reward design should remain simple in the core study so that algorithm differences stay interpretable.

Recommended baseline rewards:

- `+20.0` on successful completion;
- `-0.1` per step;
- `-1.0` for an invalid move;
- no extra shaping in the core comparison.

Why these values are a better default than larger magnitudes:

- they keep returns numerically stable for standard RL libraries;
- successful episodes remain preferable even as `n` grows towards `6` or `7`;
- they preserve the shortest-path incentive without making the terminal reward irrelevant.

Important pilot check:

- confirm that an optimal solution remains clearly better than failing trajectories for the chosen `N_max`.

Contingency only if learning stalls badly:

- introduce a shaped-reward variant as an optional extension;
- keep it outside the main algorithm comparison.

### 5.7 Environment helpers

The environment should provide helpers for analysis and debugging:

- `get_valid_actions()`
- `is_goal_state()`
- `optimal_move_count(n)`
- `export_trajectory()`

These are useful for tests, plots, and animations, but the core agents should not rely on helper shortcuts during training.

## 6. Algorithm Design

### 6.1 Random policy

Purpose:

- establish a weak baseline;
- estimate invalid move frequency without learning;
- show that the task is non-trivial.

### 6.2 Recursive oracle solver

Purpose:

- provide the optimal move count;
- verify environment correctness;
- generate reference trajectories;
- support visualisation and efficiency-gap calculations.

### 6.3 Tabular Q-learning

Purpose:

- provide a classical RL baseline;
- validate the environment and reward structure on small `n`;
- show how far simple value iteration with exploration can go before state growth becomes prohibitive.

Recommended scope:

- include it in the main comparison for training on fixed `n`;
- expect strong results only for smaller puzzle sizes.

Important caveat:

- tabular Q-learning does not truly learn a size-agnostic policy;
- its zero-shot `n -> n + 1` behaviour is therefore not expected to be meaningful.

For the dissertation:

- include tabular Q-learning in the within-size comparison;
- treat its zero-shot result as a control, not as a serious transfer baseline.

### 6.4 DQN

Role in the study:

- main value-based neural baseline.

Why include it:

- well matched to a small discrete action space;
- common and recognisable in RL literature;
- natural comparison point against tabular Q-learning.

Implementation choice:

- use `stable-baselines3` DQN with a standard MLP policy.

### 6.5 A2C

Role in the study:

- lightweight on-policy baseline.

Why include it:

- simpler and cheaper than PPO;
- gives an on-policy comparison against DQN;
- may expose whether the task benefits more from replay-based learning or direct policy updates.

Implementation choice:

- use `stable-baselines3` A2C with the same observation encoding and comparable policy width.

### 6.6 PPO

Role in the study:

- stronger policy-gradient baseline.

Why include it:

- standard, well documented, and common in dissertations;
- offers a second on-policy point of comparison;
- likely to be more stable than A2C, though not necessarily more sample efficient.

Implementation choice:

- use `stable-baselines3` PPO with a standard MLP policy.

### 6.7 Common neural architecture policy

To keep the comparison fair, the neural algorithms should use:

- the same observation encoding;
- broadly similar MLP sizes where the library allows it;
- the same seed set;
- the same evaluation procedure.

Recommended starting network:

- two hidden layers of width `128`.

This is small enough for CPU work and large enough for a low-dimensional puzzle environment.

## 7. Comparison Protocol

### 7.1 Fairness rules

For `DQN`, `A2C`, and `PPO`:

- use the same observation shape;
- use the same reward function;
- use the same maximum-step rule;
- evaluate on the same set of disk counts;
- use the same number of seeds where possible;
- report both environment steps and wall-clock time.

### 7.2 Hyperparameter policy

Do not perform a huge search for one algorithm while leaving the others near defaults.

Recommended approach:

- begin from standard SB3 defaults;
- run a small pilot tuning pass on one intermediate task such as `n = 3` or `n = 4`;
- freeze a final configuration for each algorithm before the full comparison;
- avoid per-disk-count retuning unless the dissertation explicitly frames it as part of the study.

### 7.3 Seed policy

Recommended:

- `5` seeds if time allows;
- `3` seeds minimum for the neural baselines.

Report:

- mean;
- standard deviation;
- and, if possible, confidence intervals or bootstrap intervals for the main plots.

### 7.4 Training budget policy

Budgets should be fixed in environment steps, not just episodes, for the neural baselines.

Recommended process:

- use pilot runs to estimate how long each algorithm needs on `n = 3` and `n = 4`;
- scale the final budget conservatively so the full comparison remains feasible on local hardware;
- record wall-clock time as a practical metric, even if the main comparison is step-based.

The dissertation should state clearly that compute limits are part of the experimental design.

## 8. Experiment Plan

### 8.1 Phase 1: environment validation

Tasks:

- implement the environment;
- implement the recursive solver;
- verify legal and illegal moves;
- verify goal-state detection;
- verify that the oracle solver always achieves `2^n - 1` moves.

Deliverables:

- passing unit tests;
- a debug script for stepping through sample trajectories;
- saved oracle trajectories for small `n`.

### 8.2 Phase 2: pilot calibration

Tasks:

- run the random baseline;
- run tabular Q-learning on small `n`;
- run short DQN, A2C, and PPO pilots on `n = 3` or `n = 4`;
- confirm reward scaling and episode limits are sensible;
- choose the final training budgets and seed count.

Goal:

- ensure the full comparison is realistic before spending time on repeated runs.

### 8.3 Phase 3: within-size algorithm comparison

Train each algorithm on a fixed disk count `n` and evaluate on the same `n`.

Recommended disk counts:

- `n = 2, 3, 4, 5, 6`

Optional:

- `n = 7` only if Colab runs are practical and reproducible.

For each algorithm and each `n`, report:

- success rate;
- mean moves on successful episodes;
- efficiency gap;
- invalid move rate;
- episode return;
- environment steps to a target success threshold;
- wall-clock time.

### 8.4 Phase 4: breaking-point analysis

Define the breaking point for each algorithm as the smallest `n` where, within the fixed training budget:

- success rate stays below a threshold such as `90%`; or
- efficiency gap remains materially above zero; or
- performance becomes unstable across seeds.

Why this matters:

- the project is not only about whether agents solve small puzzles;
- it is about how performance degrades as the state space expands.

### 8.5 Phase 5: zero-shot `n -> n + 1` evaluation

For the neural baselines:

- train on `n`;
- freeze the model;
- evaluate directly on `n + 1`;
- do not update weights;
- report the same evaluation metrics used in the within-size study.

Recommended zero-shot pairs:

- `2 -> 3`
- `3 -> 4`
- `4 -> 5`
- `5 -> 6`

Optional:

- `6 -> 7` if `N_max = 7` is used.

Important reporting note:

- zero-shot results should be presented as a distinct section, not folded into the normal scaling plots.

For tabular Q-learning:

- include its zero-shot behaviour only as a control result;
- state explicitly that it does not possess a shared parametric policy and therefore is not expected to generalise.

### 8.6 Phase 6: optional extension

Only after the core study is complete, choose at most one extension:

- four-peg Tower of Hanoi;
- shaped reward ablation;
- action masking comparison;
- rule-variant environment.

Do not let the extension delay the core dissertation results.

## 9. Metrics and Analysis

### 9.1 Primary metrics

- **Success rate**: proportion of evaluation episodes that solve the puzzle.
- **Moves to solve**: number of steps taken in successful episodes.
- **Efficiency gap**: `moves_to_solve - (2^n - 1)`.
- **Invalid action rate**: invalid actions divided by total actions.
- **Episode return**: cumulative reward per episode.
- **Sample efficiency**: environment steps needed to hit a target success rate.

### 9.2 Secondary metrics

- wall-clock training time;
- variance across seeds;
- zero-shot success on `n + 1`;
- zero-shot efficiency gap on `n + 1`;
- failure modes near the breaking point.

### 9.3 Core plots

- learning curves for each algorithm;
- success rate versus disk count;
- efficiency gap versus disk count;
- invalid action rate versus disk count;
- sample efficiency comparison across algorithms;
- zero-shot transfer plots for `n -> n + 1`;
- breaking-point summary chart.

### 9.4 Tables to include

- final performance table by algorithm and `n`;
- zero-shot results table;
- training-budget and wall-clock summary table;
- hyperparameter summary table.

## 10. Visualisation Plan

Manim should be used only after the trajectory logging format is stable.

Recommended animations:

- optimal recursive solution for a small puzzle;
- learned successful policy from the best RL agent;
- a failed trajectory near the breaking point;
- optional compact state-space visual for a very small `n`.

Implementation rule:

- save trajectories to JSON or CSV;
- render animations offline from saved trajectories;
- do not couple Manim directly to the training loop.

## 11. Testing and Reproducibility

### 11.1 Tests

Minimum test coverage should include:

- legal move generation;
- illegal move handling;
- reward assignment;
- goal-state detection;
- step-limit termination;
- padded observation encoding;
- recursive solver correctness;
- one smoke test for each training entry point.

### 11.2 Reproducibility controls

- fixed seeds for Python, NumPy, PyTorch, and Gymnasium;
- config files checked into the repository;
- separate training and evaluation scripts;
- saved checkpoints with metadata;
- per-seed output directories;
- versioned figures and tables.

## 12. Dissertation-Oriented Mapping

### 12.1 Chapter 1: Introduction

Use this chapter to define:

- Tower of Hanoi rules and objective;
- optimal solution length `2^n - 1`;
- motivation for RL rather than hard-coded recursion;
- research questions and project scope.

### 12.2 Chapter 2: Methods

This chapter should map directly onto the engineering design:

- MDP formulation;
- state, action, and reward definitions;
- Gymnasium environment design;
- algorithm selection: Q-learning, DQN, A2C, PPO;
- observation-padding strategy for zero-shot evaluation;
- training and evaluation protocol;
- metric definitions.

### 12.3 Chapter 3: Results

This chapter should present:

- environment validation summary;
- within-size comparison results;
- scaling curves;
- breaking-point findings;
- zero-shot `n -> n + 1` results;
- representative visualisations.

### 12.4 Chapter 4: Evaluation

This chapter should discuss:

- which algorithm family worked best and why;
- why some algorithms failed to scale;
- the limits of model-free RL on a structured puzzle;
- whether zero-shot generalisation genuinely occurred;
- limitations caused by compute budget and environment design;
- optional extension results, if included.

### 12.5 Suggested dissertation claims to target

- RL can learn efficient policies for small Tower of Hanoi instances.
- Performance degrades sharply beyond a problem-size threshold.
- Neural methods may scale further than tabular methods, but not necessarily elegantly.
- Zero-shot transfer to `n + 1` remains difficult even with fixed-size observations.

## 13. Milestone Plan

### Week 1

- set up repository, dependencies, and test harness;
- implement the recursive solver;
- finalise environment and observation specifications.

### Week 2

- implement the Gymnasium environment;
- implement reward logic and padded observation encoding;
- write unit tests.

### Week 3

- implement random policy and tabular Q-learning;
- validate solver, rewards, and metrics on small `n`;
- add trajectory export.

### Week 4

- integrate SB3 training for DQN, A2C, and PPO;
- run smoke tests for each algorithm;
- establish logging and checkpoint structure.

### Week 5

- perform pilot tuning on `n = 3` or `n = 4`;
- finalise reward constants, budgets, and seed count;
- freeze the main comparison protocol.

### Week 6

- run the within-size comparison on lower disk counts;
- generate first learning curves and debug plots.

### Week 7

- complete the main scaling runs up to the chosen `N_max`;
- begin breaking-point analysis.

### Week 8

- run zero-shot `n -> n + 1` evaluations;
- assemble transfer plots and tables.

### Week 9

- clean up figures and tables;
- create representative Manim animations from saved trajectories.

### Week 10

- repeat any weak or noisy runs;
- compute summary statistics;
- write the first full results section.

### Weeks 11-12

- finish dissertation chapters;
- refine evaluation and limitation discussion;
- add optional extension only if the core work is complete.

## 14. Risks and Mitigations

### Risk 1: reward design favours one algorithm unfairly

Mitigation:

- keep the core reward simple;
- use the same reward function across the comparison;
- separate any shaped-reward study into an extension.

### Risk 2: zero-shot evaluation becomes invalid because of observation mismatch

Mitigation:

- use a fixed `N_max` padded observation design from the start.

### Risk 3: local CPU runs are too slow for repeated neural experiments

Mitigation:

- cap the core study at `N_max = 6`;
- use Colab only for final repeated runs or an optional `N_max = 7`.

### Risk 4: the comparison becomes unfair due to uneven tuning effort

Mitigation:

- apply one small pilot-tuning stage;
- freeze configurations before the full comparison.

### Risk 5: tabular Q-learning dominates space in the write-up despite limited scaling value

Mitigation:

- present it as a small-`n` baseline;
- keep the main narrative centred on the neural comparison and zero-shot results.

### Risk 6: Manim consumes too much time

Mitigation:

- make animation strictly post-processing;
- limit the number of scenes to a few high-value examples.

## 15. Immediate Next Steps

1. Set up the repository and dependency files.
2. Implement the recursive solver and environment first.
3. Lock the padded observation design before training any neural agent.
4. Implement tabular Q-learning to validate the environment on small `n`.
5. Integrate SB3 for DQN, A2C, and PPO.
6. Run pilot experiments to finalise budgets and seeds.
