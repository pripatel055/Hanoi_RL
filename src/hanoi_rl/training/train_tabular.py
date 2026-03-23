"""Training entry point for the tabular Q-learning baseline."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from hanoi_rl.agents import TabularQAgent, TabularQConfig
from hanoi_rl.analysis.metrics import efficiency_gap
from hanoi_rl.envs import EnvConfig, TowerOfHanoiEnv
from hanoi_rl.training.common import build_env_config
from hanoi_rl.utils.io import load_yaml
from hanoi_rl.utils.io import save_json
from hanoi_rl.utils.logging import configure_logging
from hanoi_rl.utils.seeding import set_global_seeds


@dataclass
class EpisodeResult:
    """Per-episode metrics recorded during training or evaluation."""

    episode: int
    reward: float
    steps: int
    success: bool
    truncated: bool
    invalid_moves: int
    invalid_action_rate: float
    efficiency_gap: int | None
    epsilon: float | None = None

def build_agent_config(config_data: dict[str, Any]) -> tuple[TabularQConfig, int, int, int]:
    """Create a TabularQConfig and training settings from YAML data."""
    agent_config = TabularQConfig(
        alpha=float(config_data.get("alpha", 0.1)),
        gamma=float(config_data.get("gamma", 0.99)),
        epsilon_start=float(config_data.get("epsilon_start", 1.0)),
        epsilon_end=float(config_data.get("epsilon_end", 0.05)),
        epsilon_decay=float(config_data.get("epsilon_decay", 0.995)),
        action_size=int(config_data.get("action_size", 6)),
        seed=int(config_data["seed"]) if config_data.get("seed") is not None else None,
    )
    episodes = int(config_data.get("episodes", 5000))
    log_interval = int(config_data.get("log_interval", 250))
    evaluation_episodes = int(config_data.get("evaluation_episodes", 100))
    return agent_config, episodes, log_interval, evaluation_episodes


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(description="Train a tabular Q-learning baseline.")
    parser.add_argument("--env-config", default="configs/env/base.yaml")
    parser.add_argument("--agent-config", default="configs/algorithms/q_learning.yaml")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--eval-episodes", type=int, default=None)
    parser.add_argument("--log-interval", type=int, default=None)
    parser.add_argument("--output-dir", default="outputs/logs/tabular_q_learning")
    parser.add_argument("--run-name", default=None)
    return parser


def summarise_episode_results(
    episode_results: list[EpisodeResult],
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


def run_episode(
    env: TowerOfHanoiEnv,
    agent: TabularQAgent,
    *,
    episode_index: int,
    greedy: bool,
    update_agent: bool,
    seed: int | None,
) -> EpisodeResult:
    """Run one training or evaluation episode."""
    _, info = env.reset(seed=seed)
    state = info["state"]
    terminated = False
    truncated = False
    total_reward = 0.0
    invalid_moves = 0

    while not (terminated or truncated):
        action = agent.greedy_action(state) if greedy else agent.select_action(state)
        _, reward, terminated, truncated, info = env.step(action)
        next_state = info["state"]

        if update_agent:
            agent.update(
                state,
                action,
                reward,
                next_state,
                done=terminated or truncated,
            )

        total_reward += reward
        invalid_moves += int(info["invalid_move"])
        state = next_state

    steps = int(info["step_count"])
    success = bool(terminated)
    result = EpisodeResult(
        episode=episode_index,
        reward=total_reward,
        steps=steps,
        success=success,
        truncated=bool(truncated),
        invalid_moves=invalid_moves,
        invalid_action_rate=(invalid_moves / steps) if steps else 0.0,
        efficiency_gap=efficiency_gap(steps, env.config.n_disks) if success else None,
        epsilon=None if greedy else agent.epsilon,
    )

    if update_agent:
        agent.decay_epsilon()
        result.epsilon = agent.epsilon

    return result


def evaluate_agent(
    agent: TabularQAgent,
    env_config: EnvConfig,
    *,
    episodes: int,
    base_seed: int | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate a trained agent greedily on the configured environment."""
    env = TowerOfHanoiEnv(env_config)
    episode_results = [
        run_episode(
            env,
            agent,
            episode_index=episode_index,
            greedy=True,
            update_agent=False,
            seed=None if base_seed is None else base_seed + episode_index,
        )
        for episode_index in range(1, episodes + 1)
    ]
    env.close()
    return (
        summarise_episode_results(episode_results, n_disks=env_config.n_disks),
        [asdict(result) for result in episode_results],
    )


def serialise_q_table(agent: TabularQAgent) -> dict[str, list[float]]:
    """Convert the learned Q-table into a JSON-friendly format."""
    return {
        ",".join(str(value) for value in state_key): q_values.tolist()
        for state_key, q_values in agent.q_values.items()
    }


def train_tabular_agent(
    env_config: EnvConfig,
    agent_config: TabularQConfig,
    *,
    episodes: int,
    evaluation_episodes: int,
    log_interval: int,
    output_dir: str | Path | None = None,
    run_name: str | None = None,
) -> dict[str, Any]:
    """Train tabular Q-learning and optionally persist run artefacts."""
    logger = configure_logging()
    if log_interval <= 0:
        raise ValueError("log_interval must be positive")
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if evaluation_episodes <= 0:
        raise ValueError("evaluation_episodes must be positive")

    set_global_seeds(agent_config.seed or 0)
    agent = TabularQAgent(agent_config)
    env = TowerOfHanoiEnv(env_config)
    training_results: list[EpisodeResult] = []

    logger.info(
        "Starting tabular Q-learning: n=%s, episodes=%s, eval_episodes=%s, seed=%s",
        env_config.n_disks,
        episodes,
        evaluation_episodes,
        agent_config.seed,
    )

    for episode_index in range(1, episodes + 1):
        episode_result = run_episode(
            env,
            agent,
            episode_index=episode_index,
            greedy=False,
            update_agent=True,
            seed=None if agent_config.seed is None else agent_config.seed + episode_index,
        )
        training_results.append(episode_result)

        if episode_index % log_interval == 0 or episode_index == episodes:
            window = training_results[-min(log_interval, len(training_results)) :]
            window_summary = summarise_episode_results(window, n_disks=env_config.n_disks)
            logger.info(
                "Episode %s/%s | success_rate=%.3f | mean_reward=%.3f | "
                "mean_steps=%.2f | epsilon=%.4f",
                episode_index,
                episodes,
                window_summary["success_rate"],
                window_summary["mean_reward"],
                window_summary["mean_steps"],
                agent.epsilon,
            )

    env.close()

    evaluation_summary, evaluation_history = evaluate_agent(
        agent,
        env_config,
        episodes=evaluation_episodes,
        base_seed=None if agent_config.seed is None else agent_config.seed + 10_000,
    )

    training_summary = summarise_episode_results(training_results, n_disks=env_config.n_disks)
    run_summary = {
        "run_name": run_name or f"q_learning_n{env_config.n_disks}_seed{agent_config.seed or 0}",
        "environment": asdict(env_config),
        "agent": asdict(agent_config),
        "training": training_summary,
        "evaluation": evaluation_summary,
        "q_table_size": len(agent.q_values),
        "final_epsilon": agent.epsilon,
    }

    result = {
        "summary": run_summary,
        "training_history": [asdict(result) for result in training_results],
        "evaluation_history": evaluation_history,
        "q_table": serialise_q_table(agent),
    }

    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        run_stem = run_summary["run_name"]
        save_json(run_summary, output_path / f"{run_stem}_summary.json")
        save_json(result["training_history"], output_path / f"{run_stem}_training_history.json")
        save_json(result["evaluation_history"], output_path / f"{run_stem}_evaluation_history.json")
        save_json(result["q_table"], output_path / f"{run_stem}_q_table.json")

    return result


def main() -> None:
    """Train and evaluate the tabular Q-learning baseline."""
    parser = build_parser()
    args = parser.parse_args()

    env_config_data = load_yaml(args.env_config)
    agent_config_data = load_yaml(args.agent_config)

    env_config = build_env_config(env_config_data)
    agent_config, config_episodes, config_log_interval, config_eval_episodes = build_agent_config(
        agent_config_data
    )

    episodes = args.episodes if args.episodes is not None else config_episodes
    evaluation_episodes = (
        args.eval_episodes if args.eval_episodes is not None else config_eval_episodes
    )
    log_interval = args.log_interval if args.log_interval is not None else config_log_interval
    run_name = args.run_name or f"q_learning_n{env_config.n_disks}_seed{agent_config.seed or 0}"

    result = train_tabular_agent(
        env_config,
        agent_config,
        episodes=episodes,
        evaluation_episodes=evaluation_episodes,
        log_interval=log_interval,
        output_dir=args.output_dir,
        run_name=run_name,
    )

    print("Tabular Q-learning training complete.")
    print(f"Run name: {result['summary']['run_name']}")
    print(f"Training success rate: {result['summary']['training']['success_rate']:.3f}")
    print(f"Evaluation success rate: {result['summary']['evaluation']['success_rate']:.3f}")
    print(f"Q-table size: {result['summary']['q_table_size']}")


if __name__ == "__main__":
    main()
