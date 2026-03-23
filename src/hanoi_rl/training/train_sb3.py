"""Training entry point for Stable-Baselines3 experiments."""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from hanoi_rl.envs import TowerOfHanoiEnv
from hanoi_rl.training.common import build_env_config
from hanoi_rl.training.common import evaluate_policy
from hanoi_rl.utils.io import load_yaml
from hanoi_rl.utils.io import save_json
from hanoi_rl.utils.logging import configure_logging
from hanoi_rl.utils.seeding import set_global_seeds

SB3_ALGORITHMS: dict[str, type[BaseAlgorithm]] = {
    "dqn": DQN,
    "a2c": A2C,
    "ppo": PPO,
}


def build_agent_settings(config_data: dict[str, Any]) -> tuple[str, str, int, int, int, int | None, dict[str, Any]]:
    """Parse SB3 algorithm settings from YAML data."""
    algorithm_name = str(config_data["name"]).lower()
    if algorithm_name not in SB3_ALGORITHMS:
        raise ValueError(f"unsupported SB3 algorithm: {algorithm_name}")

    policy = str(config_data.get("policy", "MlpPolicy"))
    total_timesteps = int(config_data.get("total_timesteps", 100_000))
    evaluation_episodes = int(config_data.get("evaluation_episodes", 50))
    eval_freq = int(config_data.get("eval_freq", 5_000))
    seed = int(config_data["seed"]) if config_data.get("seed") is not None else None

    reserved = {"name", "policy", "total_timesteps", "evaluation_episodes", "eval_freq", "seed"}
    model_kwargs = {
        key: copy.deepcopy(value) for key, value in config_data.items() if key not in reserved
    }
    return algorithm_name, policy, total_timesteps, evaluation_episodes, eval_freq, seed, model_kwargs


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(description="Train a Stable-Baselines3 agent.")
    parser.add_argument("--env-config", default="configs/env/base.yaml")
    parser.add_argument("--agent-config", required=True)
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--eval-episodes", type=int, default=None)
    parser.add_argument("--eval-freq", type=int, default=None)
    parser.add_argument("--output-dir", default="outputs/models/sb3")
    parser.add_argument("--run-name", default=None)
    return parser


def parse_eval_callback_results(path: Path) -> list[dict[str, float]]:
    """Parse SB3 EvalCallback output into JSON-friendly records."""
    if not path.exists():
        return []

    data = np.load(path, allow_pickle=True)
    timesteps = data["timesteps"].tolist()
    results = data["results"].tolist()
    ep_lengths = data["ep_lengths"].tolist()

    return [
        {
            "timesteps": int(timestep),
            "mean_reward": float(np.mean(reward_list)),
            "mean_episode_length": float(np.mean(length_list)),
        }
        for timestep, reward_list, length_list in zip(timesteps, results, ep_lengths, strict=True)
    ]


def train_sb3_agent(
    env_config_data: dict[str, Any],
    agent_config_data: dict[str, Any],
    *,
    total_timesteps: int | None = None,
    evaluation_episodes: int | None = None,
    eval_freq: int | None = None,
    output_dir: str | Path | None = None,
    run_name: str | None = None,
) -> dict[str, Any]:
    """Train and evaluate an SB3 algorithm on the Tower of Hanoi environment."""
    logger = configure_logging()
    env_config = build_env_config(env_config_data)
    serialisable_agent_config = copy.deepcopy(agent_config_data)
    (
        algorithm_name,
        policy,
        config_total_timesteps,
        config_evaluation_episodes,
        config_eval_freq,
        seed,
        model_kwargs,
    ) = build_agent_settings(agent_config_data)

    run_total_timesteps = total_timesteps if total_timesteps is not None else config_total_timesteps
    run_evaluation_episodes = (
        evaluation_episodes if evaluation_episodes is not None else config_evaluation_episodes
    )
    run_eval_freq = eval_freq if eval_freq is not None else config_eval_freq

    if run_total_timesteps <= 0:
        raise ValueError("total_timesteps must be positive")
    if run_evaluation_episodes <= 0:
        raise ValueError("evaluation_episodes must be positive")
    if run_eval_freq <= 0:
        raise ValueError("eval_freq must be positive")

    set_global_seeds(seed or 0)

    algorithm_class = SB3_ALGORITHMS[algorithm_name]
    run_stem = run_name or f"{algorithm_name}_n{env_config.n_disks}_seed{seed or 0}"
    output_path = Path(output_dir) if output_dir is not None else None

    train_env = Monitor(TowerOfHanoiEnv(env_config))
    eval_env = Monitor(TowerOfHanoiEnv(env_config))

    callback = None
    eval_curve = []
    if output_path is not None:
        output_path.mkdir(parents=True, exist_ok=True)
        callback = EvalCallback(
            eval_env,
            best_model_save_path=str(output_path / f"{run_stem}_best"),
            log_path=str(output_path / f"{run_stem}_eval"),
            eval_freq=run_eval_freq,
            n_eval_episodes=run_evaluation_episodes,
            deterministic=True,
            verbose=0,
        )

    logger.info(
        "Starting SB3 training: algorithm=%s, n=%s, timesteps=%s, eval_episodes=%s, seed=%s",
        algorithm_name,
        env_config.n_disks,
        run_total_timesteps,
        run_evaluation_episodes,
        seed,
    )

    model = algorithm_class(
        policy,
        train_env,
        seed=seed,
        verbose=0,
        **model_kwargs,
    )
    model.learn(total_timesteps=run_total_timesteps, callback=callback, progress_bar=False)

    evaluation_summary, evaluation_history = evaluate_policy(
        model,
        env_config,
        episodes=run_evaluation_episodes,
        deterministic=True,
        base_seed=None if seed is None else seed + 10_000,
    )

    model_path_str: str | None = None
    if output_path is not None:
        model_prefix = output_path / f"{run_stem}_model"
        model.save(str(model_prefix))
        model_path_str = str(model_prefix.with_suffix(".zip"))
        eval_curve = parse_eval_callback_results(output_path / f"{run_stem}_eval" / "evaluations.npz")

    summary = {
        "run_name": run_stem,
        "algorithm": algorithm_name,
        "policy": policy,
        "environment": asdict(env_config),
        "agent_config": serialisable_agent_config,
        "total_timesteps": run_total_timesteps,
        "evaluation_episodes": run_evaluation_episodes,
        "eval_freq": run_eval_freq,
        "seed": seed,
        "evaluation": evaluation_summary,
        "model_path": model_path_str,
    }

    result = {
        "summary": summary,
        "evaluation_history": evaluation_history,
        "evaluation_curve": eval_curve,
    }

    if output_path is not None:
        save_json(summary, output_path / f"{run_stem}_summary.json")
        save_json(evaluation_history, output_path / f"{run_stem}_evaluation_history.json")
        save_json(eval_curve, output_path / f"{run_stem}_evaluation_curve.json")

    train_env.close()
    eval_env.close()

    return result


def main() -> None:
    """Train and evaluate a Stable-Baselines3 agent."""
    parser = build_parser()
    args = parser.parse_args()

    env_config_data = load_yaml(args.env_config)
    agent_config_data = load_yaml(args.agent_config)
    result = train_sb3_agent(
        env_config_data,
        agent_config_data,
        total_timesteps=args.total_timesteps,
        evaluation_episodes=args.eval_episodes,
        eval_freq=args.eval_freq,
        output_dir=args.output_dir,
        run_name=args.run_name,
    )

    print("Stable-Baselines3 training complete.")
    print(f"Run name: {result['summary']['run_name']}")
    print(f"Algorithm: {result['summary']['algorithm']}")
    print(f"Evaluation success rate: {result['summary']['evaluation']['success_rate']:.3f}")
    print(f"Mean evaluation steps: {result['summary']['evaluation']['mean_steps']:.2f}")


if __name__ == "__main__":
    main()
