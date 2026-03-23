"""Evaluation entry point for trained policies and baselines."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from stable_baselines3 import A2C, DQN, PPO

from hanoi_rl.training.common import build_env_config
from hanoi_rl.training.common import evaluate_policy
from hanoi_rl.utils.io import load_yaml
from hanoi_rl.utils.io import save_json

SB3_ALGORITHMS = {
    "dqn": DQN,
    "a2c": A2C,
    "ppo": PPO,
}


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(description="Evaluate a trained policy or baseline.")
    parser.add_argument("--env-config", default="configs/env/base.yaml")
    parser.add_argument("--agent-config", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--eval-episodes", type=int, default=None)
    parser.add_argument("--output-path", default=None)
    return parser


def evaluate_saved_sb3_model(
    env_config_data: dict[str, Any],
    agent_config_data: dict[str, Any],
    *,
    model_path: str | Path,
    evaluation_episodes: int | None = None,
) -> dict[str, Any]:
    """Load and evaluate a saved SB3 model."""
    env_config = build_env_config(env_config_data)
    algorithm_name = str(agent_config_data["name"]).lower()
    if algorithm_name not in SB3_ALGORITHMS:
        raise ValueError(f"unsupported SB3 algorithm: {algorithm_name}")

    algorithm_class = SB3_ALGORITHMS[algorithm_name]
    model = algorithm_class.load(str(model_path))
    run_evaluation_episodes = int(
        evaluation_episodes
        if evaluation_episodes is not None
        else agent_config_data.get("evaluation_episodes", 50)
    )

    summary, history = evaluate_policy(
        model,
        env_config,
        episodes=run_evaluation_episodes,
        deterministic=True,
        base_seed=(
            None
            if agent_config_data.get("seed") is None
            else int(agent_config_data["seed"]) + 20_000
        ),
    )
    return {
        "algorithm": algorithm_name,
        "model_path": str(model_path),
        "environment": env_config_data,
        "evaluation": summary,
        "history": history,
    }


def main() -> None:
    """Evaluate a saved SB3 policy."""
    parser = build_parser()
    args = parser.parse_args()

    env_config_data = load_yaml(args.env_config)
    agent_config_data = load_yaml(args.agent_config)
    result = evaluate_saved_sb3_model(
        env_config_data,
        agent_config_data,
        model_path=args.model_path,
        evaluation_episodes=args.eval_episodes,
    )

    if args.output_path is not None:
        save_json(result, args.output_path)

    print("Evaluation complete.")
    print(f"Algorithm: {result['algorithm']}")
    print(f"Evaluation success rate: {result['evaluation']['success_rate']:.3f}")
    print(f"Mean evaluation steps: {result['evaluation']['mean_steps']:.2f}")


if __name__ == "__main__":
    main()
