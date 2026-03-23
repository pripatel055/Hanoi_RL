"""Run small reproducible SB3 pilot sweeps and summarise the results."""

from __future__ import annotations

import argparse
import copy
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from hanoi_rl.training.train_sb3 import train_sb3_agent
from hanoi_rl.utils.io import load_yaml, save_json
from hanoi_rl.utils.logging import configure_logging


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for pilot sweeps."""
    parser = argparse.ArgumentParser(description="Run SB3 pilot sweeps for Tower of Hanoi.")
    parser.add_argument("--pilot-config", default="configs/experiments/pilot_sb3.yaml")
    parser.add_argument("--env-config", default="configs/env/base.yaml")
    return parser


def summarise_rows(
    rows: list[dict[str, Any]],
    *,
    success_threshold: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Aggregate pilot rows and derive per-problem recommendations."""
    grouped: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["algorithm"], row["n_disks"], row["total_timesteps"])].append(row)

    aggregates: list[dict[str, Any]] = []
    for (algorithm, n_disks, total_timesteps), group_rows in sorted(grouped.items()):
        success_rates = [float(row["success_rate"]) for row in group_rows]
        mean_steps_values = [float(row["mean_steps"]) for row in group_rows]
        efficiency_gaps = [
            float(row["mean_efficiency_gap"])
            for row in group_rows
            if row["mean_efficiency_gap"] is not None
        ]
        aggregates.append(
            {
                "algorithm": algorithm,
                "n_disks": n_disks,
                "total_timesteps": total_timesteps,
                "seeds": [int(row["seed"]) for row in group_rows],
                "mean_success_rate": mean(success_rates),
                "min_success_rate": min(success_rates),
                "mean_steps": mean(mean_steps_values),
                "mean_efficiency_gap": mean(efficiency_gaps) if efficiency_gaps else None,
                "stable_success": min(success_rates) >= success_threshold,
            }
        )

    recommendations: list[dict[str, Any]] = []
    grouped_by_problem: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for aggregate in aggregates:
        grouped_by_problem[(aggregate["algorithm"], aggregate["n_disks"])].append(aggregate)

    for (algorithm, n_disks), problem_rows in sorted(grouped_by_problem.items()):
        stable_rows = [row for row in problem_rows if row["stable_success"]]
        best_row = min(stable_rows, key=lambda row: row["total_timesteps"]) if stable_rows else max(
            problem_rows, key=lambda row: row["mean_success_rate"]
        )
        recommendations.append(
            {
                "algorithm": algorithm,
                "n_disks": n_disks,
                "recommended_total_timesteps": best_row["total_timesteps"],
                "stable_success_found": bool(stable_rows),
                "mean_success_rate": best_row["mean_success_rate"],
                "min_success_rate": best_row["min_success_rate"],
                "mean_steps": best_row["mean_steps"],
                "mean_efficiency_gap": best_row["mean_efficiency_gap"],
            }
        )

    return aggregates, recommendations


def render_markdown(
    pilot_config: dict[str, Any],
    aggregates: list[dict[str, Any]],
    recommendations: list[dict[str, Any]],
) -> str:
    """Render a concise markdown summary for the pilot runs."""
    lines = [
        "# SB3 Pilot Results",
        "",
        "## Pilot setup",
        "",
        f"- Algorithms: {', '.join(pilot_config['algorithms'])}",
        f"- Disk counts: {', '.join(str(value) for value in pilot_config['disk_counts'])}",
        f"- Budgets: {', '.join(str(value) for value in pilot_config['total_timesteps'])}",
        f"- Seeds: {', '.join(str(value) for value in pilot_config['seeds'])}",
        f"- Evaluation episodes: {pilot_config['evaluation_episodes']}",
        f"- Stable success threshold: {pilot_config['success_threshold']:.2f}",
        "",
        "## Aggregated results",
        "",
        "| Algorithm | n | Timesteps | Mean success | Min success | Mean steps | Mean efficiency gap | Stable success |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]

    for row in aggregates:
        mean_efficiency_gap = (
            f"{row['mean_efficiency_gap']:.2f}"
            if row["mean_efficiency_gap"] is not None
            else "N/A"
        )
        lines.append(
            "| "
            f"{row['algorithm']} | {row['n_disks']} | {row['total_timesteps']} | "
            f"{row['mean_success_rate']:.2f} | {row['min_success_rate']:.2f} | "
            f"{row['mean_steps']:.2f} | {mean_efficiency_gap} | "
            f"{'Yes' if row['stable_success'] else 'No'} |"
        )

    lines.extend(
        [
            "",
            "## Provisional frozen protocol",
            "",
            "Use the following first-pass budgets for the main comparison on these small problems:",
            "",
            "| Algorithm | n | Recommended timesteps | Stable success found |",
            "| --- | ---: | ---: | --- |",
        ]
    )

    for row in recommendations:
        lines.append(
            "| "
            f"{row['algorithm']} | {row['n_disks']} | {row['recommended_total_timesteps']} | "
            f"{'Yes' if row['stable_success_found'] else 'No'} |"
        )

    lines.extend(
        [
            "",
            "Interpretation:",
            "- If stable success was found, use the smallest successful pilot budget as the first main-run budget.",
            "- If stable success was not found, treat the recommendation as the best observed pilot budget and expect that larger runs may be required.",
            "- Do not start `n >= 4` runs until these small-`n` budgets are accepted as the provisional comparison baseline.",
        ]
    )

    return "\n".join(lines) + "\n"


def main() -> None:
    """Run the configured SB3 pilot sweep and persist raw plus summarised outputs."""
    parser = build_parser()
    args = parser.parse_args()

    logger = configure_logging()
    pilot_config = load_yaml(args.pilot_config)
    base_env_config = load_yaml(args.env_config)
    output_dir = Path(str(pilot_config.get("output_dir", "outputs/logs/pilot_sb3")))
    runs_dir = output_dir / "runs"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for algorithm in pilot_config["algorithms"]:
        base_agent_config = load_yaml(f"configs/algorithms/{algorithm}.yaml")
        for n_disks in pilot_config["disk_counts"]:
            env_config = copy.deepcopy(base_env_config)
            env_config["n_disks"] = int(n_disks)
            env_config["n_max"] = max(int(env_config.get("n_max", n_disks)), int(n_disks))

            for total_timesteps in pilot_config["total_timesteps"]:
                for seed in pilot_config["seeds"]:
                    agent_config = copy.deepcopy(base_agent_config)
                    agent_config["seed"] = int(seed)
                    agent_config["total_timesteps"] = int(total_timesteps)
                    agent_config["evaluation_episodes"] = int(pilot_config["evaluation_episodes"])
                    agent_config["eval_freq"] = min(
                        int(pilot_config["eval_freq"]),
                        int(total_timesteps),
                    )

                    run_name = f"{algorithm}_n{n_disks}_t{total_timesteps}_s{seed}"
                    logger.info("Pilot run %s", run_name)
                    result = train_sb3_agent(
                        env_config,
                        agent_config,
                        output_dir=runs_dir,
                        run_name=run_name,
                    )
                    evaluation = result["summary"]["evaluation"]
                    rows.append(
                        {
                            "run_name": run_name,
                            "algorithm": algorithm,
                            "n_disks": int(n_disks),
                            "total_timesteps": int(total_timesteps),
                            "seed": int(seed),
                            "success_rate": float(evaluation["success_rate"]),
                            "mean_steps": float(evaluation["mean_steps"]),
                            "mean_efficiency_gap": (
                                None
                                if evaluation["mean_efficiency_gap"] is None
                                else float(evaluation["mean_efficiency_gap"])
                            ),
                        }
                    )

    aggregates, recommendations = summarise_rows(
        rows,
        success_threshold=float(pilot_config["success_threshold"]),
    )
    markdown = render_markdown(pilot_config, aggregates, recommendations)

    save_json(rows, output_dir / "pilot_rows.json")
    save_json(aggregates, output_dir / "pilot_aggregates.json")
    save_json(recommendations, output_dir / "pilot_recommendations.json")
    (output_dir / "SB3_PILOT_RESULTS.md").write_text(markdown, encoding="utf-8")

    print("Pilot sweep complete.")
    print(f"Results written to: {output_dir}")
    for recommendation in recommendations:
        print(
            f"{recommendation['algorithm']} n={recommendation['n_disks']}: "
            f"{recommendation['recommended_total_timesteps']} timesteps "
            f"(stable={recommendation['stable_success_found']})"
        )


if __name__ == "__main__":
    main()
