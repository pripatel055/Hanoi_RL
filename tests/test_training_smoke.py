from hanoi_rl.agents import TabularQConfig
from hanoi_rl.envs import EnvConfig, RewardConfig
from hanoi_rl.training.evaluate import build_parser as build_evaluate_parser
from hanoi_rl.training.train_sb3 import train_sb3_agent
from hanoi_rl.training.train_sb3 import build_parser as build_sb3_parser
from hanoi_rl.training.train_tabular import build_parser as build_tabular_parser
from hanoi_rl.training.train_tabular import train_tabular_agent


def test_training_entry_points_expose_parsers() -> None:
    assert build_tabular_parser().prog
    assert build_sb3_parser().prog
    assert build_evaluate_parser().prog


def test_tabular_training_smoke_run_writes_outputs(tmp_path) -> None:
    env_config = EnvConfig(
        n_disks=2,
        n_max=2,
        step_limit_multiplier=2,
        reward=RewardConfig(goal_reward=10.0, step_penalty=-0.1, invalid_move_penalty=-1.0),
    )
    agent_config = TabularQConfig(
        alpha=0.2,
        gamma=0.95,
        epsilon_start=0.8,
        epsilon_end=0.1,
        epsilon_decay=0.9,
        seed=5,
    )

    result = train_tabular_agent(
        env_config,
        agent_config,
        episodes=10,
        evaluation_episodes=4,
        log_interval=5,
        output_dir=tmp_path,
        run_name="smoke",
    )

    assert result["summary"]["run_name"] == "smoke"
    assert result["summary"]["training"]["episodes"] == 10
    assert result["summary"]["evaluation"]["episodes"] == 4
    assert result["summary"]["q_table_size"] > 0
    assert (tmp_path / "smoke_summary.json").exists()
    assert (tmp_path / "smoke_training_history.json").exists()
    assert (tmp_path / "smoke_evaluation_history.json").exists()
    assert (tmp_path / "smoke_q_table.json").exists()


def test_sb3_training_smoke_run_writes_outputs(tmp_path) -> None:
    env_config = {
        "n_disks": 1,
        "n_max": 1,
        "step_limit_multiplier": 2,
        "reward": {
            "goal_reward": 10.0,
            "step_penalty": -0.1,
            "invalid_move_penalty": -1.0,
        },
    }
    agent_config = {
        "name": "a2c",
        "policy": "MlpPolicy",
        "learning_rate": 0.0007,
        "n_steps": 2,
        "gamma": 0.99,
        "gae_lambda": 1.0,
        "ent_coef": 0.0,
        "vf_coef": 0.5,
        "total_timesteps": 20,
        "evaluation_episodes": 2,
        "eval_freq": 5,
        "seed": 7,
        "policy_kwargs": {"net_arch": [16, 16]},
    }

    result = train_sb3_agent(
        env_config,
        agent_config,
        output_dir=tmp_path,
        run_name="a2c_smoke",
    )

    assert result["summary"]["run_name"] == "a2c_smoke"
    assert result["summary"]["algorithm"] == "a2c"
    assert result["summary"]["evaluation"]["episodes"] == 2
    assert (tmp_path / "a2c_smoke_summary.json").exists()
    assert (tmp_path / "a2c_smoke_evaluation_history.json").exists()
    assert (tmp_path / "a2c_smoke_evaluation_curve.json").exists()
    assert (tmp_path / "a2c_smoke_model.zip").exists()
