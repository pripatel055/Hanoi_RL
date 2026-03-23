import pytest

pytest.importorskip("gymnasium")

from hanoi_rl.envs import EnvConfig, TowerOfHanoiEnv


def test_env_reset_returns_padded_observation() -> None:
    env = TowerOfHanoiEnv(EnvConfig(n_disks=3, n_max=6))
    observation, info = env.reset()

    assert observation.shape == (18,)
    assert info["state"] == (0, 0, 0)
    assert info["terminated"] is False
    assert info["truncated"] is False
    assert len(info["valid_actions"]) == 2


def test_env_step_applies_valid_move() -> None:
    env = TowerOfHanoiEnv(EnvConfig(n_disks=3, n_max=6))
    env.reset()

    _, reward, terminated, truncated, info = env.step(0)

    assert reward == -0.1
    assert not terminated
    assert not truncated
    assert info["state"] == (1, 0, 0)
    assert info["invalid_move"] is False
    assert info["valid_actions"] == [1, 2, 3]


def test_env_rejects_move_from_empty_source() -> None:
    env = TowerOfHanoiEnv(EnvConfig(n_disks=3, n_max=6))
    env.reset()

    _, reward, terminated, truncated, info = env.step(4)

    assert reward == -1.1
    assert not terminated
    assert not truncated
    assert info["state"] == (0, 0, 0)
    assert info["invalid_move"] is True
    assert info["invalid_reason"] == "empty_source"


def test_env_rejects_move_onto_smaller_disk() -> None:
    env = TowerOfHanoiEnv(EnvConfig(n_disks=3, n_max=6))
    env.reset()
    env.step(0)

    _, reward, terminated, truncated, info = env.step(0)

    assert reward == -1.1
    assert not terminated
    assert not truncated
    assert info["state"] == (1, 0, 0)
    assert info["invalid_move"] is True
    assert info["invalid_reason"] == "smaller_disk_on_destination"


def test_env_terminates_on_goal_state() -> None:
    env = TowerOfHanoiEnv(EnvConfig(n_disks=1, n_max=1))
    env.reset()

    _, reward, terminated, truncated, info = env.step(1)

    assert reward == 19.9
    assert terminated is True
    assert truncated is False
    assert info["state"] == (2,)
    assert info["terminated"] is True
    assert info["truncated"] is False
    with pytest.raises(RuntimeError, match="episode has ended"):
        env.step(0)


def test_env_truncates_at_step_limit() -> None:
    env = TowerOfHanoiEnv(EnvConfig(n_disks=1, n_max=1, step_limit_multiplier=1))
    env.reset()

    _, reward, terminated, truncated, info = env.step(0)

    assert reward == -0.1
    assert terminated is False
    assert truncated is True
    assert info["state"] == (1,)
    assert info["terminated"] is False
    assert info["truncated"] is True
    with pytest.raises(RuntimeError, match="episode has ended"):
        env.step(1)


def test_export_trajectory_returns_independent_copy() -> None:
    env = TowerOfHanoiEnv(EnvConfig(n_disks=1, n_max=1))
    env.reset()
    env.step(1)

    trajectory = env.export_trajectory()

    assert len(trajectory) == 2
    assert trajectory[0]["action"] is None
    assert trajectory[1]["action"] == 1
    assert trajectory[1]["terminated"] is True

    trajectory[0]["state"][0] = 99

    fresh_trajectory = env.export_trajectory()
    assert fresh_trajectory[0]["state"] == [0]
