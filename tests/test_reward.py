from hanoi_rl.envs.reward import RewardConfig, compute_reward


def test_compute_reward_for_valid_non_terminal_step() -> None:
    reward = compute_reward(
        goal_reached=False,
        invalid_move=False,
        config=RewardConfig(),
    )

    assert reward == -0.1


def test_compute_reward_for_invalid_move() -> None:
    reward = compute_reward(
        goal_reached=False,
        invalid_move=True,
        config=RewardConfig(),
    )

    assert reward == -1.1
