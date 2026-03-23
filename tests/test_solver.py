from hanoi_rl.baselines.recursive_solver import optimal_move_count, solve_tower_of_hanoi


def test_recursive_solver_matches_optimal_move_count() -> None:
    moves = solve_tower_of_hanoi(3)

    assert len(moves) == optimal_move_count(3)
    assert moves[0] == (0, 2)
    assert moves[-1] == (0, 2)
