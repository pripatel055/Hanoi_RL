from hanoi_rl.envs.encoding import one_hot_encode_state


def test_one_hot_encode_state_pads_to_n_max() -> None:
    encoded = one_hot_encode_state([0, 2, 1], n_max=5)

    assert encoded.shape == (15,)
    assert encoded.sum() == 3.0
    assert encoded[:9].tolist() == [
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        1.0,
        0.0,
    ]
