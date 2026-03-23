from hanoi_rl.agents import TabularQAgent, TabularQConfig


def test_tabular_agent_update_changes_q_value() -> None:
    agent = TabularQAgent(TabularQConfig(seed=7))
    state = (0, 0, 0)
    next_state = (1, 0, 0)

    before = agent.q_values[state][0]
    agent.update(state, 0, 1.0, next_state, done=False)
    after = agent.q_values[state][0]

    assert after != before
