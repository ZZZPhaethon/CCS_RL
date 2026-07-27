import numpy as np

from experiments.generate_iterative_q_policy_data import (
    _candidate_record,
    update_gate_state,
)


class _Wrapper:
    def __init__(self):
        self.env = type("Env", (), {"t": 120})()
        self.observation_space = {
            "state": type("State", (), {"shape": (3,)})(),
        }
        self.action_space = type("Action", (), {"n": 4})()

    def action_masks(self):
        return np.ones(4, dtype=bool)


def test_gate_state_only_consumes_budget_for_override():
    state = {"used_windows": set(), "override_events": 0}
    update_gate_state(state, action=3, follow=3, active_window=0)
    assert state == {"used_windows": set(), "override_events": 0}
    update_gate_state(state, action=1, follow=3, active_window=2)
    assert state == {"used_windows": {2}, "override_events": 1}


def test_candidate_target_is_economic_saving_relative_to_anchor():
    wrapper = _Wrapper()
    observation = {
        "state": np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
    }
    baseline = {"total_cost_eur": 2_000_000.0}
    candidate = {"total_cost_eur": 1_950_000.0}
    record = _candidate_record(
        wrapper,
        observation,
        seed=1,
        candidate_index=0,
        action=1,
        anchor_action=3,
        window_index=0,
        baseline_metrics=baseline,
        candidate_metrics=candidate,
        reward_scale=1e-5,
    )
    assert record["return_to_go"][0] == 0.5
    assert record["anchor_action"] == 3
