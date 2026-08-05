import numpy as np

from experiments import generate_iterative_q_policy_data as policy_data
from experiments.generate_iterative_q_policy_data import (
    _candidate_record,
    prepare_root,
    select_target_root_h,
    select_window_indices,
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


def test_random_time_root_is_reproducible_and_inside_window():
    args = type(
        "Args",
        (),
        {"root_selection": "random_time", "dataset_seed": 20260803},
    )()
    first = select_target_root_h(args, 1500, 3, 252, 299)
    second = select_target_root_h(args, 1500, 3, 252, 299)
    assert first == second
    assert 252 <= first <= 299


def test_first_event_mode_targets_window_start():
    args = type(
        "Args",
        (),
        {"root_selection": "first_decision_event", "dataset_seed": 20260803},
    )()
    assert select_target_root_h(args, 1500, 3, 252, 299) == 252


def test_windows_per_seed_rotates_coverage_without_changing_root_count():
    args = type(
        "Args",
        (),
        {"window_indices": None, "windows_per_seed": 12},
    )()
    first = select_window_indices(args, seed=24, window_count=24)
    second = select_window_indices(args, seed=25, window_count=24)
    assert first == list(range(12))
    assert len(second) == 12
    assert second != first


def test_random_time_rollin_reserves_target_window_until_root(monkeypatch):
    class Wrapper:
        def __init__(self):
            self.env = type("Env", (), {"t": 100})()
            self.actions = []

        def reset_native_seed(self, _seed):
            return {"state": np.zeros(1, dtype=np.float32)}, {}

        def step(self, action):
            self.actions.append((self.env.t, int(action)))
            self.env.t = {100: 120, 120: 130, 130: 145}[self.env.t]
            return {"state": np.zeros(1, dtype=np.float32)}, 0.0, False, False, {}

    wrapper = Wrapper()
    locked_calls = []

    def fake_locked_action(
        wrapper, _observation, _model, _metadata, _config, _state, _device
    ):
        locked_calls.append(wrapper.env.t)
        active_window = 0 if wrapper.env.t >= 108 else None
        return 2, active_window, {}

    monkeypatch.setattr(policy_data.common, "make_event_env", lambda *_: wrapper)
    monkeypatch.setattr(policy_data, "locked_action", fake_locked_action)
    root = prepare_root(
        args=object(),
        model=object(),
        metadata={"follow_action_index": 9},
        policy_config={"windows_h": [[108, 155]], "max_overrides": 12},
        seed=1500,
        window_index=0,
        device="cpu",
        target_root_h=140,
    )

    assert root is not None
    assert wrapper.env.t == 145
    assert wrapper.actions == [(100, 2), (120, 9), (130, 9)]
    assert locked_calls == [100, 145]


def test_prepare_root_skips_when_global_intervention_budget_is_exhausted(
    monkeypatch,
):
    class Wrapper:
        def __init__(self):
            self.env = type("Env", (), {"t": 100})()

        def reset_native_seed(self, _seed):
            return {"state": np.zeros(1, dtype=np.float32)}, {}

        def step(self, _action):
            self.env.t = 120
            return {"state": np.zeros(1, dtype=np.float32)}, 0.0, False, False, {}

    def exhaust_budget(
        wrapper, _observation, _model, _metadata, _config, state, _device
    ):
        state["override_events"] = 12
        return 2, None, {}

    monkeypatch.setattr(policy_data.common, "make_event_env", lambda *_: Wrapper())
    monkeypatch.setattr(policy_data, "locked_action", exhaust_budget)
    root = prepare_root(
        args=object(),
        model=object(),
        metadata={"follow_action_index": 9},
        policy_config={"windows_h": [[108, 155]], "max_overrides": 12},
        seed=1500,
        window_index=0,
        device="cpu",
        target_root_h=108,
    )
    assert root is None
