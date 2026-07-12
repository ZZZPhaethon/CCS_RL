from types import SimpleNamespace

import numpy as np
import pytest
from gymnasium import Env, spaces

from sim.control import vessel_diagnostics as diagnostics


class _ProbabilityEnv(Env):
    observation_space = spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)
    action_space = spaces.MultiDiscrete([3, 2])

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(2, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(2, dtype=np.float32), 0.0, False, True, {}


def test_masked_probabilities_respect_forced_wait():
    from sb3_contrib import MaskablePPO

    model = MaskablePPO(
        "MlpPolicy",
        _ProbabilityEnv(),
        n_steps=2,
        batch_size=2,
        n_epochs=1,
        device="cpu",
        verbose=0,
    )
    observations = np.zeros((2, 2), dtype=np.float32)
    masks = np.asarray([[1, 0, 0, 1, 1], [1, 1, 1, 1, 1]], dtype=bool)

    probabilities = diagnostics.masked_vessel_action_probabilities(
        model, observations, masks, vessel_count=1
    )

    assert len(probabilities) == 1
    assert probabilities[0].shape == (2, 3)
    np.testing.assert_allclose(probabilities[0].sum(axis=1), 1.0)
    np.testing.assert_allclose(probabilities[0][0], [1.0, 0.0, 0.0])


def test_demonstration_diagnostics_separate_forced_wait_and_dispatch(monkeypatch):
    probabilities = [
        np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.56, 0.24, 0.2],
                [0.1, 0.8, 0.1],
                [0.34, 0.33, 0.33],
            ],
            dtype=np.float32,
        )
    ]
    monkeypatch.setattr(
        diagnostics,
        "masked_vessel_action_probabilities",
        lambda model, observations, masks, vessel_count: probabilities,
    )
    modes = np.eye(5, dtype=np.float32)[[0, 4, 1, 1]][:, None, :]
    actions = np.asarray([[0], [0], [1], [1]], dtype=np.int64)
    masks = np.asarray(
        [[1, 0, 0], [1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=bool
    )

    rows = diagnostics.demonstration_mode_diagnostics(
        object(), np.zeros((4, 2), dtype=np.float32), actions, masks, modes, 1
    )
    all_row = next(row for row in rows if row["mode"] == "all" and row["vessel"] == "all")

    assert all_row["count"] == 4
    assert all_row["forced_wait_count"] == 1
    assert all_row["voluntary_wait_count"] == 1
    assert all_row["dispatch_count"] == 2
    assert all_row["dispatch_recall"] == pytest.approx(0.5)
    assert all_row["conditional_destination_accuracy"] == pytest.approx(0.5)
    assert all_row["mean_wait_probability"] == pytest.approx(0.5)
    assert all_row["mean_dispatch_probability"] == pytest.approx(0.5)

    sailing = next(row for row in rows if row["mode"] == "sailing" and row["vessel"] == "all")
    assert sailing["forced_wait_count"] == 1
    assert sailing["dispatch_count"] == 0


class _FakeEnv:
    vessel_ids = ["ship"]
    emitter_ids = ["emitter_a", "emitter_b"]

    def __init__(self):
        self.network = SimpleNamespace(
            entities={"ship": SimpleNamespace(capacity_t=100.0)}
        )
        self.simulator = SimpleNamespace(
            state=SimpleNamespace(
                time_h=0.0,
                entity_inventory_t={"ship": 40.0},
                vessel_berths={"ship": "emitter_a"},
            ),
            vessel_states={"ship": {"mode": "berthed", "berth": "emitter_a"}},
        )

    def vessel_action_mask(self):
        return [[True, True, True, True]]

    def _vessel_action_destination(self, vessel_id, action):
        return {0: None, 1: "terminal", 2: "emitter_a", 3: "emitter_b"}[action]


def test_rollout_diagnostics_count_partial_milk_run_and_wait_streak(monkeypatch):
    env = _FakeEnv()
    monkeypatch.setattr(diagnostics, "vessel_operation_modes", lambda env: ("loading",))
    tracker = diagnostics.VesselRolloutDiagnostics()

    tracker.observe(env, [0], wait_probabilities=[0.8])
    env.simulator.state.time_h = 1.0
    tracker.observe(env, [0], wait_probabilities=[0.6])
    env.simulator.state.time_h = 2.0
    tracker.observe(env, [3], wait_probabilities=[0.2])

    row = next(
        row
        for row in tracker.rows(stage="bc", deterministic=True, model_seed=0, eval_seed=101)
        if row["mode"] == "all" and row["vessel"] == "all"
    )
    assert row["dispatch_count"] == 1
    assert row["partial_load_departure_count"] == 1
    assert row["milk_run_departure_count"] == 1
    assert row["first_dispatch_hour"] == 2.0
    assert row["longest_berthed_no_dispatch_streak"] == 2
    assert row["mean_wait_probability"] == pytest.approx((0.8 + 0.6 + 0.2) / 3)
