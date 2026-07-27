from __future__ import annotations

from dataclasses import fields, replace
import importlib
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from sim.control.baselines import greedy_shuttle_policy, idle_policy
from sim.control.replay import ReplaySnapshot
from sim.train import make_native_env


def _demonstrations():
    return importlib.import_module("sim.control.demonstrations")


def _batch(operation_modes=None, vessel_destinations=None):
    demonstrations = _demonstrations()
    return demonstrations.MpcDemonstrationBatch(
        state=np.arange(6, dtype=np.float32).reshape(2, 3),
        forecast=np.arange(2 * 168 * 9, dtype=np.float32).reshape(2, 168, 9),
        actions=np.array([[0, 2], [1, 3]], dtype=np.int64),
        masks=np.array(
            [[True, False, True, False], [False, True, False, True]],
            dtype=bool,
        ),
        seeds=np.array([11, 12], dtype=np.int64),
        hours=np.array([0, 1], dtype=np.int64),
        metadata={"schema": "demo-v1", "nested": {"b": 2, "a": 1}},
        operation_modes=operation_modes,
        vessel_destinations=vessel_destinations,
    )


def _operation_modes():
    return np.asarray(
        [
            [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]],
            [[0, 0, 1, 0, 0], [0, 0, 0, 1, 0]],
        ],
        dtype=np.float32,
    )


def _vessel_destinations():
    return np.asarray(
        [
            [[1, 0, 0], [0, 0, 0]],
            [[0, 1, 0], [0, 0, 1]],
        ],
        dtype=np.float32,
    )


def _write_corrupt_cache(path, **overrides):
    demonstrations = _demonstrations()
    demonstrations.save_demonstrations(_batch(), path)
    with np.load(path, allow_pickle=False) as cache:
        payload = {name: cache[name] for name in cache.files}
    payload.update(overrides)
    np.savez_compressed(path, **payload)


class _DemoFactory:
    def __init__(self, env_hours: int = 1):
        self.env_hours = env_hours
        self.calls = 0

    def __call__(self, *, demonstration: bool):
        assert demonstration is True
        self.calls += 1
        return make_native_env(
            episode_hours=self.env_hours,
            scenario_context_hours=169,
            scenario="northern_lights_phase1_3vessels",
            weather_mode="block",
            warm_start=False,
            capture_noise_std=0.0,
            initial_inventory_fill_max=0.0,
            include_weather_obs=False,
        )

    def metadata(self):
        return {"schema": "short-real-demo", "episode_hours": self.env_hours}


class _ExactController:
    def __init__(self, env, replan_every: int, planning_horizon_h: int):
        assert replan_every == 24
        assert planning_horizon_h == 168
        self.last_trace_replay_is_exact = True
        self.last_trace_replay_mismatches = ()
        self.last_candidate_name = "greedy"

    def __call__(self, env):
        return {
            "vessels": [0] * len(env.vessel_ids),
            "wells": [0] * len(env.well_ids),
        }


class _InexactController(_ExactController):
    def __init__(self, env, replan_every: int, planning_horizon_h: int):
        super().__init__(env, replan_every, planning_horizon_h)
        self.last_trace_replay_is_exact = False
        self.last_trace_replay_mismatches = ("candidate replay mismatch",)


def _snapshot() -> ReplaySnapshot:
    return ReplaySnapshot(
        elapsed_hours=1,
        stored_t=1.0,
        vented_t=2.0,
        captured_t=3.0,
        in_transit_t=4.0,
        vessel_fuel=5.0,
        conditioning=6.0,
        reconditioning=7.0,
        loading=8.0,
        unloading=9.0,
        operating_cost=35.0,
        total_cost=36.0,
        total_reward=-37.0,
        objective_value=38.0,
        overflow_risk_t=39.0,
        injection_tph=(1.0,),
        entity_inventory_t={"entity": 40.0},
        vessel_berths={"vessel": "terminal"},
    )


def test_cache_round_trip_preserves_arrays_metadata_and_canonical_dtypes(tmp_path):
    demonstrations = _demonstrations()
    batch = _batch()
    path = tmp_path / "nested" / "demo.npz"

    demonstrations.save_demonstrations(batch, path)
    loaded = demonstrations.load_demonstrations(path, batch.metadata)

    assert path.exists()
    for name in ("state", "forecast", "actions", "masks", "seeds", "hours"):
        np.testing.assert_array_equal(getattr(loaded, name), getattr(batch, name))
    assert loaded.metadata == batch.metadata
    assert loaded.state.dtype == np.float32
    assert loaded.forecast.dtype == np.float32
    assert loaded.actions.dtype == np.int64
    assert loaded.masks.dtype == np.bool_
    assert loaded.seeds.dtype == np.int64
    assert loaded.hours.dtype == np.int64
    assert loaded.operation_modes is None


def test_v2_cache_round_trip_preserves_operation_modes(tmp_path):
    demonstrations = _demonstrations()
    batch = _batch(operation_modes=_operation_modes())
    path = tmp_path / "demo-v2.npz"

    demonstrations.save_demonstrations(batch, path)
    loaded = demonstrations.load_demonstrations(path, batch.metadata)

    np.testing.assert_array_equal(loaded.operation_modes, batch.operation_modes)
    assert loaded.operation_modes.dtype == np.float32


def test_v3_cache_round_trip_preserves_vessel_destinations(tmp_path):
    demonstrations = _demonstrations()
    batch = _batch(
        operation_modes=_operation_modes(),
        vessel_destinations=_vessel_destinations(),
    )
    path = tmp_path / "demo-v3.npz"

    demonstrations.save_demonstrations(batch, path)
    loaded = demonstrations.load_demonstrations(path, batch.metadata)

    np.testing.assert_array_equal(
        loaded.vessel_destinations,
        batch.vessel_destinations,
    )
    assert loaded.vessel_destinations.dtype == np.float32


def test_candidate_cache_round_trip_preserves_indices_and_names(tmp_path):
    demonstrations = _demonstrations()
    batch = replace(
        _batch(
            operation_modes=_operation_modes(),
            vessel_destinations=_vessel_destinations(),
        ),
        plan_candidates=np.asarray([0, 1], dtype=np.int64),
        candidate_names=("greedy", "forecast_urgency"),
    )
    path = tmp_path / "demo-candidates.npz"

    demonstrations.save_demonstrations(batch, path)
    loaded = demonstrations.load_demonstrations(path, batch.metadata)

    np.testing.assert_array_equal(loaded.plan_candidates, [0, 1])
    assert loaded.candidate_names == ("greedy", "forecast_urgency")


def test_observation_variants_have_expected_shapes_and_flatten_time_major():
    batch = _batch()

    assert batch.observations("state") is batch.state
    flat = batch.observations("flat")
    assert flat.shape == (2, 3 + 168 * 9)
    assert flat.dtype == np.float32
    np.testing.assert_array_equal(flat[:, :3], batch.state)
    np.testing.assert_array_equal(flat[:, 3:], batch.forecast.reshape(2, -1))
    tcn = batch.observations("tcn")
    assert set(tcn) == {"state", "forecast"}
    assert tcn["state"] is batch.state
    assert tcn["forecast"] is batch.forecast
    future_mlp = batch.observations("future_mlp")
    assert future_mlp["state"] is batch.state
    assert future_mlp["forecast"] is batch.forecast
    with pytest.raises(ValueError, match="variant"):
        batch.observations("unknown")


def test_mode_observation_variants_append_flattened_vessel_major_modes():
    batch = _batch(operation_modes=_operation_modes())

    state_mode = batch.observations("state_mode")
    assert state_mode.shape == (2, 13)
    np.testing.assert_array_equal(state_mode[:, :3], batch.state)
    np.testing.assert_array_equal(state_mode[:, 3:], _operation_modes().reshape(2, 10))
    tcn_mode = batch.observations("tcn_mode")
    np.testing.assert_array_equal(tcn_mode["state"], state_mode)
    assert tcn_mode["forecast"] is batch.forecast
    future_mlp_mode = batch.observations("future_mlp_mode")
    np.testing.assert_array_equal(future_mlp_mode["state"], state_mode)
    assert future_mlp_mode["forecast"] is batch.forecast

    with pytest.raises(ValueError, match="operation mode"):
        _batch().observations("state_mode")


def test_destination_variant_appends_modes_then_sailing_destinations():
    batch = _batch(
        operation_modes=_operation_modes(),
        vessel_destinations=_vessel_destinations(),
    )

    expected_state = np.concatenate(
        (
            batch.state,
            _operation_modes().reshape(2, -1),
            _vessel_destinations().reshape(2, -1),
        ),
        axis=1,
    )
    for variant in (
        "tcn_mode_destination",
        "gnn_mode_destination",
        "larger_mlp_mode_destination",
        "edge_gnn_mode_destination",
        "future_mlp_mode_destination",
        "balanced_edge_gnn_mode_destination",
        "balanced_edge_gnn_future_mlp_mode_destination",
        "fixed_scale_larger_mlp_mode_destination",
        "fixed_scale_edge_gnn_mode_destination",
        "stable_tcn_mode_destination",
        "fixed_scale_tcn_mode_destination",
    ):
        destination = batch.observations(variant)
        np.testing.assert_array_equal(destination["state"], expected_state)
        assert destination["forecast"] is batch.forecast
        with pytest.raises(ValueError, match="destination"):
            _batch(operation_modes=_operation_modes()).observations(variant)


@pytest.mark.parametrize(
    ("variant", "zero"),
    [
        ("gated_past24_mlp_mode_destination", False),
        ("past24_mlp_mode_destination", False),
        ("past24_zero_mlp_mode_destination", True),
    ],
)
def test_past_mlp_variant_reconstructs_strictly_previous_rows(variant, zero):
    batch = replace(
        _batch(
            operation_modes=_operation_modes(),
            vessel_destinations=_vessel_destinations(),
        ),
        seeds=np.asarray([11, 11], dtype=np.int64),
        hours=np.asarray([0, 1], dtype=np.int64),
        metadata={"action_dimensions": [2, 4]},
    )

    observation = batch.observations(variant)

    assert set(observation) == {"state", "past", "forecast"}
    assert observation["past"].shape == (2, 24, 22)
    np.testing.assert_array_equal(observation["past"][0], 0.0)
    if zero:
        np.testing.assert_array_equal(observation["past"], 0.0)
    else:
        np.testing.assert_array_equal(
            observation["past"][1, -1, :19],
            observation["state"][0],
        )
        np.testing.assert_allclose(
            observation["past"][1, -1, 19:21],
            [0.0, 2.0 / 3.0],
        )
        assert observation["past"][1, -1, -1] == 1.0


def test_replan_phase_variant_appends_normalized_phase_and_indicator():
    batch = _batch(
        operation_modes=_operation_modes(),
        vessel_destinations=_vessel_destinations(),
    )

    observation = batch.observations(
        "fixed_scale_tcn_mode_destination_replan_phase"
    )

    np.testing.assert_allclose(
        observation["state"][:, -2:],
        np.asarray([[0.0, 1.0], [1.0 / 23.0, 0.0]], dtype=np.float32),
    )


def test_oracle_candidate_variant_appends_candidate_one_hot():
    batch = replace(
        _batch(
            operation_modes=_operation_modes(),
            vessel_destinations=_vessel_destinations(),
        ),
        plan_candidates=np.asarray([0, 1], dtype=np.int64),
        candidate_names=("greedy", "forecast_urgency"),
    )

    observation = batch.observations(
        "fixed_scale_tcn_mode_destination_replan_phase_oracle_candidate"
    )

    np.testing.assert_array_equal(
        observation["state"][:, -2:],
        np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )


def test_learned_plan_context_variant_appends_continuous_context():
    context = np.arange(16, dtype=np.float32).reshape(2, 8) / 16.0
    batch = replace(
        _batch(
            operation_modes=_operation_modes(),
            vessel_destinations=_vessel_destinations(),
        ),
        plan_context=context,
    )

    observation = batch.observations(
        "fixed_scale_tcn_mode_destination_replan_phase_learned_plan_context"
    )

    np.testing.assert_array_equal(observation["state"][:, -8:], context)


def test_metadata_mismatch_is_rejected(tmp_path):
    demonstrations = _demonstrations()
    path = tmp_path / "demo.npz"
    demonstrations.save_demonstrations(_batch(), path)

    with pytest.raises(ValueError, match="metadata.*schema"):
        demonstrations.load_demonstrations(path, {"schema": "different"})


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"forecast": np.zeros((2, 167, 9), dtype=np.float32)}, "forecast"),
        ({"state": np.array([[np.nan], [0.0]], dtype=np.float32)}, "finite"),
        ({"actions": np.zeros((1, 2), dtype=np.int64)}, "leading"),
        ({"state": np.array([["bad"], ["dtype"]])}, "dtype"),
        (
            {"operation_modes": np.full((2, 2, 5), 0.2, dtype=np.float32)},
            "one-hot",
        ),
        (
            {"vessel_destinations": np.full((2, 2, 3), 0.5, dtype=np.float32)},
            "destination.*one-hot",
        ),
    ],
)
def test_invalid_cache_schema_is_rejected(tmp_path, override, message):
    demonstrations = _demonstrations()
    path = tmp_path / "invalid.npz"
    _write_corrupt_cache(path, **override)

    with pytest.raises(ValueError, match=message):
        demonstrations.load_demonstrations(path, {"schema": "demo-v1"})


def test_short_real_collection_returns_feasible_rows_and_exact_forecasts():
    demonstrations = _demonstrations()
    factory = _DemoFactory(env_hours=1)

    batch = demonstrations.collect_mpc_demonstrations(
        factory,
        seeds=[3, 4],
        episode_hours=1,
    )

    assert factory.calls == 2
    assert batch.state.shape[0] == 2
    assert batch.forecast.shape == (2, 168, 9)
    assert batch.actions.shape[0] == batch.masks.shape[0] == 2
    assert batch.operation_modes.shape == (2, 3, 5)
    np.testing.assert_array_equal(batch.operation_modes.sum(axis=2), np.ones((2, 3)))
    assert batch.vessel_destinations.shape == (2, 3, 4)
    assert np.all(batch.vessel_destinations.sum(axis=2) <= 1.0)
    assert batch.seeds.tolist() == [3, 4]
    assert batch.hours.tolist() == [0, 0]
    assert batch.metadata == factory.metadata()
    assert batch.plan_candidates.shape == (2,)
    assert batch.candidate_names is not None
    assert len(batch.candidate_names) == 8
    action_dims = [5, 5, 5, 11]
    offsets = np.cumsum([0, *action_dims])
    for row in range(2):
        for dimension, choice in enumerate(batch.actions[row]):
            assert batch.masks[row, offsets[dimension] + choice]


def test_short_greedy_collection_has_no_mpc_candidate_labels():
    demonstrations = _demonstrations()
    factory = _DemoFactory(env_hours=1)

    batch = demonstrations.collect_mpc_demonstrations(
        factory,
        seeds=[3],
        episode_hours=1,
        teacher_policy=lambda env: greedy_shuttle_policy(env),
    )

    assert batch.state.shape[0] == 1
    assert batch.plan_candidates is None
    assert batch.candidate_names is None


def test_dagger_collection_executes_behavior_but_stores_expert_labels():
    demonstrations = _demonstrations()
    factory = _DemoFactory(env_hours=1)
    behavior_actions = []

    def behavior(env):
        action = idle_policy(env)
        action["vessels"] = [
            next(
                (index for index, allowed in enumerate(mask) if index > 0 and allowed),
                0,
            )
            for mask in env.vessel_action_mask()
        ]
        behavior_actions.append(action)
        return action

    batch = demonstrations.collect_dagger_demonstrations(
        factory,
        seeds=[3],
        episode_hours=1,
        behavior_policy=behavior,
        expert_policy=idle_policy,
    )

    assert len(behavior_actions) == 1
    assert any(choice != 0 for choice in behavior_actions[0]["vessels"])
    np.testing.assert_array_equal(batch.actions[0, :3], np.zeros(3, dtype=np.int64))
    assert batch.plan_candidates is None
    assert batch.candidate_names is None
    assert batch.metadata == factory.metadata()


def test_candidate_replay_mismatch_fails_with_seed_and_hour_context():
    demonstrations = _demonstrations()
    factory = _DemoFactory(env_hours=1)

    with patch.object(
        demonstrations,
        "RollingNativeMpcController",
        _InexactController,
    ), pytest.raises(RuntimeError, match=r"seed=7.*hour=0.*candidate replay mismatch"):
        demonstrations.collect_mpc_demonstrations(factory, seeds=[7], episode_hours=1)


def test_non_executable_full_trace_replay_fails_loudly():
    demonstrations = _demonstrations()
    replay_result = SimpleNamespace(
        actual=_snapshot(),
        is_executable=False,
        is_exact=False,
        mismatches=("broken trace",),
    )

    with patch.object(
        demonstrations,
        "RollingNativeMpcController",
        _ExactController,
    ), patch.object(
        demonstrations,
        "replay_native_actions",
        return_value=replay_result,
    ), pytest.raises(RuntimeError, match=r"seed=8.*broken trace"):
        demonstrations.collect_mpc_demonstrations(
            _DemoFactory(env_hours=1),
            seeds=[8],
            episode_hours=1,
        )


def test_inexact_full_trace_replay_uses_every_snapshot_field_and_fails_loudly():
    demonstrations = _demonstrations()
    snapshot = _snapshot()
    calls = 0

    def fake_replay(env, actions, *, horizon_h, expected=None):
        nonlocal calls
        calls += 1
        assert horizon_h == 1
        if expected is None:
            return SimpleNamespace(
                actual=snapshot,
                is_executable=True,
                is_exact=False,
                mismatches=(),
            )
        snapshot_fields = {field.name for field in fields(ReplaySnapshot)}
        assert expected.required_fields == snapshot_fields
        for name in snapshot_fields:
            assert getattr(expected, name) == getattr(snapshot, name)
        return SimpleNamespace(
            actual=snapshot,
            is_executable=True,
            is_exact=False,
            mismatches=("stored_t mismatch",),
        )

    with patch.object(
        demonstrations,
        "RollingNativeMpcController",
        _ExactController,
    ), patch.object(
        demonstrations,
        "replay_native_actions",
        side_effect=fake_replay,
    ), pytest.raises(RuntimeError, match=r"seed=9.*stored_t mismatch"):
        demonstrations.collect_mpc_demonstrations(
            _DemoFactory(env_hours=1),
            seeds=[9],
            episode_hours=1,
        )
    assert calls == 2


def test_premature_episode_completion_is_rejected():
    demonstrations = _demonstrations()

    with patch.object(
        demonstrations,
        "RollingNativeMpcController",
        _ExactController,
    ), pytest.raises(RuntimeError, match=r"seed=10.*hour=0.*premature"):
        demonstrations.collect_mpc_demonstrations(
            _DemoFactory(env_hours=1),
            seeds=[10],
            episode_hours=2,
        )


def _merge_shard(seed: int, hours: list[int]):
    demonstrations = _demonstrations()
    rows = len(hours)
    return demonstrations.MpcDemonstrationBatch(
        state=np.full((rows, 3), seed, dtype=np.float32),
        forecast=np.full((rows, 168, 9), seed, dtype=np.float32),
        actions=np.zeros((rows, 2), dtype=np.int64),
        masks=np.ones((rows, 4), dtype=bool),
        seeds=np.full(rows, seed, dtype=np.int64),
        hours=np.asarray(hours, dtype=np.int64),
        metadata={"schema": "merge-v2"},
        operation_modes=np.tile(
            np.asarray([[[1, 0, 0, 0, 0], [0, 0, 0, 0, 1]]], dtype=np.float32),
            (rows, 1, 1),
        ),
    )


def test_merge_demonstration_shards_requires_complete_seeds_and_sorts_rows():
    demonstrations = _demonstrations()
    merged = demonstrations.merge_demonstration_shards(
        [_merge_shard(1, [1, 0]), _merge_shard(0, [1, 0])],
        expected_seeds=[0, 1],
        episode_hours=2,
    )

    assert list(zip(merged.seeds.tolist(), merged.hours.tolist())) == [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    ]
    assert merged.operation_modes.shape == (4, 2, 5)


def test_merge_demonstration_shards_preserves_vessel_destinations():
    shards = []
    for seed in (1, 0):
        shard = _merge_shard(seed, [1, 0])
        destinations = np.zeros((2, 2, 3), dtype=np.float32)
        destinations[:, 0, seed] = 1.0
        shards.append(replace(shard, vessel_destinations=destinations))

    merged = _demonstrations().merge_demonstration_shards(
        shards,
        expected_seeds=[0, 1],
        episode_hours=2,
    )

    assert merged.vessel_destinations.shape == (4, 2, 3)
    np.testing.assert_array_equal(
        merged.vessel_destinations[:, 0],
        [[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0]],
    )


def test_merge_demonstration_shards_rejects_mixed_destination_schemas():
    with_destinations = replace(
        _merge_shard(0, [0]),
        vessel_destinations=np.zeros((1, 2, 3), dtype=np.float32),
    )

    with pytest.raises(ValueError, match="destination schema"):
        _demonstrations().merge_demonstration_shards(
            [with_destinations, _merge_shard(1, [0])],
            expected_seeds=[0, 1],
            episode_hours=1,
        )


@pytest.mark.parametrize(
    ("shards", "message"),
    [
        ([_merge_shard(0, [0, 1])], "missing.*seed"),
        ([_merge_shard(0, [0, 1]), _merge_shard(0, [0, 1])], "duplicate"),
        ([_merge_shard(0, [0]), _merge_shard(1, [0, 1])], "complete.*hours"),
    ],
)
def test_merge_demonstration_shards_rejects_missing_duplicate_or_incomplete_rows(
    shards, message
):
    demonstrations = _demonstrations()
    with pytest.raises(ValueError, match=message):
        demonstrations.merge_demonstration_shards(
            shards,
            expected_seeds=[0, 1],
            episode_hours=2,
        )
