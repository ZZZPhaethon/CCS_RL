from unittest.mock import patch

import pytest

from sim.control import rolling_milp, trip_milp
from sim.control.rolling_milp import RollingMilpController
from sim.control.replay import replay_native_actions
from sim.economics import EconomicParameters
from sim.environment import VESSEL_WAIT
from tests.test_rolling_milp import HAVE_PULP, _no_capture_env


def test_trip_materialization_uses_common_replay_metrics():
    env = _no_capture_env(cap_hours=1)
    env.reset(seed=1)
    actions = [{"vessels": [VESSEL_WAIT], "wells": [0]}]

    with patch(
        "sim.control.trip_milp.replay_native_actions",
        wraps=replay_native_actions,
    ) as common_replay:
        materialized = trip_milp.materialize_native_action_trace(
            env,
            actions,
            horizon_h=1,
            economics=EconomicParameters(),
        )

    actual = replay_native_actions(env, actions, horizon_h=1).actual
    common_replay.assert_called_once()
    assert materialized.stored_t == pytest.approx(actual.stored_t)
    assert materialized.vented_t == pytest.approx(actual.vented_t)
    assert materialized.operating_cost == pytest.approx(actual.operating_cost)
    assert materialized.total_cost == pytest.approx(actual.total_cost)
    assert materialized.objective_value == pytest.approx(actual.objective_value)
    assert materialized.injection_tph == pytest.approx(actual.injection_tph)


def test_trip_materialization_does_not_fill_an_invalid_trace_with_wait_actions():
    env = _no_capture_env(cap_hours=1)
    env.reset(seed=1)
    actions = [{"vessels": [], "wells": []}]

    materialized = trip_milp.materialize_native_action_trace(
        env,
        actions,
        horizon_h=1,
    )

    assert not materialized.is_valid
    assert materialized.native_actions_by_hour == actions
    assert "dimension" in materialized.validation_error


@pytest.mark.skipif(not HAVE_PULP, reason="pulp/CBC not installed")
def test_rolling_plan_exposes_common_replay_exactness():
    env = _no_capture_env(cap_hours=1)
    env.reset(seed=1)

    with patch(
        "sim.control.rolling_milp.replay_native_actions",
        wraps=replay_native_actions,
    ) as common_replay:
        plan = rolling_milp._plan_explicit_actions(
            env,
            planning_horizon_h=1,
            economics=EconomicParameters(),
        )

    common_replay.assert_called_once()
    assert plan.replay_is_valid
    assert plan.replay_is_exact
    assert plan.is_valid
    assert not plan.replay_mismatches
    assert {"stored_t", "vented_t", "total_cost", "injection_tph"} <= plan.replay_compared_fields


@pytest.mark.parametrize(
    "actions, now",
    [
        ({}, 0.0),
        ({"ship": [VESSEL_WAIT]}, 1.0),
        ({"ship": [99]}, 0.0),
    ],
)
def test_rolling_controller_rejects_missing_expired_or_invalid_vessel_trace(actions, now):
    env = _no_capture_env(cap_hours=2)
    env.reset(seed=1)
    controller = RollingMilpController(env)
    controller._vessel_actions_by_hour = actions
    controller._plan_origin_h = 0.0

    with pytest.raises(RuntimeError, match="rolling_milp"):
        controller._planned_vessel_action(
            env,
            "ship",
            now,
            env.vessel_action_mask()[0],
        )


@pytest.mark.skipif(not HAVE_PULP, reason="pulp/CBC not installed")
def test_rolling_plan_rejects_an_executable_but_inexact_replay():
    from tests.test_rolling_milp import _two_berth_parallel_env

    env = _two_berth_parallel_env()
    env.reset(seed=1)
    env.simulator.state.entity_inventory_t["source"] = 2_000.0

    plan = rolling_milp._plan_explicit_actions(
        env,
        planning_horizon_h=3,
        economics=EconomicParameters(),
    )

    assert plan.replay_is_valid
    assert not plan.replay_is_exact
    assert not plan.is_valid
    assert plan.replay_mismatches
    assert "expected" in plan.validation_error
