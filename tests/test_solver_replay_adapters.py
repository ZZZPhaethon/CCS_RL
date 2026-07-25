from types import SimpleNamespace

from sim.control import cplex_milp
from sim.control.replay import replay_native_actions
from sim.environment import VESSEL_WAIT
from tests.test_rolling_milp import _no_capture_env


def _matching_result(env, actions):
    snapshot = replay_native_actions(
        env,
        actions,
        horizon_h=len(actions),
    ).actual
    return SimpleNamespace(
        horizon_h=len(actions),
        native_actions_by_hour=actions,
        stored_t=snapshot.stored_t,
        vented_t=snapshot.vented_t,
        captured_from_operations_t=snapshot.captured_t,
        in_transit_t=snapshot.in_transit_t,
        overflow_risk_t=snapshot.overflow_risk_t,
        vessel_fuel=snapshot.vessel_fuel,
        conditioning=snapshot.conditioning,
        reconditioning=snapshot.reconditioning,
        loading=snapshot.loading,
        unloading=snapshot.unloading,
        operating_cost=snapshot.operating_cost,
        total_cost=snapshot.total_cost,
        objective_value=snapshot.objective_value,
        injection_tph=list(snapshot.injection_tph),
    )


def test_cplex_replay_adapter_reports_exact_match():
    env = _no_capture_env(cap_hours=1)
    env.reset(seed=1)
    actions = [{"vessels": [VESSEL_WAIT], "wells": [0]}]
    result = _matching_result(env, actions)

    replay = cplex_milp.replay_full_scenario_cplex_plan(env, result)

    assert replay.is_executable
    assert replay.is_exact
    assert not replay.mismatches
    assert {
        "elapsed_hours",
        "stored_t",
        "vented_t",
        "captured_t",
        "in_transit_t",
        "operating_cost",
        "total_cost",
        "objective_value",
        "overflow_risk_t",
    } <= replay.compared_fields


def test_cplex_replay_adapter_separates_executable_from_exact():
    env = _no_capture_env(cap_hours=1)
    env.reset(seed=1)
    actions = [{"vessels": [VESSEL_WAIT], "wells": [0]}]
    result = _matching_result(env, actions)
    result.vented_t += 1.0

    replay = cplex_milp.replay_full_scenario_cplex_plan(env, result)

    assert replay.is_executable
    assert not replay.is_exact
    assert any("vented_t" in mismatch for mismatch in replay.mismatches)


def test_cplex_replay_adapter_matches_the_one_kg_model_resolution():
    env = _no_capture_env(cap_hours=1)
    env.reset(seed=1)
    actions = [{"vessels": [VESSEL_WAIT], "wells": [0]}]
    result = _matching_result(env, actions)
    result.overflow_risk_t += 1e-3

    replay = cplex_milp.replay_full_scenario_cplex_plan(env, result)

    assert replay.is_exact

    result.overflow_risk_t += 1.5e-3
    replay = cplex_milp.replay_full_scenario_cplex_plan(env, result)

    assert not replay.is_exact
    assert any("overflow_risk_t" in mismatch for mismatch in replay.mismatches)
