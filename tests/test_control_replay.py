from __future__ import annotations

from sim.control.replay import (
    ReplayExpectation,
    ReplaySnapshot,
    ReplayTolerances,
    compare_replay_snapshots,
    replay_native_actions,
)
from sim.environment import CCSEnv, CCSEnvConfig, VESSEL_WAIT
from sim.scenario_generation import ScenarioConfig, ScenarioGenerator
from tests.fixtures.toy_networks import (
    TOY_TWO_SOURCE_LOCATIONS,
    make_toy_two_source_network,
)


def _snapshot(**overrides) -> ReplaySnapshot:
    values = {
        "elapsed_hours": 2,
        "stored_t": 10.0,
        "vented_t": 0.0,
        "captured_t": 10.0,
        "in_transit_t": 0.0,
        "vessel_fuel": 1.0,
        "conditioning": 2.0,
        "reconditioning": 3.0,
        "loading": 4.0,
        "unloading": 5.0,
        "operating_cost": 15.0,
        "total_cost": 15.0,
        "total_reward": -15.0,
        "objective_value": 15.0,
        "overflow_risk_t": 0.0,
        "injection_tph": (4.0, 6.0),
        "entity_inventory_t": {"source": 0.0, "terminal": 0.0, "ship": 0.0},
        "vessel_berths": {"ship": "terminal"},
    }
    values.update(overrides)
    return ReplaySnapshot(**values)


def _replay_env(hours: int = 2) -> CCSEnv:
    return CCSEnv(
        make_toy_two_source_network(),
        TOY_TWO_SOURCE_LOCATIONS,
        scenario_generator=ScenarioGenerator(
            config=ScenarioConfig(episode_hours=hours, randomize_initial_inventory=False)
        ),
        config=CCSEnvConfig(episode_hours=hours),
    )


def _wait_action() -> dict[str, list[int]]:
    return {"vessels": [VESSEL_WAIT, VESSEL_WAIT], "wells": [0, 0]}


def test_exact_requires_every_required_field_to_be_present_and_equal():
    expected = ReplayExpectation(
        required_fields=frozenset({"elapsed_hours", "stored_t", "vented_t"}),
        elapsed_hours=2,
        stored_t=10.0,
        vented_t=0.0,
    )

    exact, mismatches, compared = compare_replay_snapshots(expected, _snapshot())

    assert exact
    assert not mismatches
    assert compared == expected.required_fields


def test_missing_required_expectation_is_not_exact():
    expected = ReplayExpectation(
        required_fields=frozenset({"stored_t", "objective_value"}),
        stored_t=10.0,
    )

    exact, mismatches, compared = compare_replay_snapshots(expected, _snapshot())

    assert not exact
    assert compared == frozenset({"stored_t"})
    assert any("objective_value" in mismatch and "missing" in mismatch for mismatch in mismatches)


def test_field_specific_tolerance_reports_named_mismatch():
    expected = ReplayExpectation(
        required_fields=frozenset({"stored_t"}),
        stored_t=10.0,
    )

    exact, mismatches, compared = compare_replay_snapshots(
        expected,
        _snapshot(stored_t=10.01),
        tolerances=ReplayTolerances(mass_t=1e-3),
    )

    assert not exact
    assert compared == frozenset({"stored_t"})
    assert len(mismatches) == 1
    assert "stored_t" in mismatches[0]
    assert "0.001" in mismatches[0]


def test_supplied_final_state_is_compared_even_when_not_required():
    expected = ReplayExpectation(
        required_fields=frozenset({"stored_t"}),
        stored_t=10.0,
        entity_inventory_t={"source": 1.0, "terminal": 0.0, "ship": 0.0},
    )

    exact, mismatches, compared = compare_replay_snapshots(expected, _snapshot())

    assert not exact
    assert compared == frozenset({"stored_t", "entity_inventory_t"})
    assert any("entity_inventory_t[source]" in mismatch for mismatch in mismatches)


def test_replay_runs_full_horizon_without_mutating_source_env():
    env = _replay_env(hours=2)
    env.reset(seed=1)
    before = env.simulator.state.as_dict()

    result = replay_native_actions(
        env,
        [_wait_action(), _wait_action()],
        horizon_h=2,
    )

    assert result.is_executable
    assert not result.is_exact
    assert result.actual.elapsed_hours == 2
    assert len(result.actual.injection_tph) == 2
    assert env.simulator.state.as_dict() == before
    assert env.t == 0


def test_wrong_action_dimension_is_non_executable_without_stepping():
    env = _replay_env(hours=1)
    env.reset(seed=1)

    result = replay_native_actions(
        env,
        [{"vessels": [], "wells": [0, 0]}],
        horizon_h=1,
    )

    assert not result.is_executable
    assert result.actual.elapsed_hours == 0
    assert any("dimension" in mismatch for mismatch in result.mismatches)


def test_mask_invalid_action_is_non_executable():
    env = _replay_env(hours=1)
    env.reset(seed=1)
    action = _wait_action()
    action["vessels"][0] = 999

    result = replay_native_actions(env, [action], horizon_h=1)

    assert not result.is_executable
    assert result.actual.elapsed_hours == 0
    assert any("vessel_a" in mismatch and "not executable" in mismatch for mismatch in result.mismatches)


def test_short_trace_is_non_executable():
    env = _replay_env(hours=1)
    env.reset(seed=1)

    result = replay_native_actions(env, [], horizon_h=1)

    assert not result.is_executable
    assert result.actual.elapsed_hours == 0
    assert any("horizon" in mismatch for mismatch in result.mismatches)


def test_replay_rejects_non_positive_horizon():
    env = _replay_env(hours=1)
    env.reset(seed=1)

    try:
        replay_native_actions(env, [], horizon_h=0)
    except ValueError as error:
        assert "positive" in str(error)
    else:
        raise AssertionError("expected ValueError")
