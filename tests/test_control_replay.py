from __future__ import annotations

from sim.control.replay import (
    ReplayExpectation,
    ReplaySnapshot,
    ReplayTolerances,
    compare_replay_snapshots,
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
