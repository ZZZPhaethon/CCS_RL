"""Solver-neutral replay snapshots and exactness validation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
import math


@dataclass(frozen=True)
class ReplayTolerances:
    mass_t: float = 1e-6
    cost_eur: float = 1e-6
    reward: float = 1e-9
    objective: float = 1e-6
    state_value: float = 1e-6


@dataclass(frozen=True)
class ReplaySnapshot:
    elapsed_hours: int
    stored_t: float
    vented_t: float
    captured_t: float
    in_transit_t: float
    vessel_fuel: float
    conditioning: float
    reconditioning: float
    loading: float
    unloading: float
    operating_cost: float
    total_cost: float
    total_reward: float
    objective_value: float
    overflow_risk_t: float
    injection_tph: tuple[float, ...]
    entity_inventory_t: dict[str, float]
    vessel_berths: dict[str, str | None]


@dataclass(frozen=True)
class ReplayExpectation:
    required_fields: frozenset[str]
    elapsed_hours: int | None = None
    stored_t: float | None = None
    vented_t: float | None = None
    captured_t: float | None = None
    in_transit_t: float | None = None
    vessel_fuel: float | None = None
    conditioning: float | None = None
    reconditioning: float | None = None
    loading: float | None = None
    unloading: float | None = None
    operating_cost: float | None = None
    total_cost: float | None = None
    total_reward: float | None = None
    objective_value: float | None = None
    overflow_risk_t: float | None = None
    injection_tph: tuple[float, ...] | None = None
    entity_inventory_t: dict[str, float] | None = None
    vessel_berths: dict[str, str | None] | None = None


@dataclass(frozen=True)
class ReplayValidationResult:
    actual: ReplaySnapshot
    is_executable: bool
    is_exact: bool
    violations: tuple[str, ...]
    mismatches: tuple[str, ...]
    compared_fields: frozenset[str]


_MASS_FIELDS = {
    "stored_t",
    "vented_t",
    "captured_t",
    "in_transit_t",
    "overflow_risk_t",
    "injection_tph",
}
_COST_FIELDS = {
    "vessel_fuel",
    "conditioning",
    "reconditioning",
    "loading",
    "unloading",
    "operating_cost",
    "total_cost",
}
_EXPECTATION_FIELDS = {
    field.name for field in fields(ReplayExpectation) if field.name != "required_fields"
}


def compare_replay_snapshots(
    expected: ReplayExpectation,
    actual: ReplaySnapshot,
    *,
    tolerances: ReplayTolerances | None = None,
) -> tuple[bool, tuple[str, ...], frozenset[str]]:
    """Compare every supplied expectation and reject missing required fields."""

    tolerances = tolerances or ReplayTolerances()
    unknown_required = expected.required_fields - _EXPECTATION_FIELDS
    if unknown_required:
        names = ", ".join(sorted(unknown_required))
        raise ValueError(f"Unknown replay expectation fields: {names}")

    mismatches: list[str] = []
    compared: set[str] = set()
    for name in sorted(expected.required_fields):
        if getattr(expected, name) is None:
            mismatches.append(f"{name}: missing required expectation")

    for name in sorted(_EXPECTATION_FIELDS):
        expected_value = getattr(expected, name)
        if expected_value is None:
            continue
        compared.add(name)
        actual_value = getattr(actual, name)
        mismatches.extend(
            _value_mismatches(name, expected_value, actual_value, tolerances)
        )

    return not mismatches, tuple(mismatches), frozenset(compared)


def _value_mismatches(
    name: str,
    expected,
    actual,
    tolerances: ReplayTolerances,
) -> list[str]:
    if isinstance(expected, Mapping):
        return _mapping_mismatches(name, expected, actual, tolerances)
    if isinstance(expected, Sequence) and not isinstance(expected, (str, bytes)):
        return _sequence_mismatches(name, expected, actual, tolerances)
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        if name == "elapsed_hours":
            return [] if expected == actual else [f"{name}: expected {expected}, actual {actual}"]
        tolerance = _numeric_tolerance(name, tolerances)
        if math.isclose(float(expected), float(actual), rel_tol=0.0, abs_tol=tolerance):
            return []
        return [
            f"{name}: expected {expected}, actual {actual}, tolerance {tolerance}"
        ]
    return [] if expected == actual else [f"{name}: expected {expected!r}, actual {actual!r}"]


def _mapping_mismatches(
    name: str,
    expected: Mapping,
    actual,
    tolerances: ReplayTolerances,
) -> list[str]:
    if not isinstance(actual, Mapping):
        return [f"{name}: expected mapping, actual {type(actual).__name__}"]
    mismatches: list[str] = []
    missing = set(expected) - set(actual)
    unexpected = set(actual) - set(expected)
    if missing:
        mismatches.append(f"{name}: missing keys {sorted(missing)}")
    if unexpected:
        mismatches.append(f"{name}: unexpected keys {sorted(unexpected)}")
    for key in sorted(set(expected) & set(actual), key=str):
        child_name = f"{name}[{key}]"
        expected_value = expected[key]
        actual_value = actual[key]
        if isinstance(expected_value, (int, float)) and isinstance(actual_value, (int, float)):
            tolerance = tolerances.state_value
            if not math.isclose(
                float(expected_value),
                float(actual_value),
                rel_tol=0.0,
                abs_tol=tolerance,
            ):
                mismatches.append(
                    f"{child_name}: expected {expected_value}, actual {actual_value}, "
                    f"tolerance {tolerance}"
                )
        elif expected_value != actual_value:
            mismatches.append(
                f"{child_name}: expected {expected_value!r}, actual {actual_value!r}"
            )
    return mismatches


def _sequence_mismatches(
    name: str,
    expected: Sequence,
    actual,
    tolerances: ReplayTolerances,
) -> list[str]:
    if not isinstance(actual, Sequence) or isinstance(actual, (str, bytes)):
        return [f"{name}: expected sequence, actual {type(actual).__name__}"]
    if len(expected) != len(actual):
        return [f"{name}: expected length {len(expected)}, actual length {len(actual)}"]
    mismatches: list[str] = []
    tolerance = _numeric_tolerance(name, tolerances)
    for index, (expected_value, actual_value) in enumerate(zip(expected, actual)):
        if isinstance(expected_value, (int, float)) and isinstance(actual_value, (int, float)):
            if not math.isclose(
                float(expected_value),
                float(actual_value),
                rel_tol=0.0,
                abs_tol=tolerance,
            ):
                mismatches.append(
                    f"{name}[{index}]: expected {expected_value}, actual {actual_value}, "
                    f"tolerance {tolerance}"
                )
        elif expected_value != actual_value:
            mismatches.append(
                f"{name}[{index}]: expected {expected_value!r}, actual {actual_value!r}"
            )
    return mismatches


def _numeric_tolerance(name: str, tolerances: ReplayTolerances) -> float:
    if name in _MASS_FIELDS:
        return tolerances.mass_t
    if name in _COST_FIELDS:
        return tolerances.cost_eur
    if name == "total_reward":
        return tolerances.reward
    if name == "objective_value":
        return tolerances.objective
    return tolerances.state_value
