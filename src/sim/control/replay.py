"""Solver-neutral replay snapshots and exactness validation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
from dataclasses import dataclass, fields
import math
from numbers import Integral

from ..environment import CCSEnv


NativeAction = dict[str, list[int]]


def action_for_well_control_mode(
    env: CCSEnv,
    action: Mapping,
) -> NativeAction:
    """Adapt an internal solver trace to the environment control boundary."""

    adapted = {"vessels": [int(value) for value in action["vessels"]]}
    if not env.automatic_well_control:
        adapted["wells"] = [int(value) for value in action["wells"]]
    return adapted


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
) -> tuple[bool, tuple[str, ...]]:
    """Compare every supplied expectation and reject missing required fields."""

    exact, mismatches, _compared = _compare_replay_snapshots_detailed(
        expected,
        actual,
        tolerances=tolerances,
    )
    return exact, mismatches


def _compare_replay_snapshots_detailed(
    expected: ReplayExpectation,
    actual: ReplaySnapshot,
    *,
    tolerances: ReplayTolerances | None = None,
) -> tuple[bool, tuple[str, ...], frozenset[str]]:

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


def replay_native_actions(
    env: CCSEnv,
    native_actions: Sequence[NativeAction],
    *,
    horizon_h: int,
    expected: ReplayExpectation | None = None,
    tolerances: ReplayTolerances | None = None,
    copy_env: bool = True,
) -> ReplayValidationResult:
    """Replay native actions from the current state and validate the result."""

    if horizon_h <= 0:
        raise ValueError("horizon_h must be positive")
    replay_env = copy.deepcopy(env) if copy_env else env
    start_stored_t = float(replay_env.cumulative_stored_t)
    start_vented_t = float(replay_env.ledger.vented_t)
    start_captured_t = float(replay_env.cumulative_captured_t)
    start_ledger = copy.deepcopy(replay_env.ledger)

    violations: list[str] = []
    execution_mismatches: list[str] = []
    injection_tph: list[float] = []
    total_reward = 0.0
    overflow_risk_t = 0.0
    elapsed_hours = 0

    if len(native_actions) != horizon_h:
        execution_mismatches.append(
            f"horizon: expected {horizon_h} actions, actual {len(native_actions)}"
        )

    for step in range(min(horizon_h, len(native_actions))):
        action = native_actions[step]
        action_error = _native_action_error(replay_env, action, step)
        if action_error:
            execution_mismatches.append(action_error)
            break
        before_stored_t = float(replay_env.cumulative_stored_t)
        _obs, reward, terminated, truncated, info = replay_env.step(action)
        elapsed_hours += 1
        injection_tph.append(float(replay_env.cumulative_stored_t) - before_stored_t)
        total_reward += float(reward)
        overflow_risk_t += float(info.get("overflow_risk_t", 0.0))
        violations.extend(str(value) for value in info.get("violations", []))
        if terminated or truncated:
            if elapsed_hours < horizon_h:
                execution_mismatches.append(
                    f"horizon: replay ended after {elapsed_hours} of {horizon_h} hours"
                )
            break

    snapshot = _replay_snapshot(
        replay_env,
        elapsed_hours=elapsed_hours,
        total_reward=total_reward,
        overflow_risk_t=overflow_risk_t,
        injection_tph=tuple(injection_tph),
        start_stored_t=start_stored_t,
        start_vented_t=start_vented_t,
        start_captured_t=start_captured_t,
        start_ledger=start_ledger,
    )
    fatal_violations = {"berth_required", "bottomhole_pressure_clipped"}
    is_executable = (
        not execution_mismatches
        and elapsed_hours == horizon_h
        and not (set(violations) & fatal_violations)
    )

    compared_fields: frozenset[str] = frozenset()
    comparison_mismatches: tuple[str, ...] = ()
    comparison_exact = False
    if expected is not None:
        comparison_exact, comparison_mismatches, compared_fields = _compare_replay_snapshots_detailed(
            expected,
            snapshot,
            tolerances=tolerances,
        )
    mismatches = tuple(execution_mismatches) + comparison_mismatches
    return ReplayValidationResult(
        actual=snapshot,
        is_executable=is_executable,
        is_exact=is_executable and expected is not None and comparison_exact,
        violations=tuple(violations),
        mismatches=mismatches,
        compared_fields=compared_fields,
    )


def _native_action_error(env: CCSEnv, action, step: int) -> str:
    if not isinstance(action, Mapping):
        return f"action[{step}]: expected mapping, actual {type(action).__name__}"
    vessel_actions = action.get("vessels")
    well_actions = action.get("wells")
    if not isinstance(vessel_actions, Sequence) or isinstance(vessel_actions, (str, bytes)):
        return f"action[{step}].vessels: expected sequence"
    if not isinstance(well_actions, Sequence) or isinstance(well_actions, (str, bytes)):
        return f"action[{step}].wells: expected sequence"
    if len(vessel_actions) != len(env.vessel_ids):
        return (
            f"action[{step}].vessels dimension: expected {len(env.vessel_ids)}, "
            f"actual {len(vessel_actions)}"
        )
    if len(well_actions) != len(env.well_ids):
        return (
            f"action[{step}].wells dimension: expected {len(env.well_ids)}, "
            f"actual {len(well_actions)}"
        )
    for vessel_id, choice, mask in zip(
        env.vessel_ids,
        vessel_actions,
        env.vessel_action_mask(),
    ):
        if not isinstance(choice, Integral) or not (
            0 <= int(choice) < len(mask) and mask[int(choice)]
        ):
            return f"action[{step}] is not executable for {vessel_id}: {choice}"
    for well_id, choice, mask in zip(
        env.well_ids,
        well_actions,
        env.well_rate_action_mask(),
    ):
        if not isinstance(choice, Integral) or not (
            0 <= int(choice) < len(mask) and mask[int(choice)]
        ):
            return f"action[{step}] is not executable for {well_id}: {choice}"
    return ""


def _replay_snapshot(
    env: CCSEnv,
    *,
    elapsed_hours: int,
    total_reward: float,
    overflow_risk_t: float,
    injection_tph: tuple[float, ...],
    start_stored_t: float,
    start_vented_t: float,
    start_captured_t: float,
    start_ledger,
) -> ReplaySnapshot:
    state = env.simulator.state
    vessel_fuel = float(env.ledger.vessel_fuel) - float(start_ledger.vessel_fuel)
    conditioning = float(env.ledger.conditioning) - float(start_ledger.conditioning)
    reconditioning = float(env.ledger.reconditioning) - float(start_ledger.reconditioning)
    loading = float(env.ledger.loading) - float(start_ledger.loading)
    unloading = float(env.ledger.unloading) - float(start_ledger.unloading)
    operating_cost = vessel_fuel + conditioning + reconditioning + loading + unloading
    return ReplaySnapshot(
        elapsed_hours=elapsed_hours,
        stored_t=float(env.cumulative_stored_t) - start_stored_t,
        vented_t=float(env.ledger.vented_t) - start_vented_t,
        captured_t=float(env.cumulative_captured_t) - start_captured_t,
        in_transit_t=float(env._in_transit_inventory()),
        vessel_fuel=vessel_fuel,
        conditioning=conditioning,
        reconditioning=reconditioning,
        loading=loading,
        unloading=unloading,
        operating_cost=operating_cost,
        total_cost=float(env.ledger.total_cost) - float(start_ledger.total_cost),
        total_reward=total_reward,
        objective_value=-total_reward / float(env.config.reward_scale),
        overflow_risk_t=overflow_risk_t,
        injection_tph=injection_tph,
        entity_inventory_t={
            entity_id: float(state.entity_inventory_t.get(entity_id, 0.0))
            for entity_id in env.network.entities
        },
        vessel_berths={
            vessel_id: state.vessel_berths.get(vessel_id)
            for vessel_id in env.vessel_ids
        },
    )


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
