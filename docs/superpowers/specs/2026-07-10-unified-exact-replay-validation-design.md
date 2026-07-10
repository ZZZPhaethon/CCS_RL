# Unified Exact-Replay Validation Design

## Context

CPLEX, Trip MILP, Rolling MILP, and Native MPC all execute hourly native actions through `CCSEnv`, but they currently decide replay validity differently. Full CPLEX and Trip MILP share a stored-tonnes check, Rolling MILP treats a replay with no fatal violation as valid, and the Native MPC experiment compares two `EpisodeMetrics` objects. These meanings are close enough to look interchangeable but are not strong enough to support one common `replay_ok` or headroom claim.

## Decision

Create a solver-neutral replay and validation module at `src/sim/control/replay.py`, with tests in `tests/test_control_replay.py`. The module will not import CPLEX, Trip MILP, Rolling MILP, Native MPC, or experiment code. Solver-specific functions remain as compatibility adapters that translate their result objects into common replay expectations.

This keeps the dependency direction one-way:

```text
CPLEX / Trip MILP / Rolling MILP / Native MPC / experiments
                              |
                              v
                    sim.control.replay
                              |
                              v
                            CCSEnv
```

## Alternatives Considered

### Extend `metrics.py`

Rejected. `metrics.py` is the common reporting scorecard for completed episodes. Solver prediction comparison, action executability, partial-horizon replay, and mismatch diagnostics are control-validation concerns rather than reporting concerns.

### Keep one validator per solver

Rejected. This preserves the current ambiguity: identical `replay_ok` labels would continue to mean different things, and future metrics would have to be added in several places.

### Add `control/replay.py` and retain thin adapters

Selected. One module owns action replay, snapshots, tolerances, and comparison. Existing public entry points remain available while their implementations delegate to the common module.

## Public Data Model

`replay.py` will expose four immutable data types:

```python
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
```

`ReplayExpectation.required_fields` prevents partial comparisons from being presented as exact. Non-`None` fields are compared; a required field whose expectation is `None` is reported as missing. A caller that only predicts stored tonnes can still receive a diagnostic stored gap, but `is_exact` is false unless every field required by that integration is supplied and matches.

## Public Operations

The module will expose:

```python
NativeAction = dict[str, list[int]]


def replay_native_actions(
    env: CCSEnv,
    native_actions: Sequence[NativeAction],
    *,
    horizon_h: int,
    expected: ReplayExpectation | None = None,
    tolerances: ReplayTolerances | None = None,
    copy_env: bool = True,
) -> ReplayValidationResult:
    ...


def compare_replay_snapshots(
    expected: ReplayExpectation,
    actual: ReplaySnapshot,
    *,
    tolerances: ReplayTolerances | None = None,
) -> tuple[bool, tuple[str, ...]]:
    ...
```

`copy_env=True` is the safe default: replay starts from the caller's current state without mutating the live controller environment. Compatibility wrappers that historically consumed their environment may explicitly pass `copy_env=False` until their callers are migrated.

## Replay Execution Semantics

For every requested hour, the common runner will:

1. Require exactly one vessel choice per vessel and one well choice per well.
2. Validate every choice against the current action mask before calling `env.step`.
3. Execute the native action in `CCSEnv` and accumulate reward, overflow risk, violations, and hourly stored increments.
4. Stop if the environment terminates or truncates, then report a horizon mismatch if fewer than `horizon_h` actions executed.
5. Capture metric deltas relative to the supplied starting state and a normalized final physical-state snapshot.

Fatal execution violations are initially `berth_required` and `bottomhole_pressure_clipped`, matching current solver replay behavior. Other violations remain visible in the report without automatically making the trace non-executable.

## Exactness Semantics

The result separates two questions:

- `is_executable`: the entire requested horizon ran, action dimensions and masks were valid at every step, and no fatal violation occurred.
- `is_exact`: an expectation was supplied, the trace is executable, every required expectation is present, and every supplied expected field matches the replay snapshot within its field-specific tolerance.

The common headroom exactness set is:

```text
elapsed_hours
stored_t
vented_t
captured_t
in_transit_t
operating_cost
total_cost
objective_value
overflow_risk_t
```

Native MPC deterministic replay additionally requires `entity_inventory_t`, `vessel_berths`, and the complete `injection_tph` sequence. This preserves its current full-episode metrics comparison and strengthens it with physical final-state checks.

Mismatch messages name the field, expected value, actual value, and tolerance. Reports never silently replace invalid actions with `WAIT`.

## Integration Plan

### Full CPLEX

`replay_full_scenario_cplex_plan()` remains public. It builds a `ReplayExpectation` from `FullScenarioCplexMilpResult`, calls `replay_native_actions()`, and returns a compatibility result exposing existing fields plus `is_exact` and mismatch diagnostics. CPLEX result extraction must expose every field in the common headroom exactness set before an exact claim is possible.

### Trip MILP

`replay_trip_milp_plan()` remains a public adapter. `materialize_native_action_trace()` uses the common replay snapshot rather than owning another environment-step loop. Trip result construction may transform the common snapshot into trip records, but it may not redefine executability or exactness.

### Rolling MILP

The rolling planner continues to produce model-predicted metrics and native actions. It validates those actions through the common runner and stores both prediction and replay diagnostics. An invalid or inexact plan is surfaced; vessel actions are not silently replaced with `WAIT`.

### Native MPC

Candidate generation may continue to roll policies forward in a copied environment. The chosen candidate's rollout snapshot becomes the expectation, and a fresh replay of the recorded actions is checked by the common validator with the deterministic fields enabled. The experiment removes its private `_metrics_match()` implementation.

## Backward Compatibility

The first migration keeps these functions and their existing call shapes:

- `replay_full_scenario_cplex_plan()`
- `replay_trip_milp_plan()`
- `materialize_native_action_trace()`

Existing `stored_tol_t` arguments map to `ReplayTolerances.mass_t`. Existing fields such as `stored_gap_t` and `is_executable` remain available. New code should consume `ReplayValidationResult` directly.

## Error Handling

Wrong action dimensions, out-of-range choices, and mask-invalid choices are reported as non-executable mismatch diagnostics. The common validator does not invent a fallback action. Caller programming errors such as a negative horizon raise `ValueError` before replay begins.

## Testing Strategy

Tests use small deterministic `CCSEnv` fixtures and follow red-green development:

1. Replaying a valid native trace for the requested horizon produces an executable snapshot without mutating the source environment.
2. Wrong dimensions, mask-invalid actions, fatal violations, and short traces produce non-executable reports with specific diagnostics.
3. Scalar tolerance boundaries are checked for mass, cost, reward, and objective fields.
4. Aggregate metrics can match while entity inventories differ; deterministic Native MPC validation must reject that trace.
5. Existing CPLEX and Trip replay entry points retain their public fields while using the common report.
6. Rolling MILP exposes prediction/replay mismatches instead of silently substituting `WAIT`.
7. Native MPC's old metrics comparison and the common validator agree on current passing fixtures before the private comparator is removed.

## Non-Goals

- This change does not decide whether Native MPC remains heuristic rollout or becomes a native-action MILP.
- This change does not redefine economic versus vent-first objective ordering.
- This change does not change canonical scenario capacities or weather distributions.
- This change does not claim solver model fidelity; it only measures and reports fidelity consistently.

## Success Criteria

- CPLEX, Trip MILP, Rolling MILP, and Native MPC all route replay validity through `sim.control.replay`.
- `is_executable` and `is_exact` have the same meaning at every call site.
- Headroom code excludes any result that is not exact for the common headroom fields.
- Existing public replay functions remain callable during migration.
- Focused replay, solver, rolling, MPC experiment, and full regression tests pass apart from the already documented missing `milk_run_stress.json` fixture.
