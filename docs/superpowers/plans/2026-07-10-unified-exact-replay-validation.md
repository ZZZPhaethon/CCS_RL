# Unified Exact-Replay Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement one solver-neutral native-action replay validator and migrate Native MPC, Full CPLEX, Trip MILP, and Rolling MILP to consistent `is_executable` and `is_exact` semantics.

**Architecture:** `src/sim/control/replay.py` owns replay execution, snapshots, tolerances, and comparison. Existing solver replay functions remain compatibility adapters. Native MPC remains heuristic rollout with the fixed lexicographic objective `vent -> end_unstored_inventory -> operating_cost`.

**Tech Stack:** Python 3.10+, dataclasses, `CCSEnv`, unittest/pytest, existing PuLP/CPLEX adapters.

## Global Constraints

- Preserve all pre-existing user work in the dirty worktree; stage only explicit paths or reviewed hunks.
- Do not change Native MPC into a MILP.
- Native MPC objective is fixed to `vent -> end_unstored_inventory -> operating_cost`; remove alternate objective modes.
- Keep the 15,000 t buffer in `northern_lights_phase1_3vessels.json`.
- Do not treat the missing `milk_run_stress.json` fixture as a task issue or blocker.
- Do not change economic/vent-first objective definitions in this implementation plan.
- Existing public replay functions remain callable.

---

### Task 1: Snapshot Comparison Contract

**Files:**
- Create: `src/sim/control/replay.py`
- Create: `tests/test_control_replay.py`

**Interfaces:**
- Produces: `ReplayTolerances`, `ReplaySnapshot`, `ReplayExpectation`, `ReplayValidationResult`, and `compare_replay_snapshots()`.
- Consumes: no solver-specific result types.

- [ ] **Step 1: Write failing comparison tests**

Add tests that construct snapshots directly and assert:

```python
def test_exact_requires_every_required_field_to_be_present_and_equal():
    expected = ReplayExpectation(
        required_fields=frozenset({"elapsed_hours", "stored_t", "vented_t"}),
        elapsed_hours=2,
        stored_t=10.0,
        vented_t=0.0,
    )
    exact, mismatches, compared = compare_replay_snapshots(expected, snapshot(stored_t=10.0))
    assert exact
    assert not mismatches
    assert compared == expected.required_fields


def test_missing_required_expectation_is_not_exact():
    expected = ReplayExpectation(
        required_fields=frozenset({"stored_t", "objective_value"}),
        stored_t=10.0,
    )
    exact, mismatches, _ = compare_replay_snapshots(expected, snapshot(stored_t=10.0))
    assert not exact
    assert any("objective_value" in mismatch for mismatch in mismatches)


def test_field_specific_tolerance_reports_named_mismatch():
    expected = ReplayExpectation(
        required_fields=frozenset({"stored_t"}),
        stored_t=10.0,
    )
    exact, mismatches, _ = compare_replay_snapshots(
        expected,
        snapshot(stored_t=10.01),
        tolerances=ReplayTolerances(mass_t=1e-3),
    )
    assert not exact
    assert "stored_t" in mismatches[0]
```

- [ ] **Step 2: Verify RED**

Run:

```powershell
python -m pytest -q tests/test_control_replay.py
```

Expected: collection fails because `sim.control.replay` does not exist.

- [ ] **Step 3: Implement immutable types and comparison**

Implement explicit field-to-tolerance routing:

```python
_MASS_FIELDS = {"stored_t", "vented_t", "captured_t", "in_transit_t", "overflow_risk_t"}
_COST_FIELDS = {
    "vessel_fuel", "conditioning", "reconditioning", "loading", "unloading",
    "operating_cost", "total_cost",
}


def compare_replay_snapshots(expected, actual, *, tolerances=None):
    tolerances = tolerances or ReplayTolerances()
    # Report missing required values before comparing supplied values.
    # Compare sequences element-by-element and mappings by identical keys.
    # Return (is_exact, mismatches, compared_fields).
```

Non-numeric fields require equality; numeric fields use `math.isclose(rel_tol=0.0, abs_tol=...)`.

- [ ] **Step 4: Verify GREEN**

Run `python -m pytest -q tests/test_control_replay.py`.

Expected: all comparison tests pass.

- [ ] **Step 5: Commit the isolated core**

```powershell
git add -- src/sim/control/replay.py tests/test_control_replay.py
git commit -m "Add replay snapshot validation"
```

---

### Task 2: Native-Action Replay Runner

**Files:**
- Modify: `src/sim/control/replay.py`
- Modify: `tests/test_control_replay.py`

**Interfaces:**
- Consumes: `CCSEnv`, a sequence of `{"vessels": [...], "wells": [...]}` actions, a positive `horizon_h`, optional expectations and tolerances.
- Produces: `replay_native_actions(...) -> ReplayValidationResult`.

- [ ] **Step 1: Write failing environment replay tests**

Use a deterministic one-emitter/one-vessel/one-well environment and assert:

```python
def test_replay_runs_full_horizon_without_mutating_source_env():
    env = make_replay_env(hours=2)
    env.reset(seed=1)
    before = env.simulator.state.copy()
    result = replay_native_actions(
        env,
        [{"vessels": [VESSEL_WAIT], "wells": [0]}] * 2,
        horizon_h=2,
    )
    assert result.is_executable
    assert result.actual.elapsed_hours == 2
    assert env.simulator.state.as_dict() == before.as_dict()


def test_wrong_action_dimension_is_non_executable_without_stepping():
    result = replay_native_actions(
        env,
        [{"vessels": [], "wells": [0]}],
        horizon_h=1,
    )
    assert not result.is_executable
    assert "dimension" in result.mismatches[0]


def test_short_trace_is_non_executable():
    result = replay_native_actions(env, [], horizon_h=1)
    assert not result.is_executable
    assert any("horizon" in mismatch for mismatch in result.mismatches)
```

Also test a mask-invalid choice and `horizon_h <= 0`.

- [ ] **Step 2: Verify RED**

Run `python -m pytest -q tests/test_control_replay.py`.

Expected: replay-runner tests fail because `replay_native_actions` is missing.

- [ ] **Step 3: Implement the runner**

Implementation requirements:

```python
def replay_native_actions(..., copy_env=True):
    if horizon_h <= 0:
        raise ValueError("horizon_h must be positive")
    replay_env = copy.deepcopy(env) if copy_env else env
    # Capture starting cumulative metrics and ledger components.
    # Validate dimensions and masks before each env.step.
    # Record per-hour stored delta, reward, overflow risk, and violations.
    # Capture normalized final inventories and vessel berths.
    # Compare against expected only after execution diagnostics are complete.
```

Fatal violations are exactly `berth_required` and `bottomhole_pressure_clipped`.

- [ ] **Step 4: Verify GREEN**

Run `python -m pytest -q tests/test_control_replay.py`.

Expected: all comparison and runner tests pass.

- [ ] **Step 5: Commit**

```powershell
git add -- src/sim/control/replay.py tests/test_control_replay.py
git commit -m "Replay native actions consistently"
```

---

### Task 3: Full CPLEX and Trip Compatibility Adapters

**Files:**
- Modify: `src/sim/control/cplex_milp.py`
- Modify: `src/sim/control/trip_milp.py`
- Modify: `tests/test_cplex_milp.py`
- Modify: `tests/test_trip_milp.py`

**Interfaces:**
- Keeps: `replay_full_scenario_cplex_plan(env, result, stored_tol_t=...)`.
- Keeps: `replay_trip_milp_plan(env, result, stored_tol_t=...)`.
- Adds compatibility fields: `is_exact`, `mismatches`, and `compared_fields`.

- [ ] **Step 1: Write failing adapter tests**

Extend non-CPLEX-dependent replay tests using a manually constructed result:

```python
replay = replay_full_scenario_cplex_plan(env, result)
assert replay.is_executable
assert replay.is_exact
assert replay.mismatches == []
```

Perturb expected `vented_t` while retaining matching `stored_t` and assert `is_executable` remains true but `is_exact` is false with a `vented_t` mismatch. Add the equivalent Trip adapter assertion.

- [ ] **Step 2: Verify RED**

Run:

```powershell
python -m pytest -q tests/test_cplex_milp.py tests/test_trip_milp.py -k "replay or materialization"
```

Expected: failures because compatibility results do not expose exactness diagnostics.

- [ ] **Step 3: Delegate replay execution to the common runner**

`replay_full_scenario_cplex_plan()` builds `ReplayExpectation` from the solver result, calls `replay_native_actions(copy_env=False)`, and maps the report back to the existing result type. `stored_tol_t` becomes `ReplayTolerances.mass_t`.

`replay_trip_milp_plan()` remains a thin call to the CPLEX compatibility adapter. It must not define separate exactness rules.

- [ ] **Step 4: Verify GREEN and compatibility**

Run the focused command from Step 2, then:

```powershell
python -m pytest -q tests/test_cplex_milp.py tests/test_trip_milp.py
```

Expected: both modules pass, subject only to existing external CPLEX skips.

- [ ] **Step 5: Review hunks before staging**

Because these files contain pre-existing user changes, inspect `git diff` and stage only replay-related hunks. Do not commit unrelated objective changes in this task.

---

### Task 4: Trip Materialization and Rolling MILP Replay

**Files:**
- Modify: `src/sim/control/trip_milp.py`
- Modify: `src/sim/control/rolling_milp.py`
- Modify: `tests/test_trip_milp.py`
- Modify: `tests/test_rolling_milp.py`

**Interfaces:**
- `materialize_native_action_trace()` derives physical/economic metrics from `ReplayValidationResult.actual`.
- `RollingMilpPlan` exposes `replay_is_exact` and `replay_mismatches` while retaining `replay_is_valid` as the compatibility executability flag.

- [ ] **Step 1: Write failing materialization and rolling tests**

Add assertions that materialization and direct common replay produce identical stored, vented, cost, objective, and injection sequences. Add a rolling test where predicted and replay vent differ: the plan remains executable but is not exact and names the mismatched field.

- [ ] **Step 2: Verify RED**

Run:

```powershell
python -m pytest -q tests/test_trip_milp.py tests/test_rolling_milp.py -k "materializ or exact_replay or replay_mismatch"
```

- [ ] **Step 3: Replace the duplicate Trip step loop**

Call `replay_native_actions(copy_env=True)` and map `actual` into `TripMilpResult`. Preserve trip-record reconstruction and public metrics, but do not recompute replay validity separately.

- [ ] **Step 4: Wire Rolling MILP diagnostics**

Copy common executability, exactness, and mismatches into `RollingMilpPlan`. When executing a planned vessel action, an absent, expired, or mask-invalid plan action raises `RuntimeError`; it is not silently replaced with `VESSEL_WAIT`.

- [ ] **Step 5: Verify GREEN**

Run full Trip and Rolling modules. Expected: pass with existing PuLP deprecation warnings only.

- [ ] **Step 6: Review hunks before staging**

Do not stage unrelated solver-objective hunks as part of the replay migration commit.

---

### Task 5: Native MPC Parallel Validation and Fixed Objective

**Files:**
- Modify: `src/sim/control/native_mpc.py`
- Modify: `experiments/rolling_native_mpc_headroom.py`
- Modify: `tests/test_rolling_milp.py`
- Modify: `tests/test_controller_comparison_experiment.py`

**Interfaces:**
- Native MPC remains heuristic candidate rollout.
- Candidate ordering is fixed to `(vented_t, end_unstored_t, operating_cost)`.
- The experiment records common `is_executable`, `is_exact`, and mismatch diagnostics.

- [ ] **Step 1: Write characterization tests for the fixed objective**

Replace alternate-mode tests with:

```python
best = min([low_inventory, low_cost], key=RollingNativeMpcController._candidate_key)
assert best.name == "low_inventory"
```

Assert the constructor and CLI no longer accept an alternate objective mode.

- [ ] **Step 2: Write a failing new/old parallel-validation test**

For the existing 24-hour deterministic experiment fixture, compare the legacy `_metrics_match()` result with the common replay report and assert both are true. Perturb final entity inventory without changing aggregate `EpisodeMetrics`; assert legacy matching remains true while common `is_exact` is false.

- [ ] **Step 3: Verify RED**

Run:

```powershell
python -m pytest -q tests/test_rolling_milp.py tests/test_controller_comparison_experiment.py -k "native_mpc"
```

- [ ] **Step 4: Run old and new validators in parallel**

Keep `_metrics_match()` temporarily. Build a deterministic `ReplayExpectation` from the recorded rollout metrics and final state, replay the actions with `replay_native_actions()`, and expose both results. Raise a diagnostic error if the legacy and common aggregate verdicts disagree.

- [ ] **Step 5: Remove the private comparator after parity passes**

After focused and 24-hour smoke tests show parity, remove `_metrics_match()` and make `row["replay_ok"]` equal the common `is_exact`. Include mismatch text in the output row.

- [ ] **Step 6: Verify GREEN**

Run the focused tests and:

```powershell
python experiments/rolling_native_mpc_headroom.py --hours 24 --seeds 1 --disturbance-profile none --output-dir $env:TEMP\ccs_replay_smoke
```

Expected: the experiment completes and reports `replay=True` using the common validator.

- [ ] **Step 7: Review hunks before staging**

Confirm that no native-action MILP implementation or alternate objective mode remains in this batch.

---

### Task 6: Regression Verification and Review

**Files:**
- Verify all files touched above.

- [ ] **Step 1: Run focused replay/control tests**

```powershell
python -m pytest -q tests/test_control_replay.py tests/test_cplex_milp.py tests/test_trip_milp.py tests/test_rolling_milp.py tests/test_controller_comparison_experiment.py
```

- [ ] **Step 2: Run the repository suite**

```powershell
python -m pytest -q -k "not test_controller_comparison_uses_scenario_recommended_cap_hours_when_omitted"
```

The deselected milk-run fixture test is outside this task and is not reported as a blocker.

- [ ] **Step 3: Run diff validation**

```powershell
git diff --check
git status --short
```

- [ ] **Step 4: Request read-only code review**

Review the implementation against `docs/superpowers/specs/2026-07-10-unified-exact-replay-validation-design.md`. Fix all Critical and Important replay-validation findings before committing integration hunks.

- [ ] **Step 5: Commit reviewed logical batches**

Stage the isolated replay core first, then reviewed compatibility/integration hunks. Keep `task_plan.md`, `findings.md`, and `progress.md` untracked.
