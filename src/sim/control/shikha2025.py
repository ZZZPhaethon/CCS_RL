"""Shikha et al. (2025) decomposition adapted to this repository's MILP.

The paper's Case 2 method combines vessel-wise spatial Lagrangean
decomposition, a shrinking-horizon solve for each vessel subproblem, and a
full-space feasibility recovery with low-impact route binaries fixed.  This
module preserves that algorithmic structure while using the repository's
native-action CPLEX formulation and physical/economic parameters.

The original numerical instance data are not public.  Consequently, this is
an algorithm reproduction on the repository's scenarios, not a reproduction
of the paper's reported tables.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace
import math
import time

from ..entities.emitter import Emitter
from ..entities.manifold import SubseaManifold
from ..entities.pipeline import Pipeline
from ..entities.storage import InjectionWell
from ..entities.terminal import Terminal
from ..scenario_generation import Scenario
from .cplex_milp import (
    FullScenarioCplexMilpResult,
    _terminal_berth_counts,
    solve_full_scenario_with_cplex,
)
from .rolling_milp import greedy_warm_start_actions


@dataclass(frozen=True)
class Shikha2025Config:
    """Algorithm settings matching the paper unless noted otherwise."""

    active_window_h: int = 120
    fix_window_h: int = 60
    max_iterations: int = 18
    tolerance_rel: float = 0.02
    step_size: float = 1.0
    subproblem_time_limit_s: float | None = None
    repair_time_limit_s: float | None = None
    mip_gap_rel: float | None = None
    threads: int | None = None
    terminal_cleanup_value: bool = True
    initial_multiplier_eur: float = 0.0

    def __post_init__(self) -> None:
        if self.active_window_h <= 0:
            raise ValueError("active_window_h must be positive")
        if self.fix_window_h <= 0:
            raise ValueError("fix_window_h must be positive")
        if self.fix_window_h > self.active_window_h:
            raise ValueError("fix_window_h cannot exceed active_window_h")
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if not 0.0 < self.tolerance_rel < 1.0:
            raise ValueError("tolerance_rel must lie between zero and one")
        if self.step_size <= 0.0 or not math.isfinite(self.step_size):
            raise ValueError("step_size must be finite and positive")
        for name in ("subproblem_time_limit_s", "repair_time_limit_s"):
            value = getattr(self, name)
            if value is not None and value <= 0.0:
                raise ValueError(f"{name} must be positive when provided")
        if self.mip_gap_rel is not None and self.mip_gap_rel < 0.0:
            raise ValueError("mip_gap_rel must be non-negative")
        if self.threads is not None and self.threads <= 0:
            raise ValueError("threads must be positive when provided")
        if not math.isfinite(self.initial_multiplier_eur):
            raise ValueError("initial_multiplier_eur must be finite")


@dataclass(frozen=True)
class Shikha2025SubproblemDiagnostic:
    vessel_id: str
    shrinking_stage_count: int
    statuses: tuple[str, ...]
    augmented_objective: float
    wall_time_s: float


@dataclass(frozen=True)
class Shikha2025IterationDiagnostic:
    iteration: int
    surrogate_dual_objective: float
    best_feasible_objective: float
    relative_surrogate_gap: float
    maximum_service_violation: float
    multiplier_norm: float
    repair_status: str
    repair_is_valid: bool
    subproblems: tuple[Shikha2025SubproblemDiagnostic, ...]
    wall_time_s: float


@dataclass(frozen=True)
class Shikha2025Result:
    feasible_result: FullScenarioCplexMilpResult
    iterations: tuple[Shikha2025IterationDiagnostic, ...]
    converged: bool
    stopping_reason: str
    horizon_h: int
    wall_time_s: float
    multipliers_eur_by_node_hour: dict[tuple[str, int], float] = field(
        default_factory=dict
    )


def shrinking_horizon_stages(
    horizon_h: int,
    active_window_h: int = 120,
    fix_window_h: int = 60,
) -> tuple[tuple[int, int], ...]:
    """Return ``(fixed_prefix, active_end)`` pairs from paper Figure 6."""

    horizon_h = int(horizon_h)
    active_window_h = int(active_window_h)
    fix_window_h = int(fix_window_h)
    if horizon_h <= 0:
        raise ValueError("horizon_h must be positive")
    if active_window_h <= 0 or fix_window_h <= 0:
        raise ValueError("shrinking-horizon windows must be positive")
    if fix_window_h > active_window_h:
        raise ValueError("fix_window_h cannot exceed active_window_h")
    stages: list[tuple[int, int]] = []
    fixed_prefix_h = 0
    while True:
        active_end_h = min(horizon_h, fixed_prefix_h + active_window_h)
        stages.append((fixed_prefix_h, active_end_h))
        if active_end_h >= horizon_h:
            return tuple(stages)
        fixed_prefix_h += fix_window_h


def projected_subgradient_update(
    multipliers: dict[tuple[str, int], float],
    residuals: dict[tuple[str, int], float],
    objective_gap: float,
    step_size: float,
) -> dict[tuple[str, int], float]:
    """Projected minimization-form counterpart of paper equation (35)."""

    norm_squared = sum(float(value) ** 2 for value in residuals.values())
    if norm_squared <= 1e-12 or objective_gap <= 0.0:
        return dict(multipliers)
    scale = float(step_size) * float(objective_gap) / norm_squared
    return {
        key: max(
            0.0,
            float(multipliers.get(key, 0.0))
            + scale * float(residuals.get(key, 0.0)),
        )
        for key in set(multipliers) | set(residuals)
    }


def solve_shikha2025(
    env,
    *,
    scenario: Scenario | None = None,
    horizon_h: int | None = None,
    config: Shikha2025Config | None = None,
    progress=None,
) -> Shikha2025Result:
    """Solve one scenario with the paper's vessel-wise decomposition pattern."""

    config = config or Shikha2025Config()
    scenario = scenario or getattr(env, "scenario", None)
    if scenario is None:
        raise ValueError("Pass a Scenario or call env.reset(seed=...) first")
    if getattr(env, "simulator", None) is None:
        raise ValueError("Call env.reset(seed=...) before solving")
    if horizon_h is None:
        start_step = scenario.step_index(env.simulator.state.time_h)
        horizon_h = scenario.n_steps - start_step
    horizon_h = int(horizon_h)
    if horizon_h <= 0:
        raise ValueError("horizon_h must be positive")
    vessel_ids = tuple(env.vessel_ids)
    if len(vessel_ids) < 2:
        raise ValueError("Shikha2025 decomposition requires at least two vessels")

    started = time.perf_counter()
    capacities = _service_capacities(env, scenario, horizon_h)
    multipliers = {
        key: max(0.0, float(config.initial_multiplier_eur))
        for key in capacities
    }
    best = _repair_routes(
        env,
        scenario,
        horizon_h,
        greedy_warm_start_actions(env, horizon_h),
        config,
        fix_routes=True,
    )
    if not best.is_valid:
        best = _repair_routes(
            env,
            scenario,
            horizon_h,
            greedy_warm_start_actions(env, horizon_h),
            config,
            fix_routes=False,
        )
    if not best.is_valid:
        raise RuntimeError(
            best.validation_error
            or f"initial full-space feasibility recovery failed: {best.status}"
        )

    diagnostics: list[Shikha2025IterationDiagnostic] = []
    converged = False
    stopping_reason = "maximum_iterations"
    allocation_share = 1.0 / len(vessel_ids)
    for iteration in range(1, config.max_iterations + 1):
        iteration_started = time.perf_counter()
        subproblem_results: dict[str, FullScenarioCplexMilpResult] = {}
        subproblem_diagnostics: list[Shikha2025SubproblemDiagnostic] = []
        for vessel_id in vessel_ids:
            if progress is not None:
                progress(
                    f"Shikha2025 iteration {iteration}: vessel {vessel_id}"
                )
            sub_env, sub_scenario = _single_vessel_problem(
                env,
                scenario,
                vessel_id,
                allocation_share,
            )
            result, diagnostic = _solve_shrinking_subproblem(
                sub_env,
                sub_scenario,
                horizon_h,
                multipliers,
                config,
            )
            if not result.is_valid:
                raise RuntimeError(
                    result.validation_error
                    or f"subproblem {vessel_id} failed: {result.status}"
                )
            subproblem_results[vessel_id] = result
            subproblem_diagnostics.append(diagnostic)

        merged_actions = merge_subproblem_actions(
            env,
            horizon_h,
            subproblem_results,
        )
        repair = _repair_routes(
            env,
            scenario,
            horizon_h,
            merged_actions,
            config,
            fix_routes=True,
        )
        if repair.is_valid and (
            repair.augmented_objective_value
            < best.augmented_objective_value - 1e-6
        ):
            best = repair

        surrogate = sum(
            result.augmented_objective_value
            for result in subproblem_results.values()
        ) - sum(
            multipliers[key] * capacity
            for key, capacity in capacities.items()
        )
        # The repository's common terminal-cleanup value is evaluated only in
        # full-space recovery.  It is not part of the paper's RTN horizon, so
        # use the within-horizon objective for the decomposition gap.
        best_objective = float(best.objective_value)
        objective_gap = abs(best_objective - surrogate)
        relative_gap = objective_gap / max(1.0, abs(best_objective))
        residuals = _service_residuals(
            capacities,
            subproblem_results.values(),
        )
        maximum_violation = max(
            (max(0.0, value) for value in residuals.values()),
            default=0.0,
        )
        diagnostics.append(
            Shikha2025IterationDiagnostic(
                iteration=iteration,
                surrogate_dual_objective=float(surrogate),
                best_feasible_objective=best_objective,
                relative_surrogate_gap=float(relative_gap),
                maximum_service_violation=float(maximum_violation),
                multiplier_norm=math.sqrt(
                    sum(value * value for value in multipliers.values())
                ),
                repair_status=repair.status,
                repair_is_valid=bool(repair.is_valid),
                subproblems=tuple(subproblem_diagnostics),
                wall_time_s=time.perf_counter() - iteration_started,
            )
        )
        if repair.is_valid and relative_gap <= config.tolerance_rel:
            converged = True
            stopping_reason = "surrogate_gap_tolerance"
            break
        if repair.is_valid and maximum_violation <= 1e-6:
            stopping_reason = "service_coupling_consistent"
            break
        multipliers = projected_subgradient_update(
            multipliers,
            residuals,
            objective_gap,
            config.step_size,
        )

    return Shikha2025Result(
        feasible_result=best,
        iterations=tuple(diagnostics),
        converged=converged,
        stopping_reason=stopping_reason,
        horizon_h=horizon_h,
        wall_time_s=time.perf_counter() - started,
        multipliers_eur_by_node_hour=dict(multipliers),
    )


def merge_subproblem_actions(
    env,
    horizon_h: int,
    subproblem_results: dict[str, FullScenarioCplexMilpResult],
) -> list[dict[str, list[int]]]:
    """Merge one-vessel schedules into a full-model MIP start."""

    merged = greedy_warm_start_actions(env, horizon_h)
    for vessel_id, result in subproblem_results.items():
        if vessel_id not in env.vessel_ids:
            raise ValueError(f"Unknown vessel in subproblem results: {vessel_id}")
        vessel_index = env.vessel_ids.index(vessel_id)
        actions = result.vessel_actions_by_hour.get(vessel_id)
        if actions is None or len(actions) < horizon_h:
            raise ValueError(f"Incomplete subproblem actions for {vessel_id}")
        for hour in range(horizon_h):
            merged[hour]["vessels"][vessel_index] = int(actions[hour])
    return merged


def _solve_shrinking_subproblem(
    env,
    scenario: Scenario,
    horizon_h: int,
    multipliers: dict[tuple[str, int], float],
    config: Shikha2025Config,
) -> tuple[FullScenarioCplexMilpResult, Shikha2025SubproblemDiagnostic]:
    warm_start = greedy_warm_start_actions(env, horizon_h)
    statuses: list[str] = []
    vessel_id = env.vessel_ids[0]
    started = time.perf_counter()
    result = None
    stages = shrinking_horizon_stages(
        horizon_h,
        config.active_window_h,
        config.fix_window_h,
    )
    for fixed_prefix_h, active_end_h in stages:
        result = solve_full_scenario_with_cplex(
            env,
            scenario=scenario,
            horizon_h=horizon_h,
            economics=env.cost_model.parameters,
            warm_start_native_actions_by_hour=warm_start,
            time_limit_s=config.subproblem_time_limit_s,
            mip_gap_rel=config.mip_gap_rel,
            threads=config.threads,
            cplex_options=[
                "set parallel 1",
                "set simplex tolerances feasibility 1e-7",
            ],
            economic_objective=True,
            environment_aligned_service=True,
            terminal_cleanup_value=False,
            terminal_cleanup_mip_start_mode="complete",
            cleanup_unary_trip_slots=True,
            vessel_visit_load_cuts=True,
            source_visit_vent_cuts=True,
            terminal_visit_cuts=True,
            service_reachability_cuts=True,
            route_cargo_flow_linking=True,
            fix_warm_start_vessel_routes_through_h=(
                fixed_prefix_h if fixed_prefix_h > 0 else None
            ),
            integrality_relax_after_h=(
                active_end_h if active_end_h < horizon_h else None
            ),
            lagrangian_service_price_eur_by_node_hour=multipliers,
        )
        statuses.append(result.status)
        if result.status not in {"Optimal", "Integer Feasible"}:
            break
        warm_start = result.native_actions_by_hour
    assert result is not None
    return result, Shikha2025SubproblemDiagnostic(
        vessel_id=vessel_id,
        shrinking_stage_count=len(statuses),
        statuses=tuple(statuses),
        augmented_objective=float(result.augmented_objective_value),
        wall_time_s=time.perf_counter() - started,
    )


def _repair_routes(
    env,
    scenario: Scenario,
    horizon_h: int,
    warm_start,
    config: Shikha2025Config,
    *,
    fix_routes: bool,
) -> FullScenarioCplexMilpResult:
    return solve_full_scenario_with_cplex(
        env,
        scenario=scenario,
        horizon_h=horizon_h,
        economics=env.cost_model.parameters,
        warm_start_native_actions_by_hour=warm_start,
        time_limit_s=config.repair_time_limit_s,
        mip_gap_rel=config.mip_gap_rel,
        threads=config.threads,
        cplex_options=[
            "set parallel 1",
            "set simplex tolerances feasibility 1e-7",
        ],
        economic_objective=True,
        environment_aligned_service=True,
        terminal_cleanup_value=config.terminal_cleanup_value,
        terminal_cleanup_mip_start_mode="complete",
        cleanup_unary_trip_slots=True,
        vessel_visit_load_cuts=True,
        source_visit_vent_cuts=True,
        terminal_visit_cuts=True,
        service_reachability_cuts=True,
        route_cargo_flow_linking=True,
        fix_warm_start_vessel_routes=fix_routes,
    )


def _single_vessel_problem(
    env,
    scenario: Scenario,
    vessel_id: str,
    shared_resource_fraction: float,
):
    sub_env = copy.deepcopy(env)
    sub_scenario = copy.deepcopy(scenario)
    sub_env.vessel_ids = [vessel_id]
    sub_env._routes = {vessel_id: sub_env._routes[vessel_id]}
    fraction = float(shared_resource_fraction)
    for entity_id, entity in list(sub_env.network.entities.items()):
        if isinstance(entity, Emitter):
            sub_env.network.entities[entity_id] = replace(
                entity,
                buffer_capacity_t=entity.buffer_capacity_t * fraction,
            )
        elif isinstance(entity, Terminal):
            sub_env.network.entities[entity_id] = replace(
                entity,
                storage_capacity_t=entity.storage_capacity_t * fraction,
            )
        elif isinstance(entity, Pipeline):
            sub_env.network.entities[entity_id] = replace(
                entity,
                max_flow_tph=entity.max_flow_tph * fraction,
            )
        elif isinstance(entity, SubseaManifold):
            sub_env.network.entities[entity_id] = replace(
                entity,
                max_flow_tph=entity.max_flow_tph * fraction,
            )
        elif isinstance(entity, InjectionWell):
            sub_env.network.entities[entity_id] = replace(
                entity,
                max_injection_tph=entity.max_injection_tph * fraction,
                min_stable_injection_tph=(
                    entity.min_stable_injection_tph * fraction
                ),
            )

    shared_inventory_ids = {
        *sub_env.emitter_ids,
        *sub_env.terminal_ids,
    }
    for entity_id in shared_inventory_ids:
        current = float(
            sub_env.simulator.state.entity_inventory_t.get(entity_id, 0.0)
        )
        sub_env.simulator.state.entity_inventory_t[entity_id] = (
            current * fraction
        )
        if entity_id in sub_scenario.initial_inventory_t:
            sub_scenario.initial_inventory_t[entity_id] *= fraction
    sub_scenario.emitter_availability = {
        emitter_id: [fraction * float(value) for value in values]
        for emitter_id, values in sub_scenario.emitter_availability.items()
    }
    sub_env.scenario = sub_scenario
    return sub_env, sub_scenario


def _service_capacities(
    env,
    scenario: Scenario,
    horizon_h: int,
) -> dict[tuple[str, int], float]:
    start_h = float(env.simulator.state.time_h)
    start_step = scenario.step_index(start_h)
    capacities = {
        (emitter_id, hour): 1.0
        for emitter_id in env.emitter_ids
        for hour in range(horizon_h)
    }
    for hour in range(horizon_h):
        for terminal_id, berth_count in _terminal_berth_counts(
            env,
            scenario,
            start_step + hour,
        ).items():
            capacities[(terminal_id, hour)] = float(min(1, berth_count))
    return capacities


def _service_residuals(
    capacities: dict[tuple[str, int], float],
    results,
) -> dict[tuple[str, int], float]:
    results = tuple(results)
    return {
        key: sum(
            result.service_active_by_node_hour.get(
                key[0], ()
            )[key[1]]
            if key[1]
            < len(result.service_active_by_node_hour.get(key[0], ()))
            else 0.0
            for result in results
        )
        - capacity
        for key, capacity in capacities.items()
    }
