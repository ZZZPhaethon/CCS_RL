"""Rolling-horizon MILP controller over the native CCS action space.

The rolling controller is an MPC baseline: every few simulated hours it solves a
finite-horizon MILP from the current simulator state, executes the first slice of
that plan, and then re-plans later. Unlike the fixed-horizon oracle, this module
plans the same action objects the environment accepts:

- each berthed vessel chooses WAIT / GO_TERMINAL / GO_EMITTER[id];
- vessels already sailing are forced to WAIT until they arrive;
- injection is a continuous total rate that is mapped back to per-well Mt/y.

Loading and unloading remain automatic in the environment, so the MILP models
them as continuous flows that are only possible while a vessel waits at the
corresponding emitter or terminal.
"""

from __future__ import annotations

import copy
from collections import deque
from dataclasses import dataclass, replace
import math
from pathlib import Path
import time
from typing import Callable

from ..economics import EconomicParameters
from ..environment import (
    VESSEL_WAIT,
    CCSEnv,
)
from ..routes import route_distance_km, sea_route
from .baselines import greedy_shuttle_policy
from .milp import KNOTS_TO_KMH
from .replay import action_for_well_control_mode, replay_native_actions

@dataclass(frozen=True)
class RollingMilpPlan:
    vessel_actions_by_hour: dict[str, list[int]]
    injection_tph: list[float]
    native_actions_by_hour: list[dict[str, list[int]]]
    vented_t: float
    shortfall_t: float
    total_cost: float
    replay_vented_t: float
    replay_stored_t: float
    replay_total_cost: float
    replay_is_valid: bool
    replay_validation_error: str
    status: str
    is_valid: bool
    validation_error: str = ""
    max_binary_integrality_violation: float = 0.0
    replay_is_exact: bool = False
    replay_mismatches: tuple[str, ...] = ()
    replay_compared_fields: frozenset[str] = frozenset()
    solver_is_valid: bool = False
    terminal_cleanup_value_enabled: bool = False
    terminal_cleanup_cost: float = 0.0
    terminal_cleanup_headroom_risk: float = 0.0
    augmented_objective_value: float = 0.0
    solve_wall_s: float = 0.0
    best_bound: float | None = None
    relative_gap: float | None = None
    warm_start_accepted: bool | None = None
    warm_start_source: str = "native_mpc"
    warm_start_score: tuple[float, ...] | None = None
    mpc_warm_start_score: tuple[float, ...] | None = None
    shifted_warm_start_score: tuple[float, ...] | None = None
    cplex_root_algorithm: str = "automatic"
    termination_reason: str = ""
    requested_mip_gap_rel: float | None = None


def _plan_native_cplex_actions(
    env: CCSEnv,
    planning_horizon_h: int,
    economics: EconomicParameters,
    time_limit_s: float = 60.0,
    mip_gap_rel: float | None = None,
    objective_mode: str = "lexicographic",
    execution_h: int = 24,
    terminal_cleanup_value: bool = False,
    terminal_cleanup_mip_start_mode: str = "partial",
    load_min_formulation: str = "choice3",
    shifted_milp_warm_start: bool = False,
    previous_plan_actions: list[dict[str, list[int]]] | None = None,
    previous_plan_elapsed_h: int = 0,
    vessel_visit_load_cuts: bool = False,
    vessel_visit_load_cut_stride_h: int = 24,
    source_visit_vent_cuts: bool = False,
    source_visit_vent_cut_stride_h: int = 24,
    terminal_visit_cuts: bool = False,
    terminal_visit_cut_stride_h: int = 24,
    service_reachability_cuts: bool = False,
    service_reachability_cut_stride_h: int = 12,
    route_cargo_flow_linking: bool = False,
    cleanup_unary_trip_slots: bool = False,
    cleanup_aggregate_full_trip_dominance: bool = False,
    cleanup_return_partition_cut: bool = False,
    cleanup_source_mode_partition_cut: bool = False,
    weather_aware_cleanup_sailing_lower_bound: bool = False,
    cleanup_source_headroom_risk: bool = False,
    prune_unreachable_route_arcs: bool = False,
    warm_start_end_unstored_guard: bool = False,
    initial_barrier_root: bool = True,
    warm_start_mode: str = "native_mpc",
) -> RollingMilpPlan:
    """Plan a rolling window with the environment-aligned native CPLEX MILP."""
    from .cplex_milp import replay_full_scenario_cplex_plan

    warm_start_mode = str(warm_start_mode).lower()
    if warm_start_mode == "greedy":
        warm_start_actions = greedy_warm_start_actions(
            env,
            planning_horizon_h,
        )
        safe_progress_limit_t = None
        safe_vent_limit_t = None
        safe_end_unstored_limit_t = None
        safe_execution_vent_limit_t = None
        safe_execution_unstored_limit_t = None
    elif warm_start_mode == "native_mpc":
        (
            warm_start_actions,
            safe_progress_limit_t,
            safe_vent_limit_t,
            safe_end_unstored_limit_t,
            safe_execution_vent_limit_t,
            safe_execution_unstored_limit_t,
        ) = _native_mpc_plan_seed(
            env,
            planning_horizon_h,
            objective_mode=objective_mode,
            execution_h=execution_h,
        )
    else:
        raise ValueError(
            "warm_start_mode must be 'greedy' or 'native_mpc'"
        )
    warm_start_source = warm_start_mode
    mpc_score = _native_warm_start_score(
        env,
        warm_start_actions,
        planning_horizon_h,
        objective_mode,
        economics=economics,
        terminal_cleanup_value=terminal_cleanup_value,
        weather_aware_cleanup_sailing_lower_bound=(
            weather_aware_cleanup_sailing_lower_bound
        ),
        cleanup_source_headroom_risk=cleanup_source_headroom_risk,
    )
    shifted_score = None
    if shifted_milp_warm_start and previous_plan_actions:
        shifted_actions = _shifted_milp_warm_start(
            env,
            previous_plan_actions,
            elapsed_h=previous_plan_elapsed_h,
            mpc_actions=warm_start_actions,
            horizon_h=planning_horizon_h,
        )
        if shifted_actions is not None:
            shifted_score = _native_warm_start_score(
                env,
                shifted_actions,
                planning_horizon_h,
                objective_mode,
                economics=economics,
                terminal_cleanup_value=terminal_cleanup_value,
                weather_aware_cleanup_sailing_lower_bound=(
                    weather_aware_cleanup_sailing_lower_bound
                ),
                cleanup_source_headroom_risk=cleanup_source_headroom_risk,
            )
            if shifted_score is not None and (
                mpc_score is None or shifted_score < mpc_score
            ):
                warm_start_actions = shifted_actions
                warm_start_source = "shifted_milp"
    if warm_start_end_unstored_guard:
        selected_score = (
            shifted_score
            if warm_start_source == "shifted_milp"
            else mpc_score
        )
        if selected_score is not None:
            safe_end_unstored_limit_t = float(selected_score[1])
    cplex_root_algorithm, extra_cplex_options = _initial_root_cplex_options(
        env,
        planning_horizon_h,
        enabled=initial_barrier_root,
    )
    result = _solve_native_cplex_result(
        env,
        planning_horizon_h,
        economics,
        warm_start_actions,
        time_limit_s,
        mip_gap_rel=mip_gap_rel,
        objective_mode=objective_mode,
        safe_progress_limit_t=safe_progress_limit_t,
        safe_vent_limit_t=safe_vent_limit_t,
        safe_end_unstored_limit_t=safe_end_unstored_limit_t,
        execution_h=execution_h,
        safe_execution_vent_limit_t=safe_execution_vent_limit_t,
        safe_execution_unstored_limit_t=safe_execution_unstored_limit_t,
        terminal_cleanup_value=terminal_cleanup_value,
        terminal_cleanup_mip_start_mode=terminal_cleanup_mip_start_mode,
        load_min_formulation=load_min_formulation,
        vessel_visit_load_cuts=vessel_visit_load_cuts,
        vessel_visit_load_cut_stride_h=vessel_visit_load_cut_stride_h,
        source_visit_vent_cuts=source_visit_vent_cuts,
        source_visit_vent_cut_stride_h=source_visit_vent_cut_stride_h,
        terminal_visit_cuts=terminal_visit_cuts,
        terminal_visit_cut_stride_h=terminal_visit_cut_stride_h,
        service_reachability_cuts=service_reachability_cuts,
        service_reachability_cut_stride_h=service_reachability_cut_stride_h,
        route_cargo_flow_linking=route_cargo_flow_linking,
        cleanup_unary_trip_slots=cleanup_unary_trip_slots,
        cleanup_aggregate_full_trip_dominance=(
            cleanup_aggregate_full_trip_dominance
        ),
        cleanup_return_partition_cut=cleanup_return_partition_cut,
        cleanup_source_mode_partition_cut=(
            cleanup_source_mode_partition_cut
        ),
        weather_aware_cleanup_sailing_lower_bound=(
            weather_aware_cleanup_sailing_lower_bound
        ),
        cleanup_source_headroom_risk=cleanup_source_headroom_risk,
        prune_unreachable_route_arcs=prune_unreachable_route_arcs,
        extra_cplex_options=extra_cplex_options,
    )
    native_actions = _materialize_cplex_actions(env, result.native_actions_by_hour)
    replay_result = replace(result, native_actions_by_hour=native_actions)
    replay = replay_full_scenario_cplex_plan(copy.deepcopy(env), replay_result)
    replay_error = "" if replay.is_exact else ";".join(
        dict.fromkeys((*replay.violations, *replay.mismatches))
    )
    final_stage = result.stage_diagnostics[-1] if result.stage_diagnostics else None
    validation_error = ";".join(
        error for error in (result.validation_error, replay_error) if error
    )
    return RollingMilpPlan(
        vessel_actions_by_hour=result.vessel_actions_by_hour,
        injection_tph=result.injection_tph,
        native_actions_by_hour=native_actions,
        vented_t=result.vented_t,
        shortfall_t=result.shortfall_t,
        total_cost=result.total_cost,
        replay_vented_t=replay.vented_t,
        replay_stored_t=replay.stored_t,
        replay_total_cost=replay.total_cost,
        replay_is_valid=replay.is_executable,
        replay_validation_error=replay_error,
        status=result.status,
        is_valid=result.is_valid and replay.is_exact,
        validation_error=validation_error,
        max_binary_integrality_violation=result.max_binary_integrality_violation,
        replay_is_exact=replay.is_exact,
        replay_mismatches=replay.mismatches,
        replay_compared_fields=replay.compared_fields,
        solver_is_valid=result.is_valid,
        terminal_cleanup_value_enabled=result.terminal_cleanup_value_enabled,
        terminal_cleanup_cost=result.terminal_cleanup_cost,
        terminal_cleanup_headroom_risk=(
            result.terminal_cleanup_headroom_risk
        ),
        augmented_objective_value=result.augmented_objective_value,
        solve_wall_s=final_stage.wall_time_s if final_stage is not None else 0.0,
        best_bound=final_stage.best_bound if final_stage is not None else None,
        relative_gap=final_stage.relative_gap if final_stage is not None else None,
        warm_start_accepted=(
            final_stage.warm_start_accepted if final_stage is not None else None
        ),
        warm_start_source=warm_start_source,
        warm_start_score=mpc_score,
        mpc_warm_start_score=(
            mpc_score if warm_start_mode == "native_mpc" else None
        ),
        shifted_warm_start_score=shifted_score,
        cplex_root_algorithm=cplex_root_algorithm,
        termination_reason=(
            final_stage.termination_reason if final_stage is not None else ""
        ),
        requested_mip_gap_rel=mip_gap_rel,
    )


def _initial_root_cplex_options(
    env: CCSEnv,
    planning_horizon_h: int,
    *,
    enabled: bool,
) -> tuple[str, list[str]]:
    if enabled and env.t == 0 and int(planning_horizon_h) >= 168:
        return "barrier", ["set mip strategy startalgorithm 4"]
    return "automatic", []


def _shifted_milp_warm_start(
    env: CCSEnv,
    previous_plan_actions: list[dict[str, list[int]]],
    *,
    elapsed_h: int,
    mpc_actions: list[dict[str, list[int]]],
    horizon_h: int,
) -> list[dict[str, list[int]]] | None:
    elapsed_h = max(0, int(elapsed_h))
    horizon_h = max(0, int(horizon_h))
    if horizon_h == 0 or elapsed_h >= len(previous_plan_actions):
        return None
    shifted = copy.deepcopy(previous_plan_actions[elapsed_h : elapsed_h + horizon_h])
    shifted.extend(copy.deepcopy(mpc_actions[len(shifted) : horizon_h]))
    if len(shifted) != horizon_h:
        return None
    try:
        materialized = _materialize_cplex_actions(env, shifted)
        replay = replay_native_actions(
            env,
            [
                action_for_well_control_mode(env, action)
                for action in materialized
            ],
            horizon_h=horizon_h,
        )
    except (RuntimeError, ValueError, IndexError, KeyError):
        return None
    return materialized if replay.is_executable else None


def _native_warm_start_score(
    env: CCSEnv,
    actions: list[dict[str, list[int]]],
    horizon_h: int,
    objective_mode: str,
    *,
    economics: EconomicParameters,
    terminal_cleanup_value: bool,
    weather_aware_cleanup_sailing_lower_bound: bool = False,
    cleanup_source_headroom_risk: bool = False,
) -> tuple[float, ...] | None:
    replay_env = copy.deepcopy(env)
    try:
        replay = replay_native_actions(
            replay_env,
            [
                action_for_well_control_mode(replay_env, action)
                for action in actions
            ],
            horizon_h=horizon_h,
            copy_env=False,
        )
    except (RuntimeError, ValueError, IndexError, KeyError):
        return None
    if not replay.is_executable:
        return None
    actual = replay.actual
    total_cost = float(actual.total_cost)
    if terminal_cleanup_value:
        try:
            from .cplex_milp import _terminal_cleanup_cost_for_state

            total_cost += _terminal_cleanup_cost_for_state(
                replay_env,
                economics,
                weather_aware_sailing_lower_bound=(
                    weather_aware_cleanup_sailing_lower_bound
                ),
                source_headroom_risk=cleanup_source_headroom_risk,
            )
        except RuntimeError:
            return None
    if str(objective_mode).lower() == "lexicographic":
        return (
            float(actual.vented_t),
            float(actual.in_transit_t),
            total_cost,
        )
    return (total_cost, float(actual.in_transit_t))


def _solve_native_cplex_result(
    env: CCSEnv,
    planning_horizon_h: int,
    economics: EconomicParameters,
    warm_start_actions: list[dict[str, list[int]]] | None,
    time_limit_s: float,
    *,
    mip_gap_rel: float | None = None,
    objective_mode: str = "lexicographic",
    safe_progress_limit_t: float | None = None,
    safe_vent_limit_t: float | None = None,
    safe_end_unstored_limit_t: float | None = None,
    execution_h: int = 24,
    safe_execution_vent_limit_t: float | None = None,
    safe_execution_unstored_limit_t: float | None = None,
    terminal_cleanup_value: bool = False,
    terminal_cleanup_mip_start_mode: str = "partial",
    load_min_formulation: str = "choice3",
    fifo_diagnostic_mode: str = "full",
    vessel_visit_load_cuts: bool = False,
    vessel_visit_load_cut_stride_h: int = 24,
    source_visit_vent_cuts: bool = False,
    source_visit_vent_cut_stride_h: int = 24,
    terminal_visit_cuts: bool = False,
    terminal_visit_cut_stride_h: int = 24,
    service_reachability_cuts: bool = False,
    service_reachability_cut_stride_h: int = 12,
    route_cargo_flow_linking: bool = False,
    cleanup_unary_trip_slots: bool = False,
    cleanup_aggregate_full_trip_dominance: bool = False,
    cleanup_return_partition_cut: bool = False,
    cleanup_source_mode_partition_cut: bool = False,
    weather_aware_cleanup_sailing_lower_bound: bool = False,
    cleanup_source_headroom_risk: bool = False,
    prune_unreachable_route_arcs: bool = False,
    min_total_cleanup_trips: int | None = None,
    fixed_cleanup_trips_by_source: dict[str, int] | None = None,
    fixed_cleanup_trips_by_vessel_source: (
        dict[tuple[str, str], int] | None
    ) = None,
    fixed_boundary_node_by_vessel: dict[str, str] | None = None,
    fix_warm_start_vessel_routes: bool = False,
    fixed_terminal_departures_by_vessel: dict[str, int] | None = None,
    fixed_terminal_departures_by_vessel_source: (
        dict[tuple[str, str], int] | None
    ) = None,
    fixed_terminal_to_source_departures_by_vessel_source: (
        dict[tuple[str, str], int] | None
    ) = None,
    fixed_source_reposition_departures_by_vessel: dict[str, int] | None = None,
    min_total_source_reposition_departures: int | None = None,
    integrality_relax_groups: tuple[str, ...] = (),
    constraint_redundancy_audit: bool = False,
    extra_cplex_options: list[str] | None = None,
    export_model_lp_path: str | Path | None = None,
):
    from .cplex_milp import solve_full_scenario_with_cplex

    economic_objective = objective_mode in {
        "economic",
        "economic_safe",
        "economic_safe_strict",
        "economic_lex_guard",
        "economic_execution_guard",
    }
    cplex_options = ["set simplex tolerances feasibility 1e-7"]
    if objective_mode == "lexicographic":
        cplex_options.extend(
            [
                "set mip limits cutpasses 1",
                "set mip strategy heuristicfreq 10",
                "set mip strategy search 1",
            ]
        )
    cplex_options.extend(extra_cplex_options or [])
    solve_kwargs = {
        "horizon_h": planning_horizon_h,
        "economics": economics,
        "warm_start_native_actions_by_hour": warm_start_actions,
        "time_limit_s": time_limit_s,
        "mip_gap_rel": mip_gap_rel,
        "cplex_options": cplex_options,
        "lexicographic_vent_first": objective_mode == "lexicographic",
        "economic_objective": economic_objective,
        "max_nonstored_t": safe_progress_limit_t if objective_mode == "economic_safe" else None,
        "max_vented_t": safe_vent_limit_t,
        "max_end_unstored_t": safe_end_unstored_limit_t,
        "execution_boundary_h": execution_h,
        "max_execution_vented_t": safe_execution_vent_limit_t,
        "max_execution_unstored_t": safe_execution_unstored_limit_t,
        "terminal_cleanup_value": terminal_cleanup_value,
        "terminal_cleanup_mip_start_mode": terminal_cleanup_mip_start_mode,
        "load_min_formulation": load_min_formulation,
        "fifo_diagnostic_mode": fifo_diagnostic_mode,
        "vessel_visit_load_cuts": vessel_visit_load_cuts,
        "vessel_visit_load_cut_stride_h": vessel_visit_load_cut_stride_h,
        "source_visit_vent_cuts": source_visit_vent_cuts,
        "source_visit_vent_cut_stride_h": source_visit_vent_cut_stride_h,
        "terminal_visit_cuts": terminal_visit_cuts,
        "terminal_visit_cut_stride_h": terminal_visit_cut_stride_h,
        "service_reachability_cuts": service_reachability_cuts,
        "service_reachability_cut_stride_h": service_reachability_cut_stride_h,
        "route_cargo_flow_linking": route_cargo_flow_linking,
        "cleanup_unary_trip_slots": cleanup_unary_trip_slots,
        "cleanup_aggregate_full_trip_dominance": (
            cleanup_aggregate_full_trip_dominance
        ),
        "cleanup_return_partition_cut": cleanup_return_partition_cut,
        "cleanup_source_mode_partition_cut": (
            cleanup_source_mode_partition_cut
        ),
        "weather_aware_cleanup_sailing_lower_bound": (
            weather_aware_cleanup_sailing_lower_bound
        ),
        "cleanup_source_headroom_risk": cleanup_source_headroom_risk,
        "prune_unreachable_route_arcs": prune_unreachable_route_arcs,
        "min_total_cleanup_trips": min_total_cleanup_trips,
        "fixed_cleanup_trips_by_source": fixed_cleanup_trips_by_source,
        "fixed_cleanup_trips_by_vessel_source": (
            fixed_cleanup_trips_by_vessel_source
        ),
        "fixed_boundary_node_by_vessel": fixed_boundary_node_by_vessel,
        "fix_warm_start_vessel_routes": fix_warm_start_vessel_routes,
        "fixed_terminal_departures_by_vessel": (
            fixed_terminal_departures_by_vessel
        ),
        "fixed_terminal_departures_by_vessel_source": (
            fixed_terminal_departures_by_vessel_source
        ),
        "fixed_terminal_to_source_departures_by_vessel_source": (
            fixed_terminal_to_source_departures_by_vessel_source
        ),
        "fixed_source_reposition_departures_by_vessel": (
            fixed_source_reposition_departures_by_vessel
        ),
        "min_total_source_reposition_departures": (
            min_total_source_reposition_departures
        ),
        "integrality_relax_groups": integrality_relax_groups,
        "constraint_redundancy_audit": constraint_redundancy_audit,
        "export_model_lp_path": export_model_lp_path,
    }
    result = solve_full_scenario_with_cplex(
        env,
        **solve_kwargs,
        environment_aligned_service=True,
    )
    return result


def _materialize_cplex_actions(
    env: CCSEnv,
    planned_actions: list[dict[str, list[int]]],
) -> list[dict[str, list[int]]]:
    """Delay planned departures only until the native environment permits them."""

    replay_env = copy.deepcopy(env)
    pending = {vessel_id: deque() for vessel_id in replay_env.vessel_ids}
    actions: list[dict[str, list[int]]] = []
    for planned in planned_actions:
        for vessel_id, choice in zip(replay_env.vessel_ids, planned["vessels"]):
            choice = int(choice)
            if choice != VESSEL_WAIT:
                pending[vessel_id].append(choice)

        vessel_masks = replay_env.vessel_action_mask()
        vessel_actions: list[int] = []
        for vessel_id, mask in zip(replay_env.vessel_ids, vessel_masks):
            queue = pending[vessel_id]
            vessel_state = replay_env.simulator.vessel_states[vessel_id]
            while queue and vessel_state["mode"] == "berthed":
                destination = replay_env._vessel_action_destination(vessel_id, queue[0])
                if destination != vessel_state["berth"]:
                    break
                queue.popleft()
            choice = queue[0] if queue and mask[queue[0]] else VESSEL_WAIT
            if choice != VESSEL_WAIT:
                queue.popleft()
            vessel_actions.append(int(choice))

        action = {"vessels": vessel_actions}
        if not replay_env.automatic_well_control:
            well_actions: list[int] = []
            for well_id, choice, mask in zip(
                replay_env.well_ids,
                planned["wells"],
                replay_env.well_rate_action_mask(),
            ):
                choice = int(choice)
                well_actions.append(
                    choice if 0 <= choice < len(mask) and mask[choice]
                    else replay_env.highest_feasible_well_rate_index(well_id)
                )
            action["wells"] = well_actions
        actions.append(action)
        replay_env.step(action)
    return actions


def _native_mpc_plan_seed(
    env: CCSEnv,
    horizon_h: int,
    *,
    objective_mode: str = "lexicographic",
    execution_h: int | None = None,
) -> tuple[
    list[dict[str, list[int]]],
    float | None,
    float | None,
    float | None,
    float | None,
    float | None,
]:
    from .native_mpc import RollingNativeMpcController

    replay_env = copy.deepcopy(env)
    controller = RollingNativeMpcController(
        replay_env,
        replan_every=execution_h if execution_h is not None else horizon_h,
        planning_horizon_h=horizon_h,
        objective_mode=objective_mode,
    )
    controller.policy(replay_env)
    return (
        copy.deepcopy(controller._native_actions_by_hour),
        controller.last_safe_progress_limit_t,
        controller.last_safe_vent_limit_t,
        controller.last_safe_end_unstored_limit_t,
        controller.last_safe_execution_vent_limit_t,
        controller.last_safe_execution_unstored_limit_t,
    )


def _native_mpc_warm_start(
    env: CCSEnv,
    horizon_h: int,
    *,
    objective_mode: str = "lexicographic",
) -> list[dict[str, list[int]]]:
    (
        actions,
        _safe_progress_limit_t,
        _safe_vent_limit_t,
        _safe_end_unstored_limit_t,
        _safe_execution_vent_limit_t,
        _safe_execution_unstored_limit_t,
    ) = _native_mpc_plan_seed(
        env,
        horizon_h,
        objective_mode=objective_mode,
    )
    return actions


def greedy_warm_start_actions(
    env: CCSEnv,
    horizon_h: int,
) -> list[dict[str, list[int]]]:
    """Build a replay-valid Greedy trajectory from the current state."""

    if horizon_h <= 0:
        raise ValueError("horizon_h must be positive")
    replay_env = copy.deepcopy(env)
    actions: list[dict[str, list[int]]] = []
    for _step in range(int(horizon_h)):
        action = action_for_well_control_mode(
            replay_env,
            greedy_shuttle_policy(replay_env),
        )
        actions.append(copy.deepcopy(action))
        _obs, _reward, terminated, truncated, _info = replay_env.step(action)
        if terminated or truncated:
            break
    if len(actions) != int(horizon_h):
        raise RuntimeError(
            "Greedy warm start ended before the requested planning horizon"
        )
    validation = replay_native_actions(
        env,
        actions,
        horizon_h=int(horizon_h),
    )
    if not validation.is_executable:
        reason = ";".join(
            (*validation.violations, *validation.mismatches)
        )
        raise RuntimeError(
            f"Greedy warm start is not replay-valid: {reason}"
        )
    return actions


class RollingMilpController:
    """Re-planning MILP controller, usable as a metrics ``policy(env)``."""

    def __init__(
        self,
        env: CCSEnv,
        replan_every: int = 24,
        economics: EconomicParameters | None = None,
        progress: Callable[[str], None] | None = None,
        planning_horizon_h: int = 168,
        time_limit_s: float = 30.0,
        mip_gap_rel: float | None = None,
        objective_mode: str = "lexicographic",
        terminal_cleanup_value: bool = True,
        terminal_cleanup_mip_start_mode: str = "partial",
        load_min_formulation: str = "choice3",
        shifted_milp_warm_start: bool = False,
        warm_start_mode: str = "greedy",
        vessel_visit_load_cuts: bool = True,
        vessel_visit_load_cut_stride_h: int = 12,
        source_visit_vent_cuts: bool = True,
        source_visit_vent_cut_stride_h: int = 12,
        terminal_visit_cuts: bool = True,
        terminal_visit_cut_stride_h: int = 12,
        service_reachability_cuts: bool = True,
        service_reachability_cut_stride_h: int = 12,
        route_cargo_flow_linking: bool = True,
        cleanup_unary_trip_slots: bool = True,
        cleanup_aggregate_full_trip_dominance: bool = False,
        cleanup_return_partition_cut: bool = False,
        cleanup_source_mode_partition_cut: bool = False,
        weather_aware_cleanup_sailing_lower_bound: bool = False,
        cleanup_source_headroom_risk: bool = False,
        prune_unreachable_route_arcs: bool = False,
        warm_start_end_unstored_guard: bool = False,
        initial_barrier_root: bool = True,
    ):
        self.replan_every = max(1, int(replan_every))
        self.economics = economics or EconomicParameters()
        self.progress = progress
        self.planning_horizon_h = max(1, int(planning_horizon_h))
        self.time_limit_s = float(time_limit_s)
        self.mip_gap_rel = (
            None if mip_gap_rel is None else float(mip_gap_rel)
        )
        if self.mip_gap_rel is not None and self.mip_gap_rel < 0.0:
            raise ValueError("mip_gap_rel must be non-negative")
        self.objective_mode = str(objective_mode).lower()
        self.terminal_cleanup_value = bool(terminal_cleanup_value)
        self.terminal_cleanup_mip_start_mode = str(
            terminal_cleanup_mip_start_mode
        ).lower()
        if self.terminal_cleanup_mip_start_mode not in {"partial", "complete"}:
            raise ValueError(
                "terminal_cleanup_mip_start_mode must be 'partial' or 'complete'"
            )
        self.load_min_formulation = str(load_min_formulation).lower()
        self.shifted_milp_warm_start = bool(shifted_milp_warm_start)
        self.warm_start_mode = str(warm_start_mode).lower()
        if self.warm_start_mode not in {"greedy", "native_mpc"}:
            raise ValueError(
                "warm_start_mode must be 'greedy' or 'native_mpc'"
            )
        self.vessel_visit_load_cuts = bool(vessel_visit_load_cuts)
        self.vessel_visit_load_cut_stride_h = int(vessel_visit_load_cut_stride_h)
        self.source_visit_vent_cuts = bool(source_visit_vent_cuts)
        self.source_visit_vent_cut_stride_h = int(source_visit_vent_cut_stride_h)
        self.terminal_visit_cuts = bool(terminal_visit_cuts)
        self.terminal_visit_cut_stride_h = int(terminal_visit_cut_stride_h)
        self.service_reachability_cuts = bool(service_reachability_cuts)
        self.service_reachability_cut_stride_h = int(
            service_reachability_cut_stride_h
        )
        self.route_cargo_flow_linking = bool(route_cargo_flow_linking)
        self.cleanup_unary_trip_slots = bool(cleanup_unary_trip_slots)
        self.cleanup_aggregate_full_trip_dominance = bool(
            cleanup_aggregate_full_trip_dominance
        )
        self.cleanup_return_partition_cut = bool(
            cleanup_return_partition_cut
        )
        self.cleanup_source_mode_partition_cut = bool(
            cleanup_source_mode_partition_cut
        )
        self.weather_aware_cleanup_sailing_lower_bound = bool(
            weather_aware_cleanup_sailing_lower_bound
        )
        self.cleanup_source_headroom_risk = bool(
            cleanup_source_headroom_risk
        )
        self.prune_unreachable_route_arcs = bool(
            prune_unreachable_route_arcs
        )
        self.warm_start_end_unstored_guard = bool(
            warm_start_end_unstored_guard
        )
        self.initial_barrier_root = bool(initial_barrier_root)
        if self.load_min_formulation not in {"choice3", "factored"}:
            raise ValueError(
                "load_min_formulation must be either 'choice3' or 'factored'"
            )
        if self.objective_mode not in {
            "lexicographic",
            "economic",
            "economic_safe",
            "economic_safe_strict",
            "economic_lex_guard",
            "economic_execution_guard",
        }:
            raise ValueError(f"Unknown rolling MILP objective mode: {objective_mode}")
        self._native_actions_by_hour: list[dict[str, list[int]]] = []
        self._plan_origin_h: float = -1e9
        self._has_active_plan = False
        self.last_plan_status = ""
        self.last_plan_valid = False
        self.last_validation_error = ""
        self.last_model_replay_is_exact = False
        self.last_model_replay_mismatches: tuple[str, ...] = ()
        self.last_execution_replay_is_valid = False
        self.last_execution_replay_mismatches: tuple[str, ...] = ()
        self.last_warm_start_source = ""
        self.replan_count = 0
        self.status_counts: dict[str, int] = {}
        self.model_inexact_replan_count = 0
        self.replan_diagnostics: list[dict[str, object]] = []

    def __call__(self, env: CCSEnv) -> dict[str, list]:
        return self.policy(env)

    def policy(self, env: CCSEnv) -> dict[str, list]:
        now = env.simulator.state.time_h
        new_episode = now < self._plan_origin_h
        if new_episode or now - self._plan_origin_h >= self.replan_every or not self._has_active_plan:
            self._replan(env, now)

        elapsed = int(max(0.0, math.floor(now - self._plan_origin_h)))
        if elapsed >= len(self._native_actions_by_hour):
            raise RuntimeError(f"rolling_milp native trace expired at hour {elapsed}")
        action = self._native_actions_by_hour[elapsed]
        self._validate_native_action(env, action)
        control_action = {
            "vessels": [int(choice) for choice in action["vessels"]],
        }
        if not env.automatic_well_control:
            control_action["wells"] = [
                int(choice) for choice in action["wells"]
            ]
        return control_action

    def _replan(self, env: CCSEnv, now: float) -> None:
        state = env.simulator.state
        term_init = sum(state.entity_inventory_t.get(t, 0.0) for t in env.terminal_ids)
        source_buffer = sum(state.entity_inventory_t.get(e, 0.0) for e in env.emitter_ids)
        start = time.perf_counter()
        remaining_h = max(1, min(self.planning_horizon_h, env.n_steps - env.t))
        previous_plan_actions = None
        previous_plan_elapsed_h = 0
        if (
            self.shifted_milp_warm_start
            and self._has_active_plan
            and self._native_actions_by_hour
            and now >= self._plan_origin_h
        ):
            previous_plan_elapsed_h = int(
                max(0.0, math.floor(now - self._plan_origin_h))
            )
            if previous_plan_elapsed_h > 0:
                previous_plan_actions = copy.deepcopy(self._native_actions_by_hour)
        if self.progress is not None:
            self.progress(
                f"  rolling_milp replan at t={now:.0f} h; "
                f"lookahead={remaining_h} h; "
                f"terminal={term_init:,.1f} t; source_buffer={source_buffer:,.1f} t"
            )
        plan = _plan_native_cplex_actions(
            env,
            remaining_h,
            self.economics,
            time_limit_s=self.time_limit_s,
            mip_gap_rel=self.mip_gap_rel,
            objective_mode=self.objective_mode,
            execution_h=self.replan_every,
            terminal_cleanup_value=self.terminal_cleanup_value,
            terminal_cleanup_mip_start_mode=(
                self.terminal_cleanup_mip_start_mode
            ),
            load_min_formulation=self.load_min_formulation,
            shifted_milp_warm_start=self.shifted_milp_warm_start,
            previous_plan_actions=previous_plan_actions,
            previous_plan_elapsed_h=previous_plan_elapsed_h,
            vessel_visit_load_cuts=self.vessel_visit_load_cuts,
            vessel_visit_load_cut_stride_h=self.vessel_visit_load_cut_stride_h,
            source_visit_vent_cuts=self.source_visit_vent_cuts,
            source_visit_vent_cut_stride_h=self.source_visit_vent_cut_stride_h,
            terminal_visit_cuts=self.terminal_visit_cuts,
            terminal_visit_cut_stride_h=self.terminal_visit_cut_stride_h,
            service_reachability_cuts=self.service_reachability_cuts,
            service_reachability_cut_stride_h=(
                self.service_reachability_cut_stride_h
            ),
            route_cargo_flow_linking=self.route_cargo_flow_linking,
            cleanup_unary_trip_slots=self.cleanup_unary_trip_slots,
            cleanup_aggregate_full_trip_dominance=(
                self.cleanup_aggregate_full_trip_dominance
            ),
            cleanup_return_partition_cut=(
                self.cleanup_return_partition_cut
            ),
            cleanup_source_mode_partition_cut=(
                self.cleanup_source_mode_partition_cut
            ),
            weather_aware_cleanup_sailing_lower_bound=(
                self.weather_aware_cleanup_sailing_lower_bound
            ),
            cleanup_source_headroom_risk=(
                self.cleanup_source_headroom_risk
            ),
            prune_unreachable_route_arcs=(
                self.prune_unreachable_route_arcs
            ),
            warm_start_end_unstored_guard=(
                self.warm_start_end_unstored_guard
            ),
            initial_barrier_root=self.initial_barrier_root,
            warm_start_mode=self.warm_start_mode,
        )
        self.last_plan_status = plan.status
        self.status_counts[plan.status] = self.status_counts.get(plan.status, 0) + 1
        solver_is_valid = bool(getattr(plan, "solver_is_valid", plan.is_valid))
        native_actions = list(getattr(plan, "native_actions_by_hour", []))
        if native_actions:
            execution_h = min(self.replan_every, remaining_h, len(native_actions))
            replay_actions = [
                action_for_well_control_mode(env, action)
                for action in native_actions[:execution_h]
            ]
            execution_replay = replay_native_actions(
                env,
                replay_actions,
                horizon_h=execution_h,
            )
            replay_is_valid = execution_replay.is_executable
            execution_mismatches = execution_replay.mismatches
        else:
            replay_is_valid = bool(getattr(plan, "replay_is_valid", plan.is_valid))
            execution_mismatches = tuple(getattr(plan, "replay_mismatches", ()))
        # Model-to-environment metric mismatches remain diagnostics; execution
        # is gated by solver validity and feasibility of only the control slice
        # that will run before the next replan.
        execution_ready = solver_is_valid and replay_is_valid
        self.last_plan_valid = execution_ready
        self.last_validation_error = "" if execution_ready else plan.validation_error
        self.last_model_replay_is_exact = bool(getattr(plan, "replay_is_exact", plan.is_valid))
        self.last_model_replay_mismatches = tuple(getattr(plan, "replay_mismatches", ()))
        self.last_execution_replay_is_valid = replay_is_valid
        self.last_execution_replay_mismatches = execution_mismatches
        self.last_warm_start_source = str(
            getattr(plan, "warm_start_source", "native_mpc")
        )
        self.replan_count += 1
        self.replan_diagnostics.append(
            {
                "state_hour": float(now),
                "planning_horizon_h": int(remaining_h),
                "status": str(plan.status),
                "solver_is_valid": solver_is_valid,
                "execution_replay_is_valid": replay_is_valid,
                "execution_replay_mismatches": ";".join(execution_mismatches),
                "model_replay_is_exact": self.last_model_replay_is_exact,
                "model_replay_mismatches": ";".join(
                    self.last_model_replay_mismatches
                ),
                "solve_wall_s": float(getattr(plan, "solve_wall_s", 0.0)),
                "replan_wall_s": float(time.perf_counter() - start),
                "best_bound": getattr(plan, "best_bound", None),
                "relative_gap": getattr(plan, "relative_gap", None),
                "termination_reason": str(
                    getattr(plan, "termination_reason", "")
                ),
                "requested_mip_gap_rel": getattr(
                    plan, "requested_mip_gap_rel", None
                ),
                "warm_start_accepted": getattr(plan, "warm_start_accepted", None),
                "warm_start_source": self.last_warm_start_source,
                "cplex_root_algorithm": getattr(
                    plan, "cplex_root_algorithm", "automatic"
                ),
                "warm_start_score": getattr(
                    plan, "warm_start_score", None
                ),
                "mpc_warm_start_score": getattr(
                    plan, "mpc_warm_start_score", None
                ),
                "shifted_warm_start_score": getattr(
                    plan, "shifted_warm_start_score", None
                ),
                "terminal_cleanup_cost": float(
                    getattr(plan, "terminal_cleanup_cost", 0.0)
                ),
                "terminal_cleanup_headroom_risk": float(
                    getattr(plan, "terminal_cleanup_headroom_risk", 0.0)
                ),
                "augmented_objective_value": float(
                    getattr(plan, "augmented_objective_value", 0.0)
                ),
            }
        )
        if not self.last_model_replay_is_exact:
            self.model_inexact_replan_count += 1
        self._plan_origin_h = now
        self._has_active_plan = True
        if not execution_ready:
            self._has_active_plan = False
            self._native_actions_by_hour = []
            if self.progress is not None:
                self.progress(
                    f"  rolling_milp plan invalid in {time.perf_counter() - start:.1f}s; "
                    f"status={plan.status}; reason={plan.validation_error}"
                )
            raise RuntimeError(plan.validation_error or f"rolling_milp solver status {plan.status}")

        self._native_actions_by_hour = native_actions
        planned_departures = sum(
            1
            for action in native_actions
            for choice in action["vessels"]
            if choice != VESSEL_WAIT
        )
        if self.progress is not None:
            self.progress(
                f"  rolling_milp plan ready in {time.perf_counter() - start:.1f}s; "
                f"planned_departures={planned_departures}; "
                f"vented={plan.vented_t:,.1f} t; shortfall={plan.shortfall_t:,.1f} t; "
                f"model_replay_exact={self.last_model_replay_is_exact}"
            )

    @staticmethod
    def _validate_native_action(env: CCSEnv, action: dict[str, list[int]]) -> None:
        vessel_actions = action.get("vessels", [])
        well_actions = action.get("wells", [])
        if len(vessel_actions) != len(env.vessel_ids):
            raise RuntimeError("rolling_milp native trace has the wrong action dimension")
        if (
            not env.automatic_well_control
            and len(well_actions) != len(env.well_ids)
        ):
            raise RuntimeError("rolling_milp native trace has the wrong action dimension")
        for vessel_id, choice, mask in zip(env.vessel_ids, vessel_actions, env.vessel_action_mask()):
            if not (0 <= int(choice) < len(mask) and mask[int(choice)]):
                raise RuntimeError(f"rolling_milp action is infeasible for {vessel_id}: {choice}")
        if not env.automatic_well_control:
            for well_id, choice, mask in zip(
                env.well_ids,
                well_actions,
                env.well_rate_action_mask(),
            ):
                if not (0 <= int(choice) < len(mask) and mask[int(choice)]):
                    raise RuntimeError(
                        f"rolling_milp action is infeasible for {well_id}: {choice}"
                    )

def _sail_hours_between(
    env: CCSEnv,
    origin_id: str,
    destination_id: str,
    vessel_id: str,
    *,
    start_h: int = 0,
    max_horizon_h: int | None = None,
) -> int:
    route = env._routes[vessel_id]
    if {origin_id, destination_id} == {str(route["origin"]), str(route["destination"])}:
        distance_km = float(route["distance_km"])
    else:
        distance_km = _dynamic_leg_distance_km(env, route, origin_id, destination_id)
    return _sailing_duration_h(
        env,
        vessel_id,
        distance_km=distance_km,
        start_h=start_h,
        max_horizon_h=max_horizon_h,
    )


def _sailing_duration_h(
    env: CCSEnv,
    vessel_id: str,
    *,
    distance_km: float,
    start_h: int,
    max_horizon_h: int | None,
) -> int:
    route = env._routes[vessel_id]
    speed_kmh = max(0.0, float(route["speed_knots"])) * KNOTS_TO_KMH
    if distance_km <= 1e-9:
        return 0
    if speed_kmh <= 1e-9:
        return 1 if max_horizon_h is None else max_horizon_h + 1

    nominal_h = max(1, math.ceil(distance_km / speed_kmh))
    max_search_h = max(1, int(max_horizon_h)) if max_horizon_h is not None else nominal_h * 10 + 24
    covered_km = 0.0
    for elapsed_h in range(1, max_search_h + 1):
        offset_h = start_h + elapsed_h - 1
        speed_factor = _forecast_vessel_speed_factor(env, vessel_id, offset_h)
        covered_km += speed_kmh * speed_factor
        if covered_km >= distance_km - 1e-9:
            return elapsed_h
    return max_search_h + 1


def _dynamic_leg_distance_km(env: CCSEnv, route: dict, origin_id: str, destination_id: str) -> float:
    leg_routes = route.setdefault("dynamic_leg_routes", {})
    leg_id = f"{origin_id}->{destination_id}"
    if leg_id not in leg_routes:
        origin = env.locations[origin_id]
        destination = env.locations[destination_id]
        maritime_route = sea_route(origin, destination)
        coordinates = list(maritime_route.coordinates)
        if not coordinates:
            coordinates = [origin, destination]
        else:
            if coordinates[0] != origin:
                coordinates.insert(0, origin)
            if coordinates[-1] != destination:
                coordinates.append(destination)
        leg_routes[leg_id] = {
            "id": leg_id,
            "origin": origin_id,
            "destination": destination_id,
            "provider": maritime_route.provider,
            "distance_km": round(route_distance_km(coordinates), 2),
            "coordinates": coordinates,
        }
    return float(leg_routes[leg_id]["distance_km"])


def _capture_tonnes(env: CCSEnv, emitter_id: str, offset_h: int) -> float:
    state = env.simulator.state
    emitter = env.network.entities[emitter_id]
    availability = _forecast_series_value(
        env,
        env.scenario.emitter_availability if env.scenario is not None else {},
        emitter_id,
        offset_h,
        state.emitter_availability.get(emitter_id, emitter.availability),
    )
    return emitter.capture_rate_tph_at(state.time_h + offset_h) * max(0.0, float(availability))


def _forecast_vessel_speed_factor(env: CCSEnv, vessel_id: str, offset_h: int) -> float:
    state = env.simulator.state
    value = _forecast_series_value(
        env,
        env.scenario.vessel_speed_factor if env.scenario is not None else {},
        vessel_id,
        offset_h,
        state.vessel_speed_factor.get(vessel_id, 1.0),
    )
    return max(0.0, float(value))


def _forecast_series_value(
    env: CCSEnv,
    series_by_id: dict[str, list],
    entity_id: str,
    offset_h: int,
    fallback,
):
    series = series_by_id.get(entity_id)
    if not series or env.scenario is None:
        return fallback
    time_h = env.simulator.state.time_h + offset_h * env.network.time_step_hours
    return series[env.scenario.step_index(time_h)]
