"""Smoke-test the paper baselines and time-limited MILP paths on one seed."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from pathlib import Path

from sim.control.baselines import (
    greedy_shuttle_policy,
    make_cluster_shuttle_policy,
)
from sim.control.cplex_milp import (
    _terminal_cleanup_cost_for_state,
    solve_full_scenario_with_cplex,
)
from sim.control.event_based.residual_rl_v4.scenario import (
    ReplayableDifficultyScenarioGenerator,
)
from sim.control.rolling_milp import (
    RollingMilpController,
    greedy_warm_start_actions,
)
from sim.control.replay import (
    action_for_well_control_mode,
    replay_native_actions,
)
from sim.control.shikha2025 import (
    Shikha2025Config,
    solve_shikha2025,
)
from sim.environment import CCSEnvConfig, build_phase1_env
from sim.metrics import run_recorded_episode


CONTROLLERS = (
    "fixed_assignment",
    "greedy",
    "rolling_milp",
    "full_milp",
    "shikha2025",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=8_100_001)
    parser.add_argument(
        "--controllers",
        nargs="+",
        choices=CONTROLLERS,
        default=list(CONTROLLERS),
    )
    parser.add_argument("--online-episode-hours", type=int, default=72)
    parser.add_argument("--forecast-context-hours", type=int, default=168)
    parser.add_argument("--rolling-replan-hours", type=int, default=24)
    parser.add_argument("--rolling-planning-horizon-hours", type=int, default=48)
    parser.add_argument("--rolling-time-limit-seconds", type=float, default=5.0)
    parser.add_argument(
        "--rolling-warm-start-mode",
        choices=("greedy", "none"),
        default="greedy",
    )
    parser.add_argument("--full-milp-horizon-hours", type=int, default=48)
    parser.add_argument("--full-milp-time-limit-seconds", type=float, default=30.0)
    parser.add_argument("--shikha-max-iterations", type=int, default=18)
    parser.add_argument(
        "--shikha-active-window-hours", type=int, default=120
    )
    parser.add_argument(
        "--shikha-fix-window-hours", type=int, default=60
    )
    parser.add_argument(
        "--shikha-subproblem-time-limit-seconds",
        type=float,
        default=30.0,
    )
    parser.add_argument(
        "--shikha-repair-time-limit-seconds",
        type=float,
        default=30.0,
    )
    parser.add_argument(
        "--shikha-tolerance-relative", type=float, default=0.02
    )
    parser.add_argument("--shikha-step-size", type=float, default=1.0)
    parser.add_argument("--mip-gap-relative", type=float)
    parser.add_argument("--solver-threads", type=int, default=4)
    parser.add_argument(
        "--purpose",
        default="implementation_smoke_test_not_formal_results",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    positive = (
        "online_episode_hours",
        "forecast_context_hours",
        "rolling_replan_hours",
        "rolling_planning_horizon_hours",
        "rolling_time_limit_seconds",
        "full_milp_horizon_hours",
        "full_milp_time_limit_seconds",
        "shikha_max_iterations",
        "shikha_active_window_hours",
        "shikha_fix_window_hours",
        "shikha_subproblem_time_limit_seconds",
        "shikha_repair_time_limit_seconds",
        "shikha_step_size",
    )
    if any(float(getattr(args, name)) <= 0.0 for name in positive):
        parser.error("all horizons, intervals and time limits must be positive")
    if args.mip_gap_relative is not None and args.mip_gap_relative < 0.0:
        parser.error("mip gap must be non-negative")
    if not 0.0 < args.shikha_tolerance_relative < 1.0:
        parser.error("Shikha tolerance must lie between zero and one")
    if args.shikha_fix_window_hours > args.shikha_active_window_hours:
        parser.error("Shikha fix window cannot exceed active window")
    if args.solver_threads <= 0:
        parser.error("solver threads must be positive")
    return args


def make_env(episode_hours: int, forecast_context_hours: int):
    generator = ReplayableDifficultyScenarioGenerator(
        episode_hours=int(episode_hours) + int(forecast_context_hours),
        weather_process="window",
        hard_probability=0.5,
        scenario_protocol="unified_window_v1",
    )
    return build_phase1_env(
        scenario="northern_lights_phase1_3vessels",
        scenario_generator=generator,
        weather_mode="window",
        config=CCSEnvConfig(
            episode_hours=int(episode_hours),
            include_goal_obs=False,
            reward_mode="economic",
            injection_reward_eur_per_t=0.0,
            store_reward_eur_per_t=0.0,
            vent_penalty_weight=1.0,
            operating_cost_weight=1.0,
            enforce_full_load_dispatch=False,
            require_empty_terminal_departure=True,
            well_control_mode="automatic_max",
        ),
    )


def _cleanup_cost(env) -> float:
    return _terminal_cleanup_cost_for_state(
        env,
        env.cost_model.parameters,
    )


def _compact_mip_start_audit(audit) -> dict[str, object]:
    return {
        "total_variables": audit.total_variables,
        "initialized_variables": audit.initialized_variables,
        "missing_variable_count": audit.missing_variable_count,
        "bound_violation_count": audit.bound_violation_count,
        "integrality_violation_count": audit.integrality_violation_count,
        "total_constraints": audit.total_constraints,
        "evaluated_constraints": audit.evaluated_constraints,
        "partial_constraint_count": audit.partial_constraint_count,
        "violated_constraint_count": audit.violated_constraint_count,
        "max_constraint_violation": audit.max_constraint_violation,
        "top_violations": [
            {
                "constraint": violation.constraint,
                "sense": violation.sense,
                "residual": violation.residual,
                "violation": violation.violation,
                "variable_names": list(violation.variable_names[:20]),
            }
            for violation in audit.top_violations
        ],
    }


def _failure_row(
    controller: str,
    seed: int,
    error: Exception,
    wall_clock_seconds: float,
    simulator_usage,
    *,
    evaluation_role: str,
) -> dict[str, object]:
    return {
        "controller": controller,
        "seed": int(seed),
        "evaluation_role": evaluation_role,
        "online_comparable": evaluation_role == "online_controller",
        "run_status": "failed",
        "error_type": type(error).__name__,
        "error_message": str(error),
        "wall_clock_seconds": float(wall_clock_seconds),
        **simulator_usage.as_dict(),
        "operating_cost": None,
        "total_cost": None,
        "stored_t": None,
        "vented_t": None,
    }


def _run_simple_controller(
    args,
    controller: str,
    executed_actions_out: list[dict[str, list[int]]] | None = None,
) -> dict[str, object]:
    env = make_env(
        args.online_episode_hours,
        args.forecast_context_hours,
    )
    policy = (
        make_cluster_shuttle_policy(env)
        if controller == "fixed_assignment"
        else greedy_shuttle_policy
    )
    recording_policy = _ActionRecorder(policy)
    record = run_recorded_episode(
        env,
        recording_policy,
        controller=controller,
        seed=args.seed,
        terminal_cleanup_cost=_cleanup_cost,
    )
    if executed_actions_out is not None:
        executed_actions_out.extend(recording_policy.actions)
    return {
        **record.as_dict(),
        "evaluation_role": "online_controller",
        "online_comparable": True,
        "run_status": "completed",
        "error_type": "",
        "error_message": "",
        "executed_action_count": len(recording_policy.actions),
    }


class _ActionRecorder:
    def __init__(self, policy) -> None:
        self.policy = policy
        self.actions: list[dict[str, list[int]]] = []

    def __call__(self, env):
        action = self.policy(env)
        self.actions.append(_json_action(action))
        return action


def _json_action(action) -> dict[str, list[int]]:
    return {
        str(group): [int(choice) for choice in choices]
        for group, choices in action.items()
    }


def _run_rolling_milp(
    args,
    executed_actions_out: list[dict[str, list[int]]] | None = None,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    env = make_env(
        args.online_episode_hours,
        args.forecast_context_hours,
    )
    controller = RollingMilpController(
        env,
        replan_every=args.rolling_replan_hours,
        planning_horizon_h=args.rolling_planning_horizon_hours,
        time_limit_s=args.rolling_time_limit_seconds,
        mip_gap_rel=args.mip_gap_relative,
        solver_threads=args.solver_threads,
        progress=lambda message: print(message, flush=True),
        objective_mode="economic",
        terminal_cleanup_value=True,
        terminal_cleanup_mip_start_mode="complete",
        shifted_milp_warm_start=False,
        warm_start_mode=getattr(args, "rolling_warm_start_mode", "greedy"),
    )
    recording_controller = _ActionRecorder(controller)
    usage_before = env.simulator_step_usage()
    started_at = time.perf_counter()
    try:
        record = run_recorded_episode(
            env,
            recording_controller,
            controller="rolling_milp",
            seed=args.seed,
            terminal_cleanup_cost=_cleanup_cost,
        )
        row = {
            **record.as_dict(),
            "evaluation_role": "online_controller",
            "online_comparable": True,
            "run_status": "completed",
            "error_type": "",
            "error_message": "",
        }
    except Exception as error:
        row = _failure_row(
            "rolling_milp",
            args.seed,
            error,
            time.perf_counter() - started_at,
            env.simulator_step_usage() - usage_before,
            evaluation_role="online_controller",
        )
    row.update(
        {
            "solver_replan_count": controller.replan_count,
            "solver_status_counts": json.dumps(
                controller.status_counts,
                sort_keys=True,
            ),
            "solver_failure_count": sum(
                1
                for diagnostic in controller.replan_diagnostics
                if not diagnostic["solver_is_valid"]
                or not diagnostic["execution_replay_is_valid"]
            ),
            "solver_timeout_count": sum(
                "time" in str(diagnostic["termination_reason"]).lower()
                for diagnostic in controller.replan_diagnostics
            ),
            "solver_solve_wall_seconds": sum(
                float(diagnostic["solve_wall_s"])
                for diagnostic in controller.replan_diagnostics
            ),
            "solver_time_limit_seconds_per_replan": float(
                args.rolling_time_limit_seconds
            ),
            "solver_threads": int(args.solver_threads),
            "solver_executable": shutil.which("cplex") or "",
            "warm_start_mode": controller.warm_start_mode,
            "shifted_warm_start": controller.shifted_milp_warm_start,
            "fallback_used": False,
            "executed_action_count": len(recording_controller.actions),
        }
    )
    if executed_actions_out is not None:
        executed_actions_out.extend(recording_controller.actions)
    return row, controller.replan_diagnostics


def _full_milp_success_row(
    args,
    env,
    result,
    replay,
    wall_clock_seconds,
    usage,
):
    final_stage = result.stage_diagnostics[-1] if result.stage_diagnostics else None
    actual = replay.actual
    ledger = env.ledger
    stored_t = float(actual.stored_t)
    terminal_cleanup_cost = _cleanup_cost(env)
    operating_cost = float(actual.operating_cost) + terminal_cleanup_cost
    total_cost = float(actual.total_cost) + terminal_cleanup_cost
    parameters = env.cost_model.parameters
    vessel_fuel = max(0.0, float(ledger.vessel_fuel))
    loading = max(0.0, float(ledger.loading))
    unloading = max(0.0, float(ledger.unloading))
    sailing_hours = vessel_fuel / parameters.vessel_fuel_eur_per_h_sailing
    loading_hours = loading / parameters.hoteling_fuel_eur_per_h
    unloading_hours = unloading / parameters.hoteling_fuel_eur_per_h
    total_vessel_hours = (
        int(args.full_milp_horizon_hours) * len(env.vessel_ids)
    )
    return {
        "controller": "full_milp",
        "seed": int(args.seed),
        "evaluation_role": "offline_reference",
        "online_comparable": False,
        "run_status": (
            "completed" if replay.is_executable else "replay_failed"
        ),
        "error_type": "",
        "error_message": "",
        "wall_clock_seconds": float(wall_clock_seconds),
        "controller_decision_calls": 1,
        **usage.as_dict(),
        "solver_status": result.status,
        "solver_is_valid": bool(result.is_valid),
        "solver_validation_error": result.validation_error,
        "solver_incumbent_objective": (
            final_stage.objective_value if final_stage is not None else None
        ),
        "solver_augmented_objective": float(
            result.augmented_objective_value
        ),
        "solver_best_bound": (
            final_stage.best_bound if final_stage is not None else None
        ),
        "solver_relative_gap": (
            final_stage.relative_gap if final_stage is not None else None
        ),
        "solver_termination_reason": (
            final_stage.termination_reason if final_stage is not None else ""
        ),
        "solver_warm_start_accepted": (
            final_stage.warm_start_accepted if final_stage is not None else None
        ),
        "solver_solve_wall_seconds": (
            final_stage.wall_time_s if final_stage is not None else None
        ),
        "solver_time_limit_seconds": float(
            args.full_milp_time_limit_seconds
        ),
        "solver_threads": int(args.solver_threads),
        "solver_executable": shutil.which("cplex") or "",
        "warm_start_mode": "greedy",
        "fallback_used": False,
        "replay_is_executable": bool(replay.is_executable),
        "replay_is_exact": None,
        "replay_mismatches": ";".join(replay.mismatches),
        "planning_horizon_hours": int(result.horizon_h),
        "evaluation_horizon_hours": int(args.full_milp_horizon_hours),
        "episode_operating_cost": float(actual.operating_cost),
        "episode_total_cost": float(actual.total_cost),
        "terminal_cleanup_operating_cost": float(
            terminal_cleanup_cost
        ),
        "solver_terminal_cleanup_operating_cost": float(
            result.terminal_cleanup_cost
        ),
        "operating_cost": operating_cost,
        "total_cost": total_cost,
        "replay_minus_solver_objective": (
            total_cost - float(result.augmented_objective_value)
        ),
        "cost_per_stored_t": (
            operating_cost / stored_t if stored_t > 1e-9 else None
        ),
        "total_cost_per_stored_t": (
            total_cost / stored_t if stored_t > 1e-9 else None
        ),
        "vessel_fuel": vessel_fuel,
        "conditioning": max(0.0, float(ledger.conditioning)),
        "reconditioning": max(0.0, float(ledger.reconditioning)),
        "loading": loading,
        "unloading": unloading,
        "vent_penalty": float(ledger.vent_penalty),
        "storage_shortfall_penalty": float(
            ledger.storage_shortfall_penalty
        ),
        "stored_t": stored_t,
        "vented_t": float(actual.vented_t),
        "captured_t": float(actual.captured_t),
        "in_transit_t": float(actual.in_transit_t),
        "vessel_sailing_hours": sailing_hours,
        "vessel_loading_hours": loading_hours,
        "vessel_unloading_hours": unloading_hours,
        "vessel_waiting_hours": max(
            0.0,
            total_vessel_hours
            - sailing_hours
            - loading_hours
            - unloading_hours,
        ),
        "loaded_t": max(0.0, float(ledger.loaded_t)),
        "unloaded_t": max(0.0, float(ledger.unloaded_t)),
    }


def _run_full_milp(
    args,
    executed_actions_out: list[dict[str, list[int]]] | None = None,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    evaluation_horizon_h = int(args.full_milp_horizon_hours)
    planning_horizon_h = evaluation_horizon_h
    planning_env = make_env(
        evaluation_horizon_h,
        args.forecast_context_hours,
    )
    planning_env.reset(seed=args.seed)
    usage_before = planning_env.simulator_step_usage()
    started_at = time.perf_counter()
    evaluation_actions: list[dict[str, list[int]]] = []
    try:
        print(
            f"full_milp seed={args.seed}: building "
            f"{planning_horizon_h} h Greedy warm start",
            flush=True,
        )
        warm_start = greedy_warm_start_actions(
            planning_env,
            planning_horizon_h,
        )
        print(
            f"full_milp seed={args.seed}: starting CPLEX with "
            f"{args.full_milp_time_limit_seconds:.0f} s limit",
            flush=True,
        )
        result = solve_full_scenario_with_cplex(
            planning_env,
            horizon_h=planning_horizon_h,
            economics=planning_env.cost_model.parameters,
            warm_start_native_actions_by_hour=warm_start,
            time_limit_s=args.full_milp_time_limit_seconds,
            mip_gap_rel=args.mip_gap_relative,
            threads=args.solver_threads,
            cplex_options=[
                "set parallel 1",
                "set simplex tolerances feasibility 1e-7",
            ],
            economic_objective=True,
            environment_aligned_service=True,
            terminal_cleanup_value=True,
            terminal_cleanup_mip_start_mode="complete",
            cleanup_unary_trip_slots=True,
            vessel_visit_load_cuts=True,
            source_visit_vent_cuts=True,
            terminal_visit_cuts=True,
            service_reachability_cuts=True,
            route_cargo_flow_linking=True,
        )
        diagnostics = [
            {
                "stage": stage.stage,
                "time_limit_s": stage.time_limit_s,
                "wall_time_s": stage.wall_time_s,
                "status": stage.status,
                "objective_value": stage.objective_value,
                "best_bound": stage.best_bound,
                "relative_gap": stage.relative_gap,
                "termination_reason": stage.termination_reason,
                "warm_start_requested": stage.warm_start_requested,
                "warm_start_accepted": stage.warm_start_accepted,
            }
            for stage in result.stage_diagnostics
        ]
        mip_start_audit = getattr(result, "mip_start_audit", None)
        if mip_start_audit is not None:
            diagnostics.append(
                {
                    "stage": "mip_start_audit",
                    **_compact_mip_start_audit(mip_start_audit),
                }
            )
        if not result.is_valid:
            error = RuntimeError(
                result.validation_error
                or f"full MILP solver status {result.status}"
            )
            row = _failure_row(
                "full_milp",
                args.seed,
                error,
                time.perf_counter() - started_at,
                planning_env.simulator_step_usage() - usage_before,
                evaluation_role="offline_reference",
            )
            final_stage = (
                result.stage_diagnostics[-1]
                if result.stage_diagnostics
                else None
            )
            row.update(
                {
                    "solver_status": result.status,
                    "solver_is_valid": False,
                    "solver_validation_error": result.validation_error,
                    "solver_best_bound": (
                        final_stage.best_bound
                        if final_stage is not None
                        else None
                    ),
                    "solver_relative_gap": (
                        final_stage.relative_gap
                        if final_stage is not None
                        else None
                    ),
                    "solver_termination_reason": (
                        final_stage.termination_reason
                        if final_stage is not None
                        else ""
                    ),
                    "solver_warm_start_accepted": (
                        final_stage.warm_start_accepted
                        if final_stage is not None
                        else None
                    ),
                    "solver_time_limit_seconds": float(
                        args.full_milp_time_limit_seconds
                    ),
                    "solver_threads": int(args.solver_threads),
                    "solver_executable": shutil.which("cplex") or "",
                    "planning_horizon_hours": planning_horizon_h,
                    "evaluation_horizon_hours": evaluation_horizon_h,
                    "warm_start_mode": "greedy",
                    "fallback_used": False,
                }
            )
            row["executed_action_count"] = 0
            if executed_actions_out is not None:
                executed_actions_out.extend(evaluation_actions)
            return row, diagnostics
        replay_env = make_env(
            evaluation_horizon_h,
            args.forecast_context_hours,
        )
        replay_env.reset(seed=args.seed)
        evaluation_actions = [
            action_for_well_control_mode(replay_env, action)
            for action in result.native_actions_by_hour[
                :evaluation_horizon_h
            ]
        ]
        replay = replay_native_actions(
            replay_env,
            evaluation_actions,
            horizon_h=evaluation_horizon_h,
            copy_env=False,
        )
        print(
            f"full_milp seed={args.seed}: solver={result.status}; "
            f"replay_executable={replay.is_executable}",
            flush=True,
        )
        usage = (
            planning_env.simulator_step_usage() - usage_before
        )
        row = _full_milp_success_row(
            args,
            replay_env,
            result,
            replay,
            time.perf_counter() - started_at,
            usage,
        )
        row["executed_action_count"] = len(evaluation_actions)
        if executed_actions_out is not None:
            executed_actions_out.extend(evaluation_actions)
        return row, diagnostics
    except Exception as error:
        failure = _failure_row(
            "full_milp",
            args.seed,
            error,
            time.perf_counter() - started_at,
            planning_env.simulator_step_usage() - usage_before,
            evaluation_role="offline_reference",
        )
        failure.update(
            {
                "solver_time_limit_seconds": float(
                    args.full_milp_time_limit_seconds
                ),
                "solver_threads": int(args.solver_threads),
                "solver_executable": shutil.which("cplex") or "",
                "planning_horizon_hours": planning_horizon_h,
                "evaluation_horizon_hours": evaluation_horizon_h,
                "warm_start_mode": "greedy",
                "fallback_used": False,
                "executed_action_count": len(evaluation_actions),
            }
        )
        if executed_actions_out is not None:
            executed_actions_out.extend(evaluation_actions)
        return (
            failure,
            [],
        )


def _run_shikha2025(
    args,
    executed_actions_out: list[dict[str, list[int]]] | None = None,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    evaluation_horizon_h = int(args.full_milp_horizon_hours)
    planning_env = make_env(
        evaluation_horizon_h,
        args.forecast_context_hours,
    )
    planning_env.reset(seed=args.seed)
    usage_before = planning_env.simulator_step_usage()
    started_at = time.perf_counter()
    evaluation_actions: list[dict[str, list[int]]] = []
    try:
        decomposition = solve_shikha2025(
            planning_env,
            horizon_h=evaluation_horizon_h,
            config=Shikha2025Config(
                active_window_h=args.shikha_active_window_hours,
                fix_window_h=args.shikha_fix_window_hours,
                max_iterations=args.shikha_max_iterations,
                tolerance_rel=args.shikha_tolerance_relative,
                step_size=args.shikha_step_size,
                subproblem_time_limit_s=(
                    args.shikha_subproblem_time_limit_seconds
                ),
                repair_time_limit_s=(
                    args.shikha_repair_time_limit_seconds
                ),
                mip_gap_rel=args.mip_gap_relative,
                threads=args.solver_threads,
            ),
            progress=lambda message: print(message, flush=True),
        )
        result = decomposition.feasible_result
        diagnostics = [
            {
                "iteration": item.iteration,
                "surrogate_dual_objective": (
                    item.surrogate_dual_objective
                ),
                "best_feasible_objective": (
                    item.best_feasible_objective
                ),
                "relative_surrogate_gap": (
                    item.relative_surrogate_gap
                ),
                "maximum_service_violation": (
                    item.maximum_service_violation
                ),
                "multiplier_norm": item.multiplier_norm,
                "repair_status": item.repair_status,
                "repair_is_valid": item.repair_is_valid,
                "wall_time_s": item.wall_time_s,
                "subproblems": [
                    {
                        "vessel_id": sub.vessel_id,
                        "shrinking_stage_count": (
                            sub.shrinking_stage_count
                        ),
                        "statuses": list(sub.statuses),
                        "augmented_objective": (
                            sub.augmented_objective
                        ),
                        "wall_time_s": sub.wall_time_s,
                    }
                    for sub in item.subproblems
                ],
            }
            for item in decomposition.iterations
        ]
        replay_env = make_env(
            evaluation_horizon_h,
            args.forecast_context_hours,
        )
        replay_env.reset(seed=args.seed)
        evaluation_actions = [
            action_for_well_control_mode(replay_env, action)
            for action in result.native_actions_by_hour[
                :evaluation_horizon_h
            ]
        ]
        replay = replay_native_actions(
            replay_env,
            evaluation_actions,
            horizon_h=evaluation_horizon_h,
            copy_env=False,
        )
        row = _full_milp_success_row(
            args,
            replay_env,
            result,
            replay,
            time.perf_counter() - started_at,
            planning_env.simulator_step_usage() - usage_before,
        )
        row.update(
            {
                "controller": "shikha2025",
                "algorithm": (
                    "vessel_lagrangian_plus_shrinking_horizon"
                ),
                "decomposition_converged": decomposition.converged,
                "decomposition_stopping_reason": (
                    decomposition.stopping_reason
                ),
                "decomposition_iterations": len(
                    decomposition.iterations
                ),
                "subproblem_solve_count": sum(
                    sub.shrinking_stage_count
                    for item in decomposition.iterations
                    for sub in item.subproblems
                ),
                "solver_solve_wall_seconds": decomposition.wall_time_s,
                "solver_time_limit_seconds": None,
                "subproblem_time_limit_seconds": float(
                    args.shikha_subproblem_time_limit_seconds
                ),
                "repair_time_limit_seconds": float(
                    args.shikha_repair_time_limit_seconds
                ),
                "paper_active_window_hours": int(
                    args.shikha_active_window_hours
                ),
                "paper_fix_window_hours": int(
                    args.shikha_fix_window_hours
                ),
                "paper_tolerance_relative": float(
                    args.shikha_tolerance_relative
                ),
                "paper_step_size": float(args.shikha_step_size),
                "executed_action_count": len(evaluation_actions),
            }
        )
        if executed_actions_out is not None:
            executed_actions_out.extend(evaluation_actions)
        return row, diagnostics
    except Exception as error:
        failure = _failure_row(
            "shikha2025",
            args.seed,
            error,
            time.perf_counter() - started_at,
            planning_env.simulator_step_usage() - usage_before,
            evaluation_role="offline_reference",
        )
        failure.update(
            {
                "planning_horizon_hours": evaluation_horizon_h,
                "evaluation_horizon_hours": evaluation_horizon_h,
                "solver_threads": int(args.solver_threads),
                "solver_executable": shutil.which("cplex") or "",
                "executed_action_count": len(evaluation_actions),
            }
        )
        if executed_actions_out is not None:
            executed_actions_out.extend(evaluation_actions)
        return failure, []


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = list(
        dict.fromkeys(key for row in rows for key in row)
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run(args):
    if args.out_dir.exists() and any(args.out_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"refusing to overwrite non-empty output directory: {args.out_dir}"
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    diagnostics: dict[str, object] = {}
    executed_actions: dict[str, list[dict[str, list[int]]]] = {}
    for controller in args.controllers:
        if controller in {"fixed_assignment", "greedy"}:
            started_at = time.perf_counter()
            actions: list[dict[str, list[int]]] = []
            try:
                rows.append(_run_simple_controller(args, controller, actions))
            except Exception as error:
                rows.append(
                    _failure_row(
                        controller,
                        args.seed,
                        error,
                        time.perf_counter() - started_at,
                        make_env(
                            args.online_episode_hours,
                            args.forecast_context_hours,
                        ).simulator_step_usage(),
                        evaluation_role="online_controller",
                    )
                )
            executed_actions[controller] = actions
        elif controller == "rolling_milp":
            actions: list[dict[str, list[int]]] = []
            row, replans = _run_rolling_milp(args, actions)
            rows.append(row)
            diagnostics["rolling_milp_replans"] = replans
            executed_actions["rolling_milp"] = actions
        elif controller == "full_milp":
            actions = []
            row, stages = _run_full_milp(args, actions)
            rows.append(row)
            diagnostics["full_milp_stages"] = stages
            executed_actions["full_milp"] = actions
        elif controller == "shikha2025":
            actions = []
            row, iterations = _run_shikha2025(args, actions)
            rows.append(row)
            diagnostics["shikha2025_iterations"] = iterations
            executed_actions["shikha2025"] = actions

    _write_csv(args.out_dir / "per_controller.csv", rows)
    action_payload = {
        "protocol": "unified_window_v1",
        "purpose": args.purpose,
        "seed": int(args.seed),
        "actions_by_controller": {
            controller: [_json_action(action) for action in actions]
            for controller, actions in executed_actions.items()
        },
    }
    (args.out_dir / "executed_actions.json").write_text(
        json.dumps(action_payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    payload = {
        "protocol": "unified_window_v1",
        "purpose": args.purpose,
        "configuration": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "rows": rows,
        "diagnostics": diagnostics,
    }
    (args.out_dir / "smoke_summary.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, allow_nan=False), flush=True)
    return payload


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
