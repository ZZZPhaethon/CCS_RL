"""Smoke-test the paper baselines and time-limited MILP paths on one seed."""

from __future__ import annotations

import argparse
import csv
import json
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
from sim.environment import CCSEnvConfig, build_phase1_env
from sim.metrics import run_recorded_episode


CONTROLLERS = (
    "fixed_assignment",
    "greedy",
    "rolling_milp",
    "full_milp",
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
    parser.add_argument("--full-milp-horizon-hours", type=int, default=48)
    parser.add_argument("--full-milp-time-limit-seconds", type=float, default=30.0)
    parser.add_argument("--mip-gap-relative", type=float)
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
    )
    if any(float(getattr(args, name)) <= 0.0 for name in positive):
        parser.error("all horizons, intervals and time limits must be positive")
    if args.mip_gap_relative is not None and args.mip_gap_relative < 0.0:
        parser.error("mip gap must be non-negative")
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


def _run_simple_controller(args, controller: str) -> dict[str, object]:
    env = make_env(
        args.online_episode_hours,
        args.forecast_context_hours,
    )
    policy = (
        make_cluster_shuttle_policy(env)
        if controller == "fixed_assignment"
        else greedy_shuttle_policy
    )
    record = run_recorded_episode(
        env,
        policy,
        controller=controller,
        seed=args.seed,
        terminal_cleanup_cost=_cleanup_cost,
    )
    return {
        **record.as_dict(),
        "evaluation_role": "online_controller",
        "online_comparable": True,
        "run_status": "completed",
        "error_type": "",
        "error_message": "",
    }


def _run_rolling_milp(args) -> tuple[dict[str, object], list[dict[str, object]]]:
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
        objective_mode="economic",
        terminal_cleanup_value=True,
        shifted_milp_warm_start=False,
        warm_start_mode="greedy",
    )
    usage_before = env.simulator_step_usage()
    started_at = time.perf_counter()
    try:
        record = run_recorded_episode(
            env,
            controller,
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
            "warm_start_mode": controller.warm_start_mode,
            "shifted_warm_start": controller.shifted_milp_warm_start,
            "fallback_used": False,
        }
    )
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
        "operating_cost": operating_cost,
        "total_cost": total_cost,
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


def _run_full_milp(args) -> tuple[dict[str, object], list[dict[str, object]]]:
    evaluation_horizon_h = int(args.full_milp_horizon_hours)
    planning_horizon_h = (
        evaluation_horizon_h + int(args.forecast_context_hours)
    )
    planning_env = make_env(
        planning_horizon_h,
        0,
    )
    planning_env.reset(seed=args.seed)
    usage_before = planning_env.simulator_step_usage()
    started_at = time.perf_counter()
    try:
        warm_start = greedy_warm_start_actions(
            planning_env,
            planning_horizon_h,
        )
        result = solve_full_scenario_with_cplex(
            planning_env,
            horizon_h=planning_horizon_h,
            economics=planning_env.cost_model.parameters,
            warm_start_native_actions_by_hour=warm_start,
            time_limit_s=args.full_milp_time_limit_seconds,
            mip_gap_rel=args.mip_gap_relative,
            economic_objective=True,
            environment_aligned_service=True,
            terminal_cleanup_value=True,
            cleanup_unary_trip_slots=True,
            vessel_visit_load_cuts=True,
            source_visit_vent_cuts=True,
            terminal_visit_cuts=True,
            service_reachability_cuts=True,
            route_cargo_flow_linking=True,
        )
        if not result.is_valid:
            raise RuntimeError(
                result.validation_error
                or f"full MILP solver status {result.status}"
            )
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
        return row, diagnostics
    except Exception as error:
        return (
            _failure_row(
                "full_milp",
                args.seed,
                error,
                time.perf_counter() - started_at,
                planning_env.simulator_step_usage() - usage_before,
                evaluation_role="offline_reference",
            ),
            [],
        )


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
    for controller in args.controllers:
        if controller in {"fixed_assignment", "greedy"}:
            started_at = time.perf_counter()
            try:
                rows.append(_run_simple_controller(args, controller))
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
        elif controller == "rolling_milp":
            row, replans = _run_rolling_milp(args)
            rows.append(row)
            diagnostics["rolling_milp_replans"] = replans
        elif controller == "full_milp":
            row, stages = _run_full_milp(args)
            rows.append(row)
            diagnostics["full_milp_stages"] = stages

    _write_csv(args.out_dir / "per_controller.csv", rows)
    payload = {
        "protocol": "unified_window_v1",
        "purpose": "implementation_smoke_test_not_formal_results",
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
