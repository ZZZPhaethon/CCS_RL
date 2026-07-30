"""Diagnose how the shared CPLEX MILP scales with planning horizon."""

from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time

from sim.control.cplex_milp import (
    _terminal_cleanup_cost_for_state,
    replay_full_scenario_cplex_plan,
    solve_full_scenario_with_cplex,
)
from sim.control.rolling_milp import greedy_warm_start_actions

from smoke_test_paper_controllers import make_env


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = (
    ROOT
    / "experiments_results"
    / "E0"
    / "milp_horizon_scaling_seed_8100001"
)
DEFAULT_HORIZONS = (12, 24, 48, 72, 96, 168)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=8_100_001)
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=DEFAULT_HORIZONS,
    )
    parser.add_argument("--time-limit-seconds", type=float, default=300.0)
    parser.add_argument("--solver-threads", type=int, default=4)
    parser.add_argument("--online-episode-hours", type=int, default=720)
    parser.add_argument("--forecast-context-hours", type=int, default=168)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    if any(horizon <= 0 for horizon in args.horizons):
        parser.error("all horizons must be positive")
    if len(set(args.horizons)) != len(args.horizons):
        parser.error("horizons must not contain duplicates")
    if max(args.horizons) > args.online_episode_hours:
        parser.error("horizons must remain within the online episode")
    if args.time_limit_seconds <= 0.0:
        parser.error("time limit must be positive")
    if args.solver_threads <= 0:
        parser.error("solver threads must be positive")
    if args.forecast_context_hours < 0:
        parser.error("forecast context must not be negative")
    return args


def _write_json(path: Path, payload) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _final_stage(result):
    return result.stage_diagnostics[-1] if result.stage_diagnostics else None


def _result_row(
    *,
    horizon_h: int,
    warm_start_wall_s: float,
    total_wall_s: float,
    result,
    replay,
    replay_cleanup_cost: float | None,
) -> dict[str, object]:
    stage = _final_stage(result)
    audit = result.mip_start_audit
    replay_total_with_cleanup = (
        None
        if replay is None or replay_cleanup_cost is None
        else float(replay.total_cost) + float(replay_cleanup_cost)
    )
    return {
        "horizon_h": int(horizon_h),
        "run_status": "completed" if result.is_valid else "invalid_solution",
        "solver_status": result.status,
        "solver_is_valid": bool(result.is_valid),
        "solver_validation_error": result.validation_error,
        "warm_start_wall_seconds": float(warm_start_wall_s),
        "solver_wall_seconds": (
            None if stage is None else float(stage.wall_time_s)
        ),
        "total_wall_seconds": float(total_wall_s),
        "total_variables": (
            None if audit is None else int(audit.total_variables)
        ),
        "total_constraints": (
            None if audit is None else int(audit.total_constraints)
        ),
        "presolved_rows": (
            None if stage is None else stage.reduced_rows
        ),
        "presolved_columns": (
            None if stage is None else stage.reduced_columns
        ),
        "presolved_nonzeros": (
            None if stage is None else stage.reduced_nonzeros
        ),
        "branch_and_bound_nodes": (
            None if stage is None else stage.nodes
        ),
        "simplex_iterations": (
            None if stage is None else stage.iterations
        ),
        "incumbent_objective": (
            None if stage is None else stage.objective_value
        ),
        "augmented_objective": (
            float(result.augmented_objective_value)
            if result.is_valid
            else None
        ),
        "best_bound": None if stage is None else stage.best_bound,
        "relative_gap": None if stage is None else stage.relative_gap,
        "termination_reason": (
            "" if stage is None else stage.termination_reason
        ),
        "warm_start_accepted": (
            None if stage is None else stage.warm_start_accepted
        ),
        "replay_is_executable": (
            None if replay is None else bool(replay.is_executable)
        ),
        "replay_is_exact": (
            None if replay is None else bool(replay.is_exact)
        ),
        "replay_mismatches": (
            "" if replay is None else ";".join(replay.mismatches)
        ),
        "replay_total_cost_with_cleanup": replay_total_with_cleanup,
        "replay_minus_solver_augmented_objective": (
            None
            if replay_total_with_cleanup is None or not result.is_valid
            else (
                replay_total_with_cleanup
                - float(result.augmented_objective_value)
            )
        ),
        "stored_t": None if replay is None else float(replay.stored_t),
        "vented_t": None if replay is None else float(replay.vented_t),
    }


def run(args) -> int:
    if shutil.which("cplex") is None:
        raise RuntimeError("CPLEX executable is not available on PATH")
    if (
        args.out_dir.exists()
        and any(args.out_dir.iterdir())
        and not args.overwrite
    ):
        raise FileExistsError(
            f"refusing to overwrite non-empty output directory: {args.out_dir}"
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    horizons = sorted(int(value) for value in args.horizons)
    config = {
        "purpose": "milp_horizon_scaling_implementation_diagnostic",
        "formal_paper_result": False,
        "seed": int(args.seed),
        "horizons_h": horizons,
        "time_limit_seconds_per_horizon": float(args.time_limit_seconds),
        "solver_threads": int(args.solver_threads),
        "online_episode_hours": int(args.online_episode_hours),
        "forecast_context_hours": int(args.forecast_context_hours),
        "sampled_scenario_hours": (
            int(args.online_episode_hours)
            + int(args.forecast_context_hours)
        ),
        "warm_start_mode": "greedy",
        "objective_mode": "economic",
        "terminal_cleanup_value": True,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "finished_at_utc": None,
    }
    _write_json(args.out_dir / "config.json", config)

    base_env = make_env(
        args.online_episode_hours,
        args.forecast_context_hours,
    )
    base_env.reset(seed=args.seed)

    rows: list[dict[str, object]] = []
    diagnostics: dict[str, object] = {}
    for horizon_h in horizons:
        print(f"horizon={horizon_h} h: building Greedy warm start", flush=True)
        planning_env = copy.deepcopy(base_env)
        started = time.perf_counter()
        warm_started = time.perf_counter()
        warm_start = greedy_warm_start_actions(planning_env, horizon_h)
        warm_start_wall_s = time.perf_counter() - warm_started

        print(
            f"horizon={horizon_h} h: starting CPLEX "
            f"({args.time_limit_seconds:.0f} s limit)",
            flush=True,
        )
        try:
            result = solve_full_scenario_with_cplex(
                planning_env,
                horizon_h=horizon_h,
                economics=planning_env.cost_model.parameters,
                warm_start_native_actions_by_hour=warm_start,
                time_limit_s=args.time_limit_seconds,
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
            stage = _final_stage(result)
            raw_log = "" if stage is None else stage.raw_log
            (args.out_dir / f"horizon_{horizon_h:03d}h_cplex.log").write_text(
                raw_log,
                encoding="utf-8",
            )

            replay = None
            replay_cleanup_cost = None
            if result.is_valid:
                replay_env = copy.deepcopy(base_env)
                replay = replay_full_scenario_cplex_plan(replay_env, result)
                replay_cleanup_cost = _terminal_cleanup_cost_for_state(
                    replay_env,
                    replay_env.cost_model.parameters,
                )
            row = _result_row(
                horizon_h=horizon_h,
                warm_start_wall_s=warm_start_wall_s,
                total_wall_s=time.perf_counter() - started,
                result=result,
                replay=replay,
                replay_cleanup_cost=replay_cleanup_cost,
            )
            diagnostics[str(horizon_h)] = {
                "stage_diagnostics": [
                    asdict(item) | {"raw_log": ""}
                    for item in result.stage_diagnostics
                ],
                "mip_start_audit": (
                    None
                    if result.mip_start_audit is None
                    else asdict(result.mip_start_audit)
                ),
                "solution_audit": (
                    None
                    if result.solution_audit is None
                    else asdict(result.solution_audit)
                ),
            }
        except Exception as error:
            row = {
                "horizon_h": int(horizon_h),
                "run_status": "failed",
                "error_type": type(error).__name__,
                "error_message": str(error),
                "warm_start_wall_seconds": float(warm_start_wall_s),
                "total_wall_seconds": float(time.perf_counter() - started),
            }
            diagnostics[str(horizon_h)] = {
                "error_type": type(error).__name__,
                "error_message": str(error),
            }

        rows.append(row)
        _write_csv(args.out_dir / "horizon_scaling.csv", rows)
        _write_json(
            args.out_dir / "horizon_scaling.json",
            {
                "config": config,
                "rows": rows,
                "diagnostics": diagnostics,
            },
        )
        print(
            f"horizon={horizon_h} h: {row['run_status']}; "
            f"status={row.get('solver_status', '')}; "
            f"gap={row.get('relative_gap')}; "
            f"replay_exact={row.get('replay_is_exact')}",
            flush=True,
        )

    config["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    _write_json(args.out_dir / "config.json", config)
    _write_json(
        args.out_dir / "horizon_scaling.json",
        {
            "config": config,
            "rows": rows,
            "diagnostics": diagnostics,
        },
    )
    return 0 if all(row["run_status"] == "completed" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
