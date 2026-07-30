"""Run the exploratory Native MPC with the common terminal cleanup value."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.smoke_test_paper_controllers import (
    _cleanup_cost,
    _write_csv,
    make_env,
)
from sim.control.native_mpc import RollingNativeMpcController
from sim.metrics import run_recorded_episode


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=8_100_001)
    parser.add_argument("--episode-hours", type=int, default=720)
    parser.add_argument("--forecast-context-hours", type=int, default=168)
    parser.add_argument("--replan-hours", type=int, default=24)
    parser.add_argument("--planning-horizon-hours", type=int, default=168)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    for name in (
        "episode_hours",
        "forecast_context_hours",
        "replan_hours",
        "planning_horizon_hours",
    ):
        if int(getattr(args, name)) <= 0:
            parser.error(f"{name} must be positive")
    return args


def run(args):
    if args.out_dir.exists() and any(args.out_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"refusing to overwrite non-empty output directory: {args.out_dir}"
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    env = make_env(
        args.episode_hours,
        args.forecast_context_hours,
    )
    controller = RollingNativeMpcController(
        env,
        replan_every=args.replan_hours,
        planning_horizon_h=args.planning_horizon_hours,
        progress=lambda message: print(message, flush=True),
        objective_mode="economic",
        terminal_cleanup_value=True,
    )
    record = run_recorded_episode(
        env,
        controller,
        controller="native_mpc_cleanup",
        seed=args.seed,
        terminal_cleanup_cost=_cleanup_cost,
    )
    row = {
        **record.as_dict(),
        "evaluation_role": "exploratory_online_controller",
        "online_comparable": True,
        "run_status": "completed",
        "objective_mode": controller.objective_mode,
        "replan_hours": int(controller.replan_every),
        "planning_horizon_hours": int(controller.planning_horizon_h),
        "terminal_cleanup_value": controller.terminal_cleanup_value,
        "candidate_evaluations": int(controller.candidate_evaluations),
        "last_candidate_name": controller.last_candidate_name,
        "last_trace_replay_is_valid": controller.last_trace_replay_is_valid,
        "last_trace_replay_is_exact": controller.last_trace_replay_is_exact,
        "last_trace_replay_mismatches": ";".join(
            controller.last_trace_replay_mismatches
        ),
    }
    expected_total = (
        float(row["episode_total_cost"])
        + float(row["terminal_cleanup_operating_cost"])
    )
    if abs(float(row["total_cost"]) - expected_total) > 1e-6:
        raise AssertionError(
            "reported total cost does not equal episode cost plus terminal cleanup"
        )
    _write_csv(args.out_dir / "per_controller.csv", [row])
    payload = {
        "kind": "exploratory_native_mpc_terminal_cleanup",
        "configuration": {
            "seed": int(args.seed),
            "episode_hours": int(args.episode_hours),
            "forecast_context_hours": int(args.forecast_context_hours),
            "replan_hours": int(args.replan_hours),
            "planning_horizon_hours": int(args.planning_horizon_hours),
            "objective_mode": "economic",
            "terminal_cleanup_value": True,
            "reported_total_cost": (
                "episode total cost + common compact terminal cleanup"
            ),
            "paper_baseline": False,
        },
        "result": row,
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2), flush=True)
    return payload


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
