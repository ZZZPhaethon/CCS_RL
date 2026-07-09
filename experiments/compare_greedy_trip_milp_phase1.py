"""Compare greedy_shuttle with relaxed/executable trip CPLEX MILPs on Phase 1."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

from sim.control.baselines import greedy_shuttle_policy
from sim.control.trip_milp import (
    replay_trip_milp_plan,
    solve_executable_trip_milp_with_cplex,
    solve_relaxed_trip_milp_with_cplex,
)
from sim.economics import CostModel, EconomicParameters
from sim.environment import CCSEnv, CCSEnvConfig
from sim.metrics import EpisodeMetrics, run_episode
from sim.network_scenarios import (
    available_fixed_scenario_choices,
    build_fixed_scenario_demo,
    fixed_scenario_locations,
)
from sim.scenario_generation import ScenarioConfig, ScenarioGenerator


def _scenario_config(hours: int) -> ScenarioConfig:
    return ScenarioConfig(
        episode_hours=hours,
        randomize_initial_inventory=True,
    )


def _make_env(
    hours: int,
    economics: EconomicParameters,
    fixed_scenario: str = "northern_lights_phase1_3vessels",
    storage_reward_eur_per_t: float | None = None,
) -> CCSEnv:
    network, _state = build_fixed_scenario_demo(fixed_scenario)
    return CCSEnv(
        network,
        fixed_scenario_locations(fixed_scenario),
        scenario_generator=ScenarioGenerator(config=_scenario_config(hours)),
        cost_model=CostModel(economics),
        config=CCSEnvConfig(episode_hours=hours, store_reward_eur_per_t=storage_reward_eur_per_t),
    )


def _metric_row(controller: str, metrics: EpisodeMetrics, solve_time_s: float) -> dict[str, object]:
    return {
        "controller": controller,
        "status": "replayed",
        "solve_time_s": solve_time_s,
        **metrics.as_dict(),
    }


def _trip_row(controller: str, result, solve_time_s: float, replay=None) -> dict[str, object]:
    total_cost_per_stored = result.total_cost / result.stored_t if result.stored_t > 0.0 else math.nan
    row = {
        "controller": controller,
        "status": result.status,
        "solve_time_s": solve_time_s,
        "horizon_hours": result.horizon_h,
        "stored_t": result.stored_t,
        "vented_t": result.vented_t,
        "in_transit_t": result.in_transit_t,
        "in_transit_growth_t": result.in_transit_growth_t,
        "shortfall_t": result.shortfall_t,
        "operating_cost": result.operating_cost,
        "total_cost": result.total_cost,
        "storage_reward_eur_per_t": result.storage_reward_eur_per_t,
        "net_reward": result.net_reward,
        "objective_value": result.objective_value,
        "cost_per_stored_t": result.cost_per_stored_t,
        "total_cost_per_stored_t": total_cost_per_stored,
        "vessel_fuel": result.vessel_fuel,
        "conditioning": result.conditioning,
        "reconditioning": result.reconditioning,
        "loading": result.loading,
        "unloading": result.unloading,
        "deliveries": result.deliveries,
        "is_valid": result.is_valid,
        "validation_error": result.validation_error,
        "max_binary_integrality_violation": result.max_binary_integrality_violation,
        "binary_count": result.binary_count,
        "variable_count": result.variable_count,
        "constraint_count": result.constraint_count,
    }
    if replay is not None:
        row.update(
            {
                "elapsed_hours": replay.elapsed_hours,
                "replay_stored_t": replay.stored_t,
                "replay_vented_t": replay.vented_t,
                "replay_operating_cost": replay.operating_cost,
                "replay_total_cost": replay.total_cost,
                "stored_gap_t": replay.stored_gap_t,
                "replay_is_executable": replay.is_executable,
                "replay_violations": ";".join(replay.violations),
            }
        )
    return row


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _log(message: str, output_dir: Path) -> None:
    with (output_dir / "progress.log").open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")
    try:
        print(message, flush=True)
    except OSError:
        pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hours", type=int, default=720)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=Path("output/phase1_3vessels_720h_greedy_vs_trip_milp_seed1"))
    parser.add_argument("--cplex-time-limit-s", type=float, default=43_200.0)
    parser.add_argument("--cplex-mip-gap-rel", type=float, default=0.01)
    parser.add_argument("--cplex-threads", type=int, default=None)
    parser.add_argument("--cplex-msg", action="store_true")
    parser.add_argument("--storage-shortfall-eur-per-t", type=float, default=0.0)
    parser.add_argument("--storage-reward-eur-per-t", type=float, default=1_000.0)
    parser.add_argument(
        "--fixed-scenario",
        choices=available_fixed_scenario_choices(),
        default="northern_lights_phase1_3vessels",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "progress.log").write_text("", encoding="utf-8")
    economics = EconomicParameters(storage_shortfall_eur_per_t=args.storage_shortfall_eur_per_t)

    _log(
        (
            f"scenario={args.fixed_scenario} seed={args.seed} hours={args.hours} "
            f"cplex_time_limit_s={args.cplex_time_limit_s:.0f} "
            f"storage_reward_eur_per_t={args.storage_reward_eur_per_t:g}"
        ),
        args.output_dir,
    )

    greedy_env = _make_env(
        args.hours,
        economics,
        args.fixed_scenario,
        storage_reward_eur_per_t=args.storage_reward_eur_per_t,
    )
    start = time.perf_counter()
    greedy_metrics = run_episode(greedy_env, greedy_shuttle_policy, seed=args.seed)
    rows = [_metric_row("greedy_shuttle", greedy_metrics, time.perf_counter() - start)]
    _log(
        (
            f"greedy_shuttle done: stored={greedy_metrics.stored_t:.1f} t "
            f"vented={greedy_metrics.vented_t:.1f} t total_cost={greedy_metrics.total_cost:.0f}"
        ),
        args.output_dir,
    )

    relaxed_env = _make_env(
        args.hours,
        economics,
        args.fixed_scenario,
        storage_reward_eur_per_t=args.storage_reward_eur_per_t,
    )
    relaxed_env.reset(seed=args.seed)
    _log("relaxed_trip MILP started", args.output_dir)
    start = time.perf_counter()
    relaxed = solve_relaxed_trip_milp_with_cplex(
        relaxed_env,
        horizon_h=args.hours,
        economics=economics,
        storage_reward_eur_per_t=args.storage_reward_eur_per_t,
        time_limit_s=args.cplex_time_limit_s,
        mip_gap_rel=args.cplex_mip_gap_rel,
        threads=args.cplex_threads,
        msg=args.cplex_msg,
    )
    rows.append(_trip_row("relaxed_trip_milp", relaxed, time.perf_counter() - start))
    _log(
        f"relaxed_trip MILP done: status={relaxed.status} stored={relaxed.stored_t:.1f} t total_cost={relaxed.total_cost:.0f}",
        args.output_dir,
    )

    executable_env = _make_env(
        args.hours,
        economics,
        args.fixed_scenario,
        storage_reward_eur_per_t=args.storage_reward_eur_per_t,
    )
    executable_env.reset(seed=args.seed)
    _log("executable_trip MILP started", args.output_dir)
    start = time.perf_counter()
    executable = solve_executable_trip_milp_with_cplex(
        executable_env,
        horizon_h=args.hours,
        economics=economics,
        storage_reward_eur_per_t=args.storage_reward_eur_per_t,
        time_limit_s=args.cplex_time_limit_s,
        mip_gap_rel=args.cplex_mip_gap_rel,
        threads=args.cplex_threads,
        msg=args.cplex_msg,
    )
    executable_solve_time_s = time.perf_counter() - start
    replay = replay_trip_milp_plan(executable_env, executable, stored_tol_t=1e-3)
    rows.append(_trip_row("executable_trip_milp", executable, executable_solve_time_s, replay))
    _log(
        (
            f"executable_trip MILP done: status={executable.status} "
            f"stored={executable.stored_t:.1f} t replay_stored={replay.stored_t:.1f} t "
            f"total_cost={executable.total_cost:.0f} replay_total_cost={replay.total_cost:.0f}"
        ),
        args.output_dir,
    )

    _write_csv(args.output_dir / "comparison.csv", rows)
    (args.output_dir / "scenario_initial_inventory.json").write_text(
        json.dumps(greedy_env.scenario.initial_inventory_t, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _log(f"wrote {args.output_dir}", args.output_dir)


if __name__ == "__main__":
    main()
