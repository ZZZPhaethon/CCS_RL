"""Compare greedy_shuttle with the full-scenario CPLEX MILP on Phase 1."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

from sim.control.baselines import greedy_shuttle_policy
from sim.control.cplex_milp import replay_full_scenario_cplex_plan, solve_full_scenario_with_cplex
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
        capture_noise_std=0.0,
        capture_outage_rate_per_week=0.0,
        enable_weather=False,
        well_maintenance_rate_per_week=0.0,
        injectivity_max_decline=0.0,
        injectivity_noise_std=0.0,
    )


def _make_env(
    hours: int,
    economics: EconomicParameters,
    fixed_scenario: str = "northern_lights_phase1",
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
    row = {
        "controller": controller,
        "solve_time_s": solve_time_s,
        **metrics.as_dict(),
    }
    return row


def _collect_policy_actions(env: CCSEnv, policy, *, seed: int, hours: int) -> list[dict[str, list[int]]]:
    env.reset(seed=seed)
    actions: list[dict[str, list[int]]] = []
    for _ in range(hours):
        action = policy(env)
        native_action = {
            "vessels": [int(choice) for choice in action["vessels"]],
            "wells": [int(choice) for choice in action["wells"]],
        }
        actions.append(native_action)
        _obs, _reward, terminated, truncated, _info = env.step(native_action)
        if terminated or truncated:
            break
    return actions


def _cplex_row(result, replay, solve_time_s: float) -> dict[str, object]:
    total_cost_per_stored = result.total_cost / result.stored_t if result.stored_t > 0.0 else math.nan
    return {
        "controller": "cplex_milp",
        "status": result.status,
        "solve_time_s": solve_time_s,
        "horizon_hours": result.horizon_h,
        "elapsed_hours": replay.elapsed_hours,
        "stored_t": result.stored_t,
        "replay_stored_t": replay.stored_t,
        "stored_gap_t": replay.stored_gap_t,
        "vented_t": result.vented_t,
        "replay_vented_t": replay.vented_t,
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
        "replay_is_executable": replay.is_executable,
        "replay_violations": ";".join(replay.violations),
        "max_binary_integrality_violation": result.max_binary_integrality_violation,
    }


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hours", type=int, default=720)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=Path("output/phase1_720h_greedy_vs_cplex_seed1"))
    parser.add_argument("--cplex-time-limit-s", type=float, default=300.0)
    parser.add_argument("--cplex-mip-gap-rel", type=float, default=0.01)
    parser.add_argument("--cplex-threads", type=int, default=None)
    parser.add_argument("--cplex-msg", action="store_true")
    parser.add_argument("--storage-shortfall-eur-per-t", type=float, default=0.0)
    parser.add_argument("--storage-reward-eur-per-t", type=float, default=1000.0)
    parser.add_argument(
        "--fixed-scenario",
        choices=available_fixed_scenario_choices(),
        default="northern_lights_phase1",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    economics = EconomicParameters(storage_shortfall_eur_per_t=args.storage_shortfall_eur_per_t)

    greedy_env = _make_env(
        args.hours,
        economics,
        args.fixed_scenario,
        storage_reward_eur_per_t=args.storage_reward_eur_per_t,
    )
    start = time.perf_counter()
    greedy_metrics = run_episode(greedy_env, greedy_shuttle_policy, seed=args.seed)
    greedy_solve_time_s = time.perf_counter() - start

    cplex_env = _make_env(
        args.hours,
        economics,
        args.fixed_scenario,
        storage_reward_eur_per_t=args.storage_reward_eur_per_t,
    )
    cplex_env.reset(seed=args.seed)
    warm_start_env = _make_env(
        args.hours,
        economics,
        args.fixed_scenario,
        storage_reward_eur_per_t=args.storage_reward_eur_per_t,
    )
    warm_start_actions = _collect_policy_actions(
        warm_start_env,
        greedy_shuttle_policy,
        seed=args.seed,
        hours=args.hours,
    )
    start = time.perf_counter()
    cplex_result = solve_full_scenario_with_cplex(
        cplex_env,
        horizon_h=args.hours,
        economics=economics,
        storage_reward_eur_per_t=args.storage_reward_eur_per_t,
        warm_start_native_actions_by_hour=warm_start_actions,
        time_limit_s=args.cplex_time_limit_s,
        mip_gap_rel=args.cplex_mip_gap_rel,
        threads=args.cplex_threads,
        msg=args.cplex_msg,
    )
    cplex_solve_time_s = time.perf_counter() - start
    cplex_replay = replay_full_scenario_cplex_plan(cplex_env, cplex_result, stored_tol_t=1e-3)

    rows = [
        _metric_row("greedy_shuttle", greedy_metrics, greedy_solve_time_s),
        _cplex_row(cplex_result, cplex_replay, cplex_solve_time_s),
    ]
    _write_csv(args.output_dir / "comparison.csv", rows)
    (args.output_dir / "scenario_initial_inventory.json").write_text(
        json.dumps(greedy_env.scenario.initial_inventory_t, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print(f"scenario={args.fixed_scenario} seed={args.seed} hours={args.hours}")
    print(f"initial_inventory_t={greedy_env.scenario.initial_inventory_t}")
    for row in rows:
        stored = float(row.get("stored_t", 0.0))
        vented = float(row.get("vented_t", 0.0))
        total_cost = float(row.get("total_cost", 0.0))
        print(
            f"{row['controller']}: stored={stored:.1f} t, vented={vented:.1f} t, "
            f"total_cost={total_cost:.0f}, solve_time={float(row['solve_time_s']):.1f}s"
        )
    print(f"wrote {args.output_dir}")


if __name__ == "__main__":
    main()
