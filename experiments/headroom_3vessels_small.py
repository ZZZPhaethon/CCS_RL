"""Small-horizon headroom sweep for the 3-vessel Phase 1 scenario."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import replace
from pathlib import Path

from experiments.stress_forecast_benchmark import ForecastStressScenarioGenerator
from sim.control.baselines import greedy_shuttle_policy
from sim.control.trip_milp import (
    replay_trip_milp_plan,
    solve_executable_trip_milp_with_cplex,
    solve_relaxed_trip_milp_with_cplex,
)
from sim.economics import CostModel, EconomicParameters
from sim.environment import CCSEnv, CCSEnvConfig
from sim.metrics import run_episode
from sim.network_scenarios import build_fixed_scenario_demo, fixed_scenario_locations
from sim.scenario_generation import ScenarioConfig, ScenarioGenerator

SCENARIO_ID = "northern_lights_phase1_3vessels"


def quiet_config(hours: int, *, random_initial_inventory: bool) -> ScenarioConfig:
    return ScenarioConfig(
        episode_hours=hours,
        randomize_initial_inventory=random_initial_inventory,
        capture_noise_std=0.0,
        capture_outage_rate_per_week=0.0,
        enable_weather=False,
        well_maintenance_rate_per_week=0.0,
        injectivity_max_decline=0.0,
        injectivity_noise_std=0.0,
    )


def case_config(case: str, hours: int, *, random_initial_inventory: bool) -> ScenarioConfig:
    if case == "quiet":
        return quiet_config(hours, random_initial_inventory=random_initial_inventory)
    if case == "weather_only":
        config = quiet_config(hours, random_initial_inventory=random_initial_inventory)
        config.enable_weather = True
        return config
    if case == "capture_only":
        config = quiet_config(hours, random_initial_inventory=random_initial_inventory)
        config.capture_noise_std = 0.15
        config.capture_outage_rate_per_week = 0.8
        config.capture_outage_mean_hours = 18.0
        return config
    if case == "combined_mild":
        return ScenarioConfig(episode_hours=hours, randomize_initial_inventory=random_initial_inventory)
    if case in {"forecast_stress", "tight_forecast_stress"}:
        return ScenarioConfig(
            episode_hours=hours,
            randomize_initial_inventory=random_initial_inventory,
            injectivity_max_decline=0.0,
            injectivity_noise_std=0.0,
        )
    raise ValueError(f"Unknown case: {case}")


def make_env(
    *,
    case: str,
    hours: int,
    economics: EconomicParameters,
    storage_reward_eur_per_t: float,
    random_initial_inventory: bool,
) -> CCSEnv:
    config = case_config(case, hours, random_initial_inventory=random_initial_inventory)
    network, _state = build_fixed_scenario_demo(SCENARIO_ID)

    if case in {"forecast_stress", "tight_forecast_stress"}:
        buffer_t = 3_000.0 if case == "tight_forecast_stress" else 7_500.0
        yara = network.entities["yara_sluiskil"]
        network.entities["yara_sluiskil"] = replace(yara, buffer_capacity_t=buffer_t)
        terminal = network.entities["oygarden_terminal"]
        network.entities["oygarden_terminal"] = replace(terminal, storage_capacity_t=buffer_t)
        generator = ForecastStressScenarioGenerator(
            config,
            weather_window_count=max(1, math.ceil(hours / 168)),
            emitter_event_count=max(2, math.ceil(hours / 48)),
        )
    else:
        generator = ScenarioGenerator(config=config)

    return CCSEnv(
        network,
        fixed_scenario_locations(SCENARIO_ID),
        scenario_generator=generator,
        cost_model=CostModel(economics),
        config=CCSEnvConfig(
            episode_hours=hours,
            store_reward_eur_per_t=storage_reward_eur_per_t,
        ),
    )


def objective(total_cost: float, stored_t: float, storage_reward_eur_per_t: float) -> float:
    return float(total_cost) - storage_reward_eur_per_t * float(stored_t)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_case(args, case: str, seed: int, economics: EconomicParameters) -> dict[str, object]:
    greedy_env = make_env(
        case=case,
        hours=args.hours,
        economics=economics,
        storage_reward_eur_per_t=args.storage_reward_eur_per_t,
        random_initial_inventory=args.random_initial_inventory,
    )
    start = time.perf_counter()
    greedy = run_episode(greedy_env, greedy_shuttle_policy, seed=seed)
    greedy_s = time.perf_counter() - start
    greedy_obj = objective(greedy.total_cost, greedy.stored_t, args.storage_reward_eur_per_t)

    relaxed_env = make_env(
        case=case,
        hours=args.hours,
        economics=economics,
        storage_reward_eur_per_t=args.storage_reward_eur_per_t,
        random_initial_inventory=args.random_initial_inventory,
    )
    relaxed_env.reset(seed=seed)
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
    relaxed_s = time.perf_counter() - start

    executable = None
    replay = None
    executable_s = None
    if not args.skip_executable:
        executable_env = make_env(
            case=case,
            hours=args.hours,
            economics=economics,
            storage_reward_eur_per_t=args.storage_reward_eur_per_t,
            random_initial_inventory=args.random_initial_inventory,
        )
        executable_env.reset(seed=seed)
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
        executable_s = time.perf_counter() - start
        replay = replay_trip_milp_plan(executable_env, executable, stored_tol_t=1e-3)

    row: dict[str, object] = {
        "case": case,
        "seed": seed,
        "hours": args.hours,
        "greedy_solve_time_s": greedy_s,
        "greedy_stored_t": greedy.stored_t,
        "greedy_vented_t": greedy.vented_t,
        "greedy_operating_cost": greedy.operating_cost,
        "greedy_total_cost": greedy.total_cost,
        "greedy_objective": greedy_obj,
        "relaxed_status": relaxed.status,
        "relaxed_is_valid": relaxed.is_valid,
        "relaxed_solve_time_s": relaxed_s,
        "relaxed_stored_t": relaxed.stored_t,
        "relaxed_vented_t": relaxed.vented_t,
        "relaxed_operating_cost": relaxed.operating_cost,
        "relaxed_total_cost": relaxed.total_cost,
        "relaxed_objective": relaxed.objective_value,
        "relaxed_binary_count": relaxed.binary_count,
        "relaxed_variable_count": relaxed.variable_count,
        "relaxed_constraint_count": relaxed.constraint_count,
        "relaxed_objective_headroom_eur": greedy_obj - relaxed.objective_value,
        "relaxed_objective_headroom_pct_abs_greedy": (greedy_obj - relaxed.objective_value)
        / abs(greedy_obj)
        if abs(greedy_obj) > 1e-9
        else math.nan,
        "relaxed_stored_delta_t": relaxed.stored_t - greedy.stored_t,
        "relaxed_vented_delta_t": relaxed.vented_t - greedy.vented_t,
        "relaxed_total_cost_delta_eur": relaxed.total_cost - greedy.total_cost,
    }
    if executable is not None and replay is not None and executable_s is not None:
        executable_obj = executable.objective_value
        replay_obj = objective(replay.total_cost, replay.stored_t, args.storage_reward_eur_per_t)
        row.update(
            {
                "executable_status": executable.status,
                "executable_is_valid": executable.is_valid,
                "executable_validation_error": executable.validation_error,
                "executable_solve_time_s": executable_s,
                "executable_stored_t": executable.stored_t,
                "executable_vented_t": executable.vented_t,
                "executable_operating_cost": executable.operating_cost,
                "executable_total_cost": executable.total_cost,
                "executable_objective": executable_obj,
                "executable_objective_headroom_eur": greedy_obj - executable_obj,
                "executable_objective_headroom_pct_abs_greedy": (greedy_obj - executable_obj)
                / abs(greedy_obj)
                if abs(greedy_obj) > 1e-9
                else math.nan,
                "executable_stored_delta_t": executable.stored_t - greedy.stored_t,
                "executable_vented_delta_t": executable.vented_t - greedy.vented_t,
                "executable_total_cost_delta_eur": executable.total_cost - greedy.total_cost,
                "replay_is_executable": replay.is_executable,
                "replay_violations": ";".join(replay.violations),
                "replay_stored_t": replay.stored_t,
                "replay_vented_t": replay.vented_t,
                "replay_total_cost": replay.total_cost,
                "replay_objective": replay_obj,
                "replay_objective_headroom_eur": greedy_obj - replay_obj,
                "replay_objective_headroom_pct_abs_greedy": (greedy_obj - replay_obj)
                / abs(greedy_obj)
                if abs(greedy_obj) > 1e-9
                else math.nan,
            }
        )
    return row


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for case in sorted({str(row["case"]) for row in rows}):
        subset = [row for row in rows if row["case"] == case]
        summary: dict[str, object] = {"case": case, "episodes": len(subset)}
        for key in [
            "greedy_stored_t",
            "greedy_vented_t",
            "greedy_total_cost",
            "greedy_objective",
            "relaxed_stored_delta_t",
            "relaxed_vented_delta_t",
            "relaxed_objective_headroom_eur",
            "relaxed_objective_headroom_pct_abs_greedy",
            "executable_stored_delta_t",
            "executable_vented_delta_t",
            "executable_objective_headroom_eur",
            "executable_objective_headroom_pct_abs_greedy",
            "replay_objective_headroom_eur",
            "replay_objective_headroom_pct_abs_greedy",
        ]:
            values = [float(row[key]) for row in subset if key in row and row[key] not in ("", None)]
            summary[f"{key}_mean"] = sum(values) / len(values) if values else ""
        out.append(summary)
    return out


def write_report(path: Path, rows: list[dict[str, object]], summary: list[dict[str, object]]) -> None:
    lines = [
        "# 3-Vessel Small-Horizon Headroom",
        "",
        "Objective is `total_cost - storage_reward_eur_per_t * stored_t`; lower is better.",
        "Relaxed trip MILP is an optimistic lower-bound-style oracle; executable/replay rows are feasible-policy checks.",
        "",
        "## Summary",
        "",
        "| Case | Greedy stored t | Greedy vented t | Relaxed headroom | Relaxed stored delta | Executable headroom | Replay headroom |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['case']} | "
            f"{float(row.get('greedy_stored_t_mean') or 0.0):,.1f} | "
            f"{float(row.get('greedy_vented_t_mean') or 0.0):,.1f} | "
            f"{float(row.get('relaxed_objective_headroom_pct_abs_greedy_mean') or 0.0):.2%} | "
            f"{float(row.get('relaxed_stored_delta_t_mean') or 0.0):,.1f} | "
            f"{float(row.get('executable_objective_headroom_pct_abs_greedy_mean') or 0.0):.2%} | "
            f"{float(row.get('replay_objective_headroom_pct_abs_greedy_mean') or 0.0):.2%} |"
        )
    lines.extend(["", "## Per Case Details", ""])
    for row in rows:
        lines.append(
            f"- {row['case']} seed={row['seed']}: "
            f"greedy stored={float(row['greedy_stored_t']):,.1f} t, "
            f"vented={float(row['greedy_vented_t']):,.1f} t; "
            f"relaxed status={row['relaxed_status']}, "
            f"headroom={float(row['relaxed_objective_headroom_pct_abs_greedy']):.2%}; "
            f"executable status={row.get('executable_status', 'skipped')}, "
            f"replay headroom={float(row.get('replay_objective_headroom_pct_abs_greedy') or 0.0):.2%}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hours", type=int, default=72)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1])
    parser.add_argument(
        "--cases",
        nargs="+",
        default=[
            "quiet",
            "weather_only",
            "capture_only",
            "combined_mild",
            "forecast_stress",
            "tight_forecast_stress",
        ],
    )
    parser.add_argument("--output-dir", type=Path, default=Path("output/headroom_3vessels_small"))
    parser.add_argument("--storage-reward-eur-per-t", type=float, default=1_000.0)
    parser.add_argument("--carbon-price-eur-per-t", type=float, default=80.0)
    parser.add_argument("--cplex-time-limit-s", type=float, default=45.0)
    parser.add_argument("--cplex-mip-gap-rel", type=float, default=0.01)
    parser.add_argument("--cplex-threads", type=int, default=None)
    parser.add_argument("--cplex-msg", action="store_true")
    parser.add_argument("--skip-executable", action="store_true")
    parser.add_argument("--random-initial-inventory", action="store_true", default=True)
    parser.add_argument("--fixed-initial-inventory", dest="random_initial_inventory", action="store_false")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    economics = EconomicParameters(carbon_price_eur_per_t=args.carbon_price_eur_per_t)
    rows: list[dict[str, object]] = []
    start = time.perf_counter()
    for case in args.cases:
        for seed in args.seeds:
            case_start = time.perf_counter()
            row = run_case(args, case, seed, economics)
            rows.append(row)
            write_csv(args.output_dir / "headroom_by_case.partial.csv", rows)
            print(
                f"{case} seed={seed}: greedy_stored={float(row['greedy_stored_t']):.1f} "
                f"relaxed_headroom={float(row['relaxed_objective_headroom_pct_abs_greedy']):.2%} "
                f"replay_headroom={float(row.get('replay_objective_headroom_pct_abs_greedy') or 0.0):.2%} "
                f"time={time.perf_counter() - case_start:.1f}s",
                flush=True,
            )
    summary = summarize(rows)
    write_csv(args.output_dir / "headroom_by_case.csv", rows)
    write_csv(args.output_dir / "headroom_summary.csv", summary)
    write_report(args.output_dir / "headroom_report.md", rows, summary)
    with (args.output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "scenario": SCENARIO_ID,
                "hours": args.hours,
                "seeds": args.seeds,
                "cases": args.cases,
                "storage_reward_eur_per_t": args.storage_reward_eur_per_t,
                "carbon_price_eur_per_t": args.carbon_price_eur_per_t,
                "cplex_time_limit_s": args.cplex_time_limit_s,
                "cplex_mip_gap_rel": args.cplex_mip_gap_rel,
                "skip_executable": args.skip_executable,
                "elapsed_s": time.perf_counter() - start,
            },
            handle,
            indent=2,
            sort_keys=True,
        )
    print(f"wrote {args.output_dir}")


if __name__ == "__main__":
    main()
