"""Headroom sweep for probabilistic high-output and weather-window stress."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Iterator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from sim.control import trip_milp
from sim.control.baselines import greedy_shuttle_policy
from sim.control.trip_milp import replay_trip_milp_plan
from sim.economics import CostModel, EconomicParameters
from sim.environment import CCSEnv, CCSEnvConfig
from sim.metrics import EpisodeMetrics, run_episode
from sim.network_scenarios import build_fixed_scenario_demo, fixed_scenario_locations
from sim.scenario_generation import Scenario, ScenarioConfig, ScenarioGenerator

SCENARIO_ID = "northern_lights_phase1_3vessels"


@contextmanager
def pruned_executable_options(load_hours: set[int], load_start_stride_h: int) -> Iterator[None]:
    original = trip_milp._executable_trip_options

    def filtered(env, scenario, start_step: int, horizon_h: int):
        options = original(env, scenario, start_step, horizon_h)
        return [
            option
            for option in options
            if len(option.load_profile_t) in load_hours
            and option.load_start_h % load_start_stride_h == 0
        ]

    trip_milp._executable_trip_options = filtered
    try:
        yield
    finally:
        trip_milp._executable_trip_options = original


def parse_csv_floats(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def reported_total_cost(operating_cost: float, vented_t: float, carbon_price_eur_per_t: float) -> float:
    return float(operating_cost) + float(vented_t) * float(carbon_price_eur_per_t)


def make_config(
    args: argparse.Namespace,
    capture_high_output_rate_per_week: float,
    weather_rate_per_week: float,
) -> ScenarioConfig:
    return ScenarioConfig(
        episode_hours=args.hours,
        randomize_initial_inventory=True,
        capture_noise_std=args.capture_noise_std,
        capture_outage_rate_per_week=args.capture_outage_rate_per_week,
        capture_outage_mean_hours=args.capture_outage_mean_hours,
        capture_high_output_rate_per_week=capture_high_output_rate_per_week,
        capture_high_output_mean_hours=args.capture_high_output_mean_hours,
        capture_high_output_multiplier_range=(
            args.capture_multiplier_min,
            args.capture_multiplier_max,
        ),
        weather_window_rate_per_week=weather_rate_per_week,
        weather_window_mean_hours=args.weather_window_mean_hours,
        weather_window_speed_factor_range=(
            args.weather_speed_min,
            args.weather_speed_max,
        ),
        well_maintenance_rate_per_week=args.well_maintenance_rate_per_week,
        well_maintenance_mean_hours=args.well_maintenance_mean_hours,
        injectivity_max_decline=0.0,
        injectivity_noise_std=0.0,
    )


def make_env(
    args: argparse.Namespace,
    economics: EconomicParameters,
    capture_high_output_rate_per_week: float,
    weather_rate_per_week: float,
) -> CCSEnv:
    network, _state = build_fixed_scenario_demo(SCENARIO_ID)
    network.entities["yara_sluiskil"] = replace(
        network.entities["yara_sluiskil"],
        buffer_capacity_t=args.yara_buffer_t,
    )
    network.entities["oygarden_terminal"] = replace(
        network.entities["oygarden_terminal"],
        storage_capacity_t=args.terminal_buffer_t,
    )
    return CCSEnv(
        network,
        fixed_scenario_locations(SCENARIO_ID),
        scenario_generator=ScenarioGenerator(
            make_config(args, capture_high_output_rate_per_week, weather_rate_per_week)
        ),
        cost_model=CostModel(economics),
        config=CCSEnvConfig(
            episode_hours=args.hours,
            store_reward_eur_per_t=0.0,
        ),
    )


def max_run(values: list[bool]) -> int:
    best = 0
    current = 0
    for value in values:
        if value:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def scenario_stats(args: argparse.Namespace, scenario: Scenario, seed: int, rate_per_week: float) -> dict[str, object]:
    high_hours_by_emitter: dict[str, int] = {}
    outage_hours_by_emitter: dict[str, int] = {}
    high_value_sum = 0.0
    high_value_count = 0
    for emitter_id, series in scenario.emitter_availability.items():
        high = []
        outage = []
        for value in series:
            is_high = value > 1.2
            high.append(is_high)
            outage.append(value <= 1e-9)
            if is_high:
                high_value_sum += value
                high_value_count += 1
        high_hours_by_emitter[emitter_id] = sum(high)
        outage_hours_by_emitter[emitter_id] = sum(outage)

    first_speed = next(iter(scenario.vessel_speed_factor.values()))
    slow = [speed < 0.999 for speed in first_speed]
    well_down_by_well = {
        well_id: sum(not available for available in series)
        for well_id, series in scenario.well_available.items()
    }
    return {
        "capture_high_hours_total": sum(high_hours_by_emitter.values()),
        "capture_high_hours_max_emitter": max(high_hours_by_emitter.values(), default=0),
        "capture_high_multiplier_mean": (
            high_value_sum / high_value_count
            if high_value_count
            else math.nan
        ),
        "capture_outage_hours_total": sum(outage_hours_by_emitter.values()),
        "capture_outage_hours_max_emitter": max(outage_hours_by_emitter.values(), default=0),
        "weather_slow_hours": sum(slow),
        "weather_slow_max_run_h": max_run(slow),
        "weather_speed_mean": sum(first_speed) / len(first_speed) if first_speed else math.nan,
        "weather_speed_min": min(first_speed) if first_speed else math.nan,
        "well_down_hours_total": sum(well_down_by_well.values()),
        "well_down_hours_max_well": max(well_down_by_well.values(), default=0),
    }


def metric_fields(prefix: str, metrics: EpisodeMetrics) -> dict[str, object]:
    return {
        f"{prefix}_captured_t": metrics.captured_t,
        f"{prefix}_stored_t": metrics.stored_t,
        f"{prefix}_vented_t": metrics.vented_t,
        f"{prefix}_loss_rate": metrics.loss_rate,
        f"{prefix}_storage_rate": metrics.storage_rate,
        f"{prefix}_operating_cost": metrics.operating_cost,
        f"{prefix}_vent_penalty": metrics.vent_penalty,
        f"{prefix}_total_cost": metrics.total_cost,
        f"{prefix}_unit_cost": (
            metrics.total_cost / metrics.stored_t if metrics.stored_t > 0.0 else math.nan
        ),
    }


def replay_metric_fields(prefix: str, replay) -> dict[str, object]:
    return {
        f"{prefix}_stored_t": replay.stored_t,
        f"{prefix}_vented_t": replay.vented_t,
        f"{prefix}_operating_cost": replay.operating_cost,
        f"{prefix}_vent_penalty": replay.total_cost - replay.operating_cost,
        f"{prefix}_total_cost": replay.total_cost,
        f"{prefix}_unit_cost": (
            replay.total_cost / replay.stored_t if replay.stored_t > 0.0 else math.nan
        ),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def read_existing(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def collect_policy_actions(env: CCSEnv, policy, *, seed: int, hours: int) -> list[dict[str, list[int]]]:
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


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    numeric_keys = [
        "vent_penalty_eur_per_t",
        "greedy_stored_t",
        "greedy_vented_t",
        "greedy_total_cost",
        "greedy_unit_cost",
        "greedy_report_total_cost",
        "greedy_report_unit_cost",
        "optimized_stored_t",
        "optimized_vented_t",
        "optimized_total_cost",
        "optimized_unit_cost",
        "optimized_report_total_cost",
        "optimized_report_unit_cost",
        "vented_reduction_t",
        "vented_reduction_pct",
        "total_cost_saving_eur",
        "total_cost_saving_pct",
        "unit_cost_saving_eur_per_t",
        "report_total_cost_saving_eur",
        "report_unit_cost_saving_eur_per_t",
        "best_vented_reduction_t",
        "best_total_cost_saving_eur",
        "best_unit_cost_saving_eur_per_t",
        "best_report_total_cost_saving_eur",
        "best_report_unit_cost_saving_eur_per_t",
        "capture_high_hours_total",
        "capture_outage_hours_total",
        "weather_slow_hours",
        "weather_speed_mean",
        "well_down_hours_total",
        "executable_solve_time_s",
    ]
    out: list[dict[str, object]] = []
    groups = sorted({
        (
            float(row.get("vent_penalty_eur_per_t", 80.0)),
            float(row["rate_per_week"]),
            float(row.get("weather_rate_per_week", row["rate_per_week"])),
        )
        for row in rows
    })
    for vent_penalty, rate, weather_rate in groups:
        subset = [
            row
            for row in rows
            if float(row.get("vent_penalty_eur_per_t", 80.0)) == vent_penalty
            and float(row["rate_per_week"]) == rate
            and float(row.get("weather_rate_per_week", row["rate_per_week"])) == weather_rate
        ]
        summary: dict[str, object] = {
            "vent_penalty_eur_per_t": vent_penalty,
            "rate_per_week": rate,
            "capture_high_output_rate_per_week": rate,
            "weather_rate_per_week": weather_rate,
            "episodes": len(subset),
            "seeds": ",".join(str(row["seed"]) for row in subset),
            "all_replay_executable": all(str(row["replay_is_executable"]) == "True" for row in subset),
        }
        for key in numeric_keys:
            values = [float(row[key]) for row in subset if row.get(key, "") not in {"", "nan"}]
            if values:
                mean = sum(values) / len(values)
                variance = sum((value - mean) ** 2 for value in values) / len(values)
                summary[f"{key}_mean"] = mean
                summary[f"{key}_std"] = math.sqrt(variance)
        out.append(summary)
    return out


def run_case(
    args: argparse.Namespace,
    economics: EconomicParameters,
    capture_high_output_rate_per_week: float,
    weather_rate_per_week: float,
    seed: int,
) -> dict[str, object]:
    greedy_env = make_env(
        args,
        economics,
        capture_high_output_rate_per_week,
        weather_rate_per_week,
    )
    start = time.perf_counter()
    greedy = run_episode(greedy_env, greedy_shuttle_policy, seed=seed)
    greedy_s = time.perf_counter() - start

    warm_start_actions = None
    if args.warm_start_policy == "greedy":
        warm_start_env = make_env(
            args,
            economics,
            capture_high_output_rate_per_week,
            weather_rate_per_week,
        )
        warm_start_actions = collect_policy_actions(
            warm_start_env,
            greedy_shuttle_policy,
            seed=seed,
            hours=args.hours,
        )

    solve_env = make_env(
        args,
        economics,
        capture_high_output_rate_per_week,
        weather_rate_per_week,
    )
    solve_env.reset(seed=seed)
    stats = scenario_stats(args, solve_env.scenario, seed, capture_high_output_rate_per_week)
    start = time.perf_counter()
    result = trip_milp.solve_executable_trip_milp_with_cplex(
        solve_env,
        horizon_h=args.hours,
        economics=economics,
        storage_reward_eur_per_t=0.0,
        warm_start_native_actions_by_hour=warm_start_actions,
        time_limit_s=args.cplex_time_limit_s,
        mip_gap_rel=args.cplex_mip_gap_rel,
        threads=args.cplex_threads,
        msg=args.cplex_msg,
    )
    solve_s = time.perf_counter() - start
    replay = replay_trip_milp_plan(solve_env, result, stored_tol_t=1e-3)

    optimized_unit_cost = replay.total_cost / replay.stored_t if replay.stored_t > 0.0 else math.nan
    greedy_unit_cost = greedy.total_cost / greedy.stored_t if greedy.stored_t > 0.0 else math.nan
    greedy_report_total_cost = reported_total_cost(
        greedy.operating_cost,
        greedy.vented_t,
        args.report_carbon_price_eur_per_t,
    )
    optimized_report_total_cost = reported_total_cost(
        replay.operating_cost,
        replay.vented_t,
        args.report_carbon_price_eur_per_t,
    )
    greedy_report_unit_cost = (
        greedy_report_total_cost / greedy.stored_t if greedy.stored_t > 0.0 else math.nan
    )
    optimized_report_unit_cost = (
        optimized_report_total_cost / replay.stored_t if replay.stored_t > 0.0 else math.nan
    )
    vented_reduction_t = greedy.vented_t - replay.vented_t
    total_cost_saving_eur = greedy.total_cost - replay.total_cost
    use_replay = total_cost_saving_eur > 0.0
    best_stored_t = replay.stored_t if use_replay else greedy.stored_t
    best_vented_t = replay.vented_t if use_replay else greedy.vented_t
    best_total_cost = replay.total_cost if use_replay else greedy.total_cost
    best_unit_cost = best_total_cost / best_stored_t if best_stored_t > 0.0 else math.nan
    best_report_total_cost = (
        optimized_report_total_cost if use_replay else greedy_report_total_cost
    )
    best_report_unit_cost = best_report_total_cost / best_stored_t if best_stored_t > 0.0 else math.nan

    return {
        "rate_per_week": capture_high_output_rate_per_week,
        "capture_high_output_rate_per_week": capture_high_output_rate_per_week,
        "weather_rate_per_week": weather_rate_per_week,
        "seed": seed,
        "hours": args.hours,
        "objective": "vent_first_total_cost",
        "vent_penalty_eur_per_t": economics.carbon_price_eur_per_t,
        "report_carbon_price_eur_per_t": args.report_carbon_price_eur_per_t,
        "warm_start_policy": args.warm_start_policy,
        "warm_start_hours": len(warm_start_actions) if warm_start_actions is not None else 0,
        "capture_outage_rate_per_week": args.capture_outage_rate_per_week,
        "capture_outage_mean_hours": args.capture_outage_mean_hours,
        "well_maintenance_rate_per_week": args.well_maintenance_rate_per_week,
        "well_maintenance_mean_hours": args.well_maintenance_mean_hours,
        "window_mean_hours": args.window_mean_hours,
        "capture_high_output_mean_hours": args.capture_high_output_mean_hours,
        "weather_window_mean_hours": args.weather_window_mean_hours,
        "capture_multiplier_range": f"{args.capture_multiplier_min:g}-{args.capture_multiplier_max:g}",
        "weather_speed_range": f"{args.weather_speed_min:g}-{args.weather_speed_max:g}",
        "greedy_solve_time_s": greedy_s,
        "executable_solve_time_s": solve_s,
        "executable_status": result.status,
        "executable_is_valid": result.is_valid,
        "executable_validation_error": result.validation_error,
        "executable_deliveries": result.deliveries,
        "executable_binary_count": result.binary_count,
        "executable_variable_count": result.variable_count,
        "executable_constraint_count": result.constraint_count,
        "replay_is_executable": replay.is_executable,
        "replay_stored_gap_t": replay.stored_gap_t,
        "replay_violation_count": len(replay.violations),
        "replay_violations_sample": ";".join(replay.violations[:20]),
        "vented_reduction_t": vented_reduction_t,
        "vented_reduction_pct": (
            vented_reduction_t / greedy.vented_t if greedy.vented_t > 1e-9 else math.nan
        ),
        "total_cost_saving_eur": total_cost_saving_eur,
        "total_cost_saving_pct": (
            total_cost_saving_eur / greedy.total_cost if greedy.total_cost > 1e-9 else math.nan
        ),
        "unit_cost_saving_eur_per_t": greedy_unit_cost - optimized_unit_cost,
        "greedy_report_total_cost": greedy_report_total_cost,
        "optimized_report_total_cost": optimized_report_total_cost,
        "report_total_cost_saving_eur": greedy_report_total_cost - optimized_report_total_cost,
        "greedy_report_unit_cost": greedy_report_unit_cost,
        "optimized_report_unit_cost": optimized_report_unit_cost,
        "report_unit_cost_saving_eur_per_t": greedy_report_unit_cost - optimized_report_unit_cost,
        "stored_delta_t": replay.stored_t - greedy.stored_t,
        "best_policy": "optimized_replay" if use_replay else "greedy_fallback",
        "best_stored_t": best_stored_t,
        "best_vented_t": best_vented_t,
        "best_total_cost": best_total_cost,
        "best_unit_cost": best_unit_cost,
        "best_report_total_cost": best_report_total_cost,
        "best_report_unit_cost": best_report_unit_cost,
        "best_vented_reduction_t": greedy.vented_t - best_vented_t,
        "best_total_cost_saving_eur": greedy.total_cost - best_total_cost,
        "best_unit_cost_saving_eur_per_t": greedy_unit_cost - best_unit_cost,
        "best_report_total_cost_saving_eur": greedy_report_total_cost - best_report_total_cost,
        "best_report_unit_cost_saving_eur_per_t": greedy_report_unit_cost - best_report_unit_cost,
        **stats,
        **metric_fields("greedy", greedy),
        **replay_metric_fields("optimized", replay),
        "milp_stored_t": result.stored_t,
        "milp_vented_t": result.vented_t,
        "milp_total_cost": result.total_cost,
        "milp_objective": result.objective_value,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours", type=int, default=720)
    parser.add_argument("--rates", default="0.3,0.4,0.6")
    parser.add_argument("--capture-high-output-rates", default=None)
    parser.add_argument("--weather-rates", default=None)
    parser.add_argument("--seeds", default="1,2,3,4,5")
    parser.add_argument("--vent-penalties-eur-per-t", default="80")
    parser.add_argument("--report-carbon-price-eur-per-t", type=float, default=80.0)
    parser.add_argument("--warm-start-policy", choices=("none", "greedy"), default="greedy")
    parser.add_argument("--output-dir", default="output/window_stress_720h_total_cost_headroom")
    parser.add_argument("--cplex-time-limit-s", type=float, default=180.0)
    parser.add_argument("--cplex-mip-gap-rel", type=float, default=0.02)
    parser.add_argument("--cplex-threads", type=int, default=1)
    parser.add_argument("--cplex-msg", action="store_true")
    parser.add_argument("--load-hours", default="4,6,8,10")
    parser.add_argument("--load-start-stride-h", type=int, default=12)
    parser.add_argument("--window-mean-hours", type=float, default=48.0)
    parser.add_argument("--capture-high-output-mean-hours", type=float, default=None)
    parser.add_argument("--weather-window-mean-hours", type=float, default=None)
    parser.add_argument("--capture-noise-std", type=float, default=0.10)
    parser.add_argument("--capture-outage-rate-per-week", type=float, default=0.0)
    parser.add_argument("--capture-outage-mean-hours", type=float, default=12.0)
    parser.add_argument("--capture-multiplier-min", type=float, default=1.25)
    parser.add_argument("--capture-multiplier-max", type=float, default=1.75)
    parser.add_argument("--weather-speed-min", type=float, default=0.45)
    parser.add_argument("--weather-speed-max", type=float, default=0.75)
    parser.add_argument("--well-maintenance-rate-per-week", type=float, default=0.0)
    parser.add_argument("--well-maintenance-mean-hours", type=float, default=24.0)
    parser.add_argument("--yara-buffer-t", type=float, default=7500.0)
    parser.add_argument("--terminal-buffer-t", type=float, default=7500.0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.capture_high_output_mean_hours is None:
        args.capture_high_output_mean_hours = args.window_mean_hours
    if args.weather_window_mean_hours is None:
        args.weather_window_mean_hours = args.window_mean_hours

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    by_seed_path = output_dir / "by_seed.csv"
    summary_path = output_dir / "summary.csv"
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2, sort_keys=True)

    existing = read_existing(by_seed_path) if args.resume else []
    done = {
        (
            float(row.get("vent_penalty_eur_per_t", 80.0)),
            float(row["rate_per_week"]),
            float(row.get("weather_rate_per_week", row["rate_per_week"])),
            int(row["seed"]),
        )
        for row in existing
        if row.get("rate_per_week") and row.get("seed")
    }
    rows = list(existing)
    vent_penalties = parse_csv_floats(args.vent_penalties_eur_per_t)
    capture_rates = parse_csv_floats(args.capture_high_output_rates or args.rates)
    weather_rates = parse_csv_floats(args.weather_rates or args.rates)
    seeds = parse_csv_ints(args.seeds)
    load_hours = set(parse_csv_ints(args.load_hours))

    with pruned_executable_options(load_hours, args.load_start_stride_h):
        for vent_penalty in vent_penalties:
            economics = EconomicParameters(carbon_price_eur_per_t=vent_penalty)
            for capture_rate in capture_rates:
                for weather_rate in weather_rates:
                    for seed in seeds:
                        key = (vent_penalty, capture_rate, weather_rate, seed)
                        if key in done:
                            print(
                                f"skip Pvent={vent_penalty:g} capture_rate={capture_rate:g} "
                                f"weather_rate={weather_rate:g} seed={seed}",
                                flush=True,
                            )
                            continue
                        row = run_case(args, economics, capture_rate, weather_rate, seed)
                        rows.append(row)
                        write_csv(by_seed_path, rows)
                        write_csv(summary_path, summarize(rows))
                        print(
                            "done "
                            f"Pvent={vent_penalty:g} "
                            f"capture_rate={capture_rate:g} "
                            f"weather_rate={weather_rate:g} "
                            f"seed={seed} "
                            f"greedy_vent={row['greedy_vented_t']:.1f} "
                            f"opt_vent={row['optimized_vented_t']:.1f} "
                            f"vent_red={row['vented_reduction_t']:.1f} "
                            f"objective_save={row['total_cost_saving_eur']:.0f} "
                            f"solve_s={row['executable_solve_time_s']:.1f} "
                            f"status={row['executable_status']}",
                            flush=True,
                        )


if __name__ == "__main__":
    main()
