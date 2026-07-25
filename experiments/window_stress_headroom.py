"""Headroom sweep for probabilistic high-output and weather-window stress."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from sim.control import cplex_milp
from sim.control.baselines import greedy_shuttle_policy
from sim.economics import CostModel, EconomicParameters
from sim.environment import CCSEnv, CCSEnvConfig
from sim.metrics import EpisodeMetrics, run_episode
from sim.network_scenarios import build_fixed_scenario_demo, fixed_scenario_locations
from sim.scenario_generation import Scenario, ScenarioConfig, ScenarioGenerator

SCENARIO_ID = "northern_lights_phase1_3vessels"


def parse_csv_floats(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def apply_scenario_config_defaults(args: argparse.Namespace) -> None:
    defaults = ScenarioConfig()
    args.capture_noise_std = defaults.capture_noise_std
    args.capture_outage_rate_per_week = defaults.capture_outage_rate_per_week
    args.capture_outage_mean_hours = defaults.capture_outage_mean_hours
    args.capture_high_output_rates = str(defaults.capture_high_output_rate_per_week)
    args.capture_high_output_mean_hours = defaults.capture_high_output_mean_hours
    args.capture_multiplier_min, args.capture_multiplier_max = defaults.capture_high_output_multiplier_range
    args.weather_window_mode = "hourly"
    args.weather_rates = str(defaults.weather_window_rate_per_week)
    args.weather_window_mean_hours = defaults.weather_window_mean_hours
    args.weather_speed_min, args.weather_speed_max = defaults.weather_window_speed_factor_range
    args.well_maintenance_rate_per_week = defaults.well_maintenance_rate_per_week
    args.well_maintenance_mean_hours = defaults.well_maintenance_mean_hours


def reported_total_cost(operating_cost: float, vented_t: float, carbon_price_eur_per_t: float) -> float:
    return float(operating_cost) + float(vented_t) * float(carbon_price_eur_per_t)


class WeeklyWeatherScenarioGenerator(ScenarioGenerator):
    """Apply at most one probabilistic weather window per full week."""

    def __init__(
        self,
        config: ScenarioConfig,
        *,
        weekly_probability: float,
        duration_min_h: int,
        duration_max_h: int,
        speed_factor_range: tuple[float, float],
    ) -> None:
        super().__init__(config=config)
        self.weekly_probability = max(0.0, min(1.0, float(weekly_probability)))
        self.duration_min_h = max(1, int(duration_min_h))
        self.duration_max_h = max(self.duration_min_h, int(duration_max_h))
        self.speed_factor_range = speed_factor_range

    def sample(self, network, seed: int | None = None) -> Scenario:
        scenario = super().sample(network, seed=seed)
        if self.weekly_probability <= 0.0:
            return scenario

        rng = random.Random(f"weekly-weather:{seed}")
        full_weeks = max(1, int((scenario.n_steps * scenario.time_step_hours) // 168))
        lo, hi = self.speed_factor_range
        for week_index in range(full_weeks):
            if rng.random() > self.weekly_probability:
                continue
            week_start = int(round(week_index * 168 / scenario.time_step_hours))
            if week_start >= scenario.n_steps:
                continue
            latest_offset = min(120, max(0, scenario.n_steps - week_start - 1))
            start = week_start + rng.randint(0, latest_offset)
            duration_h = rng.randint(self.duration_min_h, self.duration_max_h)
            duration_steps = max(1, int(round(duration_h / scenario.time_step_hours)))
            end = min(scenario.n_steps, start + duration_steps)
            factor = rng.uniform(lo, hi)
            for series in scenario.vessel_speed_factor.values():
                for t in range(start, end):
                    series[t] = min(series[t], factor)
        return scenario


def make_config(
    args: argparse.Namespace,
    capture_high_output_rate_per_week: float,
    weather_rate_per_week: float,
) -> ScenarioConfig:
    weather_rate = 0.0 if args.weather_window_mode == "weekly" else weather_rate_per_week
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
        weather_window_rate_per_week=weather_rate,
        weather_window_mean_hours=args.weather_window_mean_hours,
        weather_window_speed_factor_range=(
            args.weather_speed_min,
            args.weather_speed_max,
        ),
        well_maintenance_rate_per_week=args.well_maintenance_rate_per_week,
        well_maintenance_mean_hours=args.well_maintenance_mean_hours,
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
    config = make_config(args, capture_high_output_rate_per_week, weather_rate_per_week)
    if args.weather_window_mode == "weekly":
        weekly_probability = (
            args.weather_weekly_probability
            if args.weather_weekly_probability is not None
            else 1.0
        )
        duration_min_h = (
            args.weather_weekly_duration_min_hours
            if args.weather_weekly_duration_min_hours is not None
            else int(round(args.weather_window_mean_hours * 0.75))
        )
        duration_max_h = (
            args.weather_weekly_duration_max_hours
            if args.weather_weekly_duration_max_hours is not None
            else int(round(args.weather_window_mean_hours * 1.25))
        )
        scenario_generator = WeeklyWeatherScenarioGenerator(
            config,
            weekly_probability=weekly_probability,
            duration_min_h=duration_min_h,
            duration_max_h=duration_max_h,
            speed_factor_range=(args.weather_speed_min, args.weather_speed_max),
        )
    else:
        scenario_generator = ScenarioGenerator(config)

    return CCSEnv(
        network,
        fixed_scenario_locations(SCENARIO_ID),
        scenario_generator=scenario_generator,
        cost_model=CostModel(economics),
        config=CCSEnvConfig(
            episode_hours=args.hours,
            store_reward_eur_per_t=args.storage_reward_eur_per_t,
            reward_mode=args.reward_mode,
            vent_first_vent_eur_per_t=args.vent_first_vent_eur_per_t,
            overflow_risk_eur_per_t=args.overflow_risk_eur_per_t,
            overflow_risk_lookahead_h=args.overflow_risk_lookahead_h,
            operating_cost_weight=args.operating_cost_weight,
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
    exact_only_keys = {
        key
        for key in numeric_keys
        if key.startswith(("optimized_", "best_"))
        or "saving" in key
        or "reduction" in key
    }
    out: list[dict[str, object]] = []
    groups = sorted({
        (
            float(row.get("vent_penalty_eur_per_t", 80.0)),
            str(row.get("weather_window_mode", "hourly")),
            float(row["rate_per_week"]),
            float(row.get("weather_rate_per_week", row["rate_per_week"])),
        )
        for row in rows
    })
    for vent_penalty, weather_window_mode, rate, weather_rate in groups:
        subset = [
            row
            for row in rows
            if float(row.get("vent_penalty_eur_per_t", 80.0)) == vent_penalty
            and str(row.get("weather_window_mode", "hourly")) == weather_window_mode
            and float(row["rate_per_week"]) == rate
            and float(row.get("weather_rate_per_week", row["rate_per_week"])) == weather_rate
        ]
        summary: dict[str, object] = {
            "vent_penalty_eur_per_t": vent_penalty,
            "weather_window_mode": weather_window_mode,
            "rate_per_week": rate,
            "capture_high_output_rate_per_week": rate,
            "weather_rate_per_week": weather_rate,
            "episodes": len(subset),
            "seeds": ",".join(str(row["seed"]) for row in subset),
            "all_replay_executable": all(str(row["replay_is_executable"]) == "True" for row in subset),
            "all_replay_exact": all(
                str(row.get("replay_is_exact", "False")) == "True"
                for row in subset
            ),
        }
        for key in numeric_keys:
            value_rows = subset
            if key in exact_only_keys:
                value_rows = [
                    row
                    for row in subset
                    if str(row.get("replay_is_exact", "False")) == "True"
                ]
            values = [
                float(row[key])
                for row in value_rows
                if row.get(key, "") not in {"", "nan"}
            ]
            if values:
                mean = sum(values) / len(values)
                variance = sum((value - mean) ** 2 for value in values) / len(values)
                summary[f"{key}_mean"] = mean
                summary[f"{key}_std"] = math.sqrt(variance)
        out.append(summary)
    return out


def solve_oracle(
    args: argparse.Namespace,
    env: CCSEnv,
    economics: EconomicParameters,
    warm_start_actions: list[dict[str, list[int]]] | None,
):
    kwargs = {
        "horizon_h": args.hours,
        "economics": economics,
        "storage_reward_eur_per_t": args.storage_reward_eur_per_t,
        "warm_start_native_actions_by_hour": warm_start_actions,
        "cplex_path": getattr(args, "cplex_path", None),
        "time_limit_s": args.cplex_time_limit_s,
        "mip_gap_rel": args.cplex_mip_gap_rel,
        "threads": args.cplex_threads,
        "msg": args.cplex_msg,
    }
    return cplex_milp.solve_full_scenario_with_cplex(env, **kwargs)


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
    result = solve_oracle(args, solve_env, economics, warm_start_actions)
    solve_s = time.perf_counter() - start
    replay = cplex_milp.replay_full_scenario_cplex_plan(solve_env, result, stored_tol_t=1e-3)

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
    greedy_objective = -greedy.total_reward / float(greedy_env.config.reward_scale)
    optimized_objective = replay.objective_value
    use_replay = replay.is_exact and optimized_objective < greedy_objective
    best_stored_t = replay.stored_t if use_replay else greedy.stored_t
    best_vented_t = replay.vented_t if use_replay else greedy.vented_t
    best_total_cost = replay.total_cost if use_replay else greedy.total_cost
    best_unit_cost = best_total_cost / best_stored_t if best_stored_t > 0.0 else math.nan
    best_report_total_cost = (
        optimized_report_total_cost if use_replay else greedy_report_total_cost
    )
    best_report_unit_cost = best_report_total_cost / best_stored_t if best_stored_t > 0.0 else math.nan
    optimized_metric_values = replay_metric_fields("optimized", replay)
    if not replay.is_exact:
        optimized_metric_values = {key: "" for key in optimized_metric_values}

    def exact_value(value):
        return value if replay.is_exact else ""

    return {
        "rate_per_week": capture_high_output_rate_per_week,
        "capture_high_output_rate_per_week": capture_high_output_rate_per_week,
        "weather_rate_per_week": weather_rate_per_week,
        "weather_window_mode": args.weather_window_mode,
        "weather_weekly_probability": (
            args.weather_weekly_probability
            if args.weather_weekly_probability is not None
            else 1.0
        )
        if args.weather_window_mode == "weekly"
        else "",
        "seed": seed,
        "hours": args.hours,
        "oracle_model": "native_action",
        "objective": args.reward_mode,
        "greedy_objective": greedy_objective,
        "optimized_objective": exact_value(optimized_objective),
        "objective_headroom": exact_value(greedy_objective - optimized_objective),
        "vent_first_vent_eur_per_t": args.vent_first_vent_eur_per_t,
        "overflow_risk_eur_per_t": args.overflow_risk_eur_per_t,
        "overflow_risk_lookahead_h": args.overflow_risk_lookahead_h,
        "operating_cost_weight": args.operating_cost_weight,
        "vent_penalty_eur_per_t": economics.carbon_price_eur_per_t,
        "report_carbon_price_eur_per_t": args.report_carbon_price_eur_per_t,
        "storage_reward_eur_per_t": args.storage_reward_eur_per_t,
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
        "executable_binary_count": getattr(result, "binary_count", ""),
        "executable_variable_count": getattr(result, "variable_count", ""),
        "executable_constraint_count": getattr(result, "constraint_count", ""),
        "replay_is_executable": replay.is_executable,
        "replay_is_exact": replay.is_exact,
        "replay_mismatch_count": len(replay.mismatches),
        "replay_mismatches_sample": ";".join(replay.mismatches[:20]),
        "replay_compared_fields": ";".join(sorted(replay.compared_fields)),
        "replay_stored_gap_t": replay.stored_gap_t,
        "replay_violation_count": len(replay.violations),
        "replay_violations_sample": ";".join(replay.violations[:20]),
        "vented_reduction_t": exact_value(vented_reduction_t),
        "vented_reduction_pct": exact_value(
            vented_reduction_t / greedy.vented_t if greedy.vented_t > 1e-9 else math.nan
        ),
        "total_cost_saving_eur": exact_value(total_cost_saving_eur),
        "total_cost_saving_pct": exact_value(
            total_cost_saving_eur / greedy.total_cost if greedy.total_cost > 1e-9 else math.nan
        ),
        "unit_cost_saving_eur_per_t": exact_value(greedy_unit_cost - optimized_unit_cost),
        "greedy_report_total_cost": greedy_report_total_cost,
        "optimized_report_total_cost": exact_value(optimized_report_total_cost),
        "report_total_cost_saving_eur": exact_value(
            greedy_report_total_cost - optimized_report_total_cost
        ),
        "greedy_report_unit_cost": greedy_report_unit_cost,
        "optimized_report_unit_cost": exact_value(optimized_report_unit_cost),
        "report_unit_cost_saving_eur_per_t": exact_value(
            greedy_report_unit_cost - optimized_report_unit_cost
        ),
        "stored_delta_t": exact_value(replay.stored_t - greedy.stored_t),
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
        **optimized_metric_values,
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
    parser.add_argument("--storage-reward-eur-per-t", type=float, default=0.0)
    parser.add_argument("--reward-mode", choices=("economic", "vent_first"), default="vent_first")
    parser.add_argument("--vent-first-vent-eur-per-t", type=float, default=10_000.0)
    parser.add_argument("--overflow-risk-eur-per-t", type=float, default=100.0)
    parser.add_argument("--overflow-risk-lookahead-h", type=float, default=24.0)
    parser.add_argument("--operating-cost-weight", type=float, default=1.0)
    parser.add_argument("--warm-start-policy", choices=("none", "greedy"), default="greedy")
    parser.add_argument("--scenario-config-defaults", action="store_true")
    parser.add_argument("--output-dir", default="output/window_stress_720h_total_cost_headroom")
    parser.add_argument("--cplex-time-limit-s", type=float, default=180.0)
    parser.add_argument("--cplex-path", default=None)
    parser.add_argument("--cplex-mip-gap-rel", type=float, default=0.02)
    parser.add_argument("--cplex-threads", type=int, default=1)
    parser.add_argument("--cplex-msg", action="store_true")
    parser.add_argument("--window-mean-hours", type=float, default=48.0)
    parser.add_argument("--capture-high-output-mean-hours", type=float, default=None)
    parser.add_argument("--weather-window-mean-hours", type=float, default=None)
    parser.add_argument("--weather-window-mode", choices=("hourly", "weekly"), default="hourly")
    parser.add_argument("--weather-weekly-probability", type=float, default=1.0)
    parser.add_argument("--weather-weekly-duration-min-hours", type=int, default=None)
    parser.add_argument("--weather-weekly-duration-max-hours", type=int, default=None)
    parser.add_argument("--capture-noise-std", type=float, default=0.10)
    parser.add_argument("--capture-outage-rate-per-week", type=float, default=0.0)
    parser.add_argument("--capture-outage-mean-hours", type=float, default=12.0)
    parser.add_argument("--capture-multiplier-min", type=float, default=1.25)
    parser.add_argument("--capture-multiplier-max", type=float, default=1.75)
    parser.add_argument("--weather-speed-min", type=float, default=0.6)
    parser.add_argument("--weather-speed-max", type=float, default=0.8)
    parser.add_argument("--well-maintenance-rate-per-week", type=float, default=0.0)
    parser.add_argument("--well-maintenance-mean-hours", type=float, default=24.0)
    parser.add_argument("--yara-buffer-t", type=float, default=7500.0)
    parser.add_argument("--terminal-buffer-t", type=float, default=7500.0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.scenario_config_defaults:
        apply_scenario_config_defaults(args)
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
            str(row.get("weather_window_mode", "hourly")),
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
    for vent_penalty in vent_penalties:
        economics = EconomicParameters(carbon_price_eur_per_t=vent_penalty)
        for capture_rate in capture_rates:
            for weather_rate in weather_rates:
                for seed in seeds:
                    key = (vent_penalty, args.weather_window_mode, capture_rate, weather_rate, seed)
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
                    if row["replay_is_exact"]:
                        message = (
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
                            f"status={row['executable_status']} replay_exact=True"
                        )
                    else:
                        message = (
                            "done "
                            f"Pvent={vent_penalty:g} "
                            f"capture_rate={capture_rate:g} "
                            f"weather_rate={weather_rate:g} "
                            f"seed={seed} "
                            f"solve_s={row['executable_solve_time_s']:.1f} "
                            f"status={row['executable_status']} replay_exact=False "
                            f"mismatches={row['replay_mismatches_sample']}"
                        )
                    print(message, flush=True)


if __name__ == "__main__":
    main()
