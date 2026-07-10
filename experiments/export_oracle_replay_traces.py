"""Export executable trip traces and validate them through a native replay policy."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, replace
from pathlib import Path
from typing import Iterator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.stress_forecast_benchmark import ForecastStressScenarioGenerator
from sim.control import trip_milp
from sim.control.baselines import greedy_shuttle_policy
from sim.economics import CostModel, EconomicParameters
from sim.environment import CCSEnv, CCSEnvConfig, OFF_WELL_RATE_INDEX, VESSEL_WAIT
from sim.metrics import EpisodeMetrics, run_episode
from sim.network_scenarios import build_fixed_scenario_demo, fixed_scenario_locations
from sim.scenario_generation import Scenario, ScenarioConfig


SCENARIO_ID = "northern_lights_phase1_3vessels"
DEFAULT_SOURCE_DIR = Path(
    "output/phase1_3vessels_720h_weather_floor07_greedy_vs_executable_pruned12h_seeds1-5_300s"
)


class FlooredForecastStressScenarioGenerator(ForecastStressScenarioGenerator):
    def __init__(self, config: ScenarioConfig, *, speed_floor: float) -> None:
        super().__init__(config)
        self.speed_floor = float(speed_floor)

    def sample(self, network, seed: int | None = None) -> Scenario:
        scenario = super().sample(network, seed=seed)
        for series in scenario.vessel_speed_factor.values():
            for index, value in enumerate(series):
                series[index] = max(float(value), self.speed_floor)
        self.last_events.append(
            {"type": "weather_speed_floor", "speed_factor_floor": self.speed_floor}
        )
        return scenario


class NativeActionTracePolicy:
    def __init__(self, native_actions_by_hour: list[dict[str, list[int]]]) -> None:
        self.native_actions_by_hour = native_actions_by_hour
        self.mask_violations: list[str] = []

    def __call__(self, env: CCSEnv) -> dict[str, list[int]]:
        t = int(env.t)
        if t < len(self.native_actions_by_hour):
            action = self.native_actions_by_hour[t]
        else:
            action = {
                "vessels": [VESSEL_WAIT] * len(env.vessel_ids),
                "wells": [OFF_WELL_RATE_INDEX] * len(env.well_ids),
            }
        self._check_masks(env, t, action)
        return {
            "vessels": [int(value) for value in action["vessels"]],
            "wells": [int(value) for value in action["wells"]],
        }

    def _check_masks(self, env: CCSEnv, t: int, action: dict[str, list[int]]) -> None:
        for index, choice in enumerate(action["vessels"]):
            mask = env.vessel_action_mask()[index]
            if int(choice) < 0 or int(choice) >= len(mask) or not mask[int(choice)]:
                self.mask_violations.append(f"h{t}:vessel:{env.vessel_ids[index]}:{choice}")
        for index, choice in enumerate(action["wells"]):
            mask = env.well_rate_action_mask()[index]
            if int(choice) < 0 or int(choice) >= len(mask) or not mask[int(choice)]:
                self.mask_violations.append(f"h{t}:well:{env.well_ids[index]}:{choice}")


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


def make_env(args: argparse.Namespace, economics: EconomicParameters) -> CCSEnv:
    network, _state = build_fixed_scenario_demo(SCENARIO_ID)
    network.entities["yara_sluiskil"] = replace(
        network.entities["yara_sluiskil"],
        buffer_capacity_t=args.yara_buffer_t,
    )
    network.entities["oygarden_terminal"] = replace(
        network.entities["oygarden_terminal"],
        storage_capacity_t=args.terminal_buffer_t,
    )
    config = ScenarioConfig(
        episode_hours=args.hours,
        randomize_initial_inventory=True,
    )
    return CCSEnv(
        network,
        fixed_scenario_locations(SCENARIO_ID),
        scenario_generator=FlooredForecastStressScenarioGenerator(
            config,
            speed_floor=args.speed_floor,
        ),
        cost_model=CostModel(economics),
        config=CCSEnvConfig(
            episode_hours=args.hours,
            store_reward_eur_per_t=args.storage_reward_eur_per_t,
        ),
    )


def objective(total_cost: float, stored_t: float, storage_reward_eur_per_t: float) -> float:
    return float(total_cost) - float(storage_reward_eur_per_t) * float(stored_t)


def load_reference_rows(path: Path) -> dict[tuple[int, str], dict[str, str]]:
    rows: dict[tuple[int, str], dict[str, str]] = {}
    if not path.exists():
        return rows
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows[(int(row["seed"]), row["controller"])] = row
    return rows


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


def read_existing_csv(path: Path, replacing_seeds: set[int]) -> list[dict[str, object]]:
    if not path.exists():
        return []
    rows: list[dict[str, object]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if "seed" in row and int(row["seed"]) in replacing_seeds:
                continue
            rows.append(dict(row))
    return rows


def metric_fields(metrics: EpisodeMetrics) -> dict[str, float | int | None]:
    data = metrics.as_dict()
    return {
        "captured_t": data["captured_t"],
        "stored_t": data["stored_t"],
        "vented_t": data["vented_t"],
        "in_transit_t": data["in_transit_t"],
        "in_transit_growth_t": data["in_transit_growth_t"],
        "loss_rate": data["loss_rate"],
        "storage_rate": data["storage_rate"],
        "operating_cost": data["operating_cost"],
        "vent_penalty": data["vent_penalty"],
        "total_cost_ex_shortfall": data["total_cost"],
        "unit_cost_per_stored_t": data["total_cost_per_stored_t"],
    }


def replay_policy_metrics(
    args: argparse.Namespace,
    economics: EconomicParameters,
    seed: int,
    native_actions_by_hour: list[dict[str, list[int]]],
) -> tuple[EpisodeMetrics, list[str], float]:
    env = make_env(args, economics)
    policy = NativeActionTracePolicy(native_actions_by_hour)
    start = time.perf_counter()
    metrics = run_episode(env, policy, seed=seed)
    elapsed_s = time.perf_counter() - start
    return metrics, policy.mask_violations, elapsed_s


def collect_policy_actions(
    args: argparse.Namespace,
    economics: EconomicParameters,
    seed: int,
    policy_fn,
) -> list[dict[str, list[int]]]:
    env = make_env(args, economics)
    env.reset(seed=seed)
    actions: list[dict[str, list[int]]] = []
    for _ in range(args.hours):
        action = policy_fn(env)
        native_action = {
            "vessels": [int(value) for value in action["vessels"]],
            "wells": [int(value) for value in action["wells"]],
        }
        actions.append(native_action)
        _obs, _reward, terminated, truncated, _info = env.step(native_action)
        if terminated or truncated:
            break
    return actions


def load_source_native_actions(args: argparse.Namespace, seed: int) -> list[dict[str, list[int]]] | None:
    path = args.source_dir / f"seed_{seed}" / "native_actions_by_hour.json"
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def replay_diff_row(
    *,
    seed: int,
    policy_metrics: EpisodeMetrics,
    direct_replay,
    reference_replay: dict[str, str] | None,
    storage_reward_eur_per_t: float,
    mask_violations: list[str],
) -> dict[str, object]:
    policy_obj = objective(
        policy_metrics.total_cost,
        policy_metrics.stored_t,
        storage_reward_eur_per_t,
    )
    row: dict[str, object] = {
        "seed": seed,
        "policy_replay_stored_t": policy_metrics.stored_t,
        "direct_replay_stored_t": direct_replay.stored_t,
        "stored_gap_policy_minus_direct_t": policy_metrics.stored_t - direct_replay.stored_t,
        "policy_replay_vented_t": policy_metrics.vented_t,
        "direct_replay_vented_t": direct_replay.vented_t,
        "vented_gap_policy_minus_direct_t": policy_metrics.vented_t - direct_replay.vented_t,
        "policy_replay_total_cost": policy_metrics.total_cost,
        "direct_replay_total_cost": direct_replay.total_cost,
        "total_cost_gap_policy_minus_direct": policy_metrics.total_cost - direct_replay.total_cost,
        "policy_replay_objective": policy_obj,
        "direct_replay_is_executable": direct_replay.is_executable,
        "direct_replay_stored_gap_t": direct_replay.stored_gap_t,
        "direct_replay_violations": ";".join(direct_replay.violations),
        "action_mask_violation_count": len(mask_violations),
        "action_mask_violations": ";".join(mask_violations[:20]),
    }
    if reference_replay is not None:
        row.update(
            {
                "reference_replay_stored_t": float(reference_replay["stored_t"]),
                "stored_gap_policy_minus_reference_t": policy_metrics.stored_t
                - float(reference_replay["stored_t"]),
                "reference_replay_vented_t": float(reference_replay["vented_t"]),
                "vented_gap_policy_minus_reference_t": policy_metrics.vented_t
                - float(reference_replay["vented_t"]),
                "reference_replay_total_cost": float(reference_replay["total_cost_ex_shortfall"]),
                "total_cost_gap_policy_minus_reference": policy_metrics.total_cost
                - float(reference_replay["total_cost_ex_shortfall"]),
            }
        )
    return row


def run_seed(
    args: argparse.Namespace,
    economics: EconomicParameters,
    seed: int,
    reference_rows: dict[tuple[int, str], dict[str, str]],
) -> tuple[
    list[dict[str, object]],
    dict[str, object],
    dict[str, object],
    list[dict[str, list[int]]],
    dict[str, object],
]:
    greedy_env = make_env(args, economics)
    greedy_start = time.perf_counter()
    greedy_metrics = run_episode(greedy_env, greedy_shuttle_policy, seed=seed)
    greedy_s = time.perf_counter() - greedy_start

    solve_env = make_env(args, economics)
    solve_env.reset(seed=seed)
    warm_start_actions = None
    if args.oracle_level == "executable":
        if args.warm_start_policy == "greedy":
            warm_start_actions = collect_policy_actions(
                args,
                economics,
                seed,
                greedy_shuttle_policy,
            )
        elif args.warm_start_policy == "source-native":
            warm_start_actions = load_source_native_actions(args, seed)
    solve_start = time.perf_counter()
    if args.oracle_level == "relaxed-materialized":
        relaxed = trip_milp.solve_relaxed_trip_milp_with_cplex(
            solve_env,
            horizon_h=args.hours,
            economics=economics,
            storage_reward_eur_per_t=args.storage_reward_eur_per_t,
            time_limit_s=args.cplex_time_limit_s,
            mip_gap_rel=args.cplex_mip_gap_rel,
            threads=args.cplex_threads,
            msg=args.cplex_msg,
        )
        result = trip_milp.materialize_relaxed_trip_plan(
            solve_env,
            relaxed,
            horizon_h=args.hours,
            economics=economics,
        )
    else:
        result = trip_milp.solve_executable_trip_milp_with_cplex(
            solve_env,
            horizon_h=args.hours,
            economics=economics,
            storage_reward_eur_per_t=args.storage_reward_eur_per_t,
            warm_start_native_actions_by_hour=warm_start_actions,
            time_limit_s=args.cplex_time_limit_s,
            mip_gap_rel=args.cplex_mip_gap_rel,
            threads=args.cplex_threads,
            msg=args.cplex_msg,
        )
    solve_s = time.perf_counter() - solve_start
    direct_replay = trip_milp.replay_trip_milp_plan(solve_env, result, stored_tol_t=1e-3)
    policy_metrics, mask_violations, policy_s = replay_policy_metrics(
        args,
        economics,
        seed,
        result.native_actions_by_hour,
    )

    greedy_obj = objective(
        greedy_metrics.total_cost,
        greedy_metrics.stored_t,
        args.storage_reward_eur_per_t,
    )
    oracle_obj = objective(
        policy_metrics.total_cost,
        policy_metrics.stored_t,
        args.storage_reward_eur_per_t,
    )
    headroom = greedy_obj - oracle_obj
    headroom_row = {
        "seed": seed,
        "greedy_solve_time_s": greedy_s,
        "oracle_policy_replay_time_s": policy_s,
        "executable_solve_time_s": solve_s,
        "oracle_level": args.oracle_level,
        "warm_start_policy": args.warm_start_policy,
        "warm_start_hours": len(warm_start_actions) if warm_start_actions is not None else 0,
        "executable_status": result.status,
        "result_level": result.level,
        "executable_is_valid": result.is_valid,
        "executable_validation_error": result.validation_error,
        "executable_deliveries": result.deliveries,
        "executable_binary_count": result.binary_count,
        "executable_variable_count": result.variable_count,
        "executable_constraint_count": result.constraint_count,
        "direct_replay_is_executable": direct_replay.is_executable,
        "direct_replay_stored_gap_t": direct_replay.stored_gap_t,
        "action_mask_violation_count": len(mask_violations),
        "greedy_objective": greedy_obj,
        "oracle_replay_objective": oracle_obj,
        "objective_headroom_eur": headroom,
        "objective_headroom_pct_abs_greedy": headroom / abs(greedy_obj)
        if abs(greedy_obj) > 1e-9
        else math.nan,
        "stored_headroom_t": policy_metrics.stored_t - greedy_metrics.stored_t,
        "vented_reduction_t": greedy_metrics.vented_t - policy_metrics.vented_t,
        "total_cost_saving_eur": greedy_metrics.total_cost - policy_metrics.total_cost,
        **{f"greedy_{key}": value for key, value in metric_fields(greedy_metrics).items()},
        **{f"oracle_replay_{key}": value for key, value in metric_fields(policy_metrics).items()},
    }

    reference_replay = reference_rows.get((seed, "executable_replay"))
    validation_row = replay_diff_row(
        seed=seed,
        policy_metrics=policy_metrics,
        direct_replay=direct_replay,
        reference_replay=reference_replay,
        storage_reward_eur_per_t=args.storage_reward_eur_per_t,
        mask_violations=mask_violations,
    )

    trip_rows = [
        {
            "seed": seed,
            "trip_index": index,
            **asdict(trip),
        }
        for index, trip in enumerate(result.trips)
    ]
    result_summary = {
        "seed": seed,
        "status": result.status,
        "is_valid": result.is_valid,
        "validation_error": result.validation_error,
        "stored_t": result.stored_t,
        "vented_t": result.vented_t,
        "total_cost": result.total_cost,
        "objective_value": result.objective_value,
        "deliveries": result.deliveries,
        "binary_count": result.binary_count,
        "variable_count": result.variable_count,
        "constraint_count": result.constraint_count,
    }
    return (
        trip_rows,
        headroom_row,
        validation_row,
        result.native_actions_by_hour,
        result_summary,
    )


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    metrics = [
        "objective_headroom_eur",
        "objective_headroom_pct_abs_greedy",
        "stored_headroom_t",
        "vented_reduction_t",
        "total_cost_saving_eur",
        "greedy_stored_t",
        "oracle_replay_stored_t",
        "greedy_vented_t",
        "oracle_replay_vented_t",
        "greedy_total_cost_ex_shortfall",
        "oracle_replay_total_cost_ex_shortfall",
    ]
    summary: dict[str, object] = {"episodes": len(rows)}
    for metric in metrics:
        values = [float(row[metric]) for row in rows if row.get(metric) not in (None, "")]
        if values:
            mean = sum(values) / len(values)
            summary[f"{metric}_mean"] = mean
            summary[f"{metric}_min"] = min(values)
            summary[f"{metric}_max"] = max(values)
    return [summary]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--hours", type=int, default=720)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--yara-buffer-t", type=float, default=7500.0)
    parser.add_argument("--terminal-buffer-t", type=float, default=7500.0)
    parser.add_argument("--speed-floor", type=float, default=0.7)
    parser.add_argument("--load-hours", type=int, nargs="+", default=[4, 6, 8, 10])
    parser.add_argument("--load-start-stride-h", type=int, default=12)
    parser.add_argument(
        "--oracle-level",
        choices=("executable", "relaxed-materialized"),
        default="executable",
    )
    parser.add_argument(
        "--warm-start-policy",
        choices=("none", "greedy", "source-native"),
        default="greedy",
    )
    parser.add_argument("--storage-reward-eur-per-t", type=float, default=1000.0)
    parser.add_argument("--carbon-price-eur-per-t", type=float, default=80.0)
    parser.add_argument("--cplex-time-limit-s", type=float, default=300.0)
    parser.add_argument("--cplex-mip-gap-rel", type=float, default=0.01)
    parser.add_argument("--cplex-threads", type=int, default=None)
    parser.add_argument("--cplex-msg", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir is None:
        args.output_dir = args.source_dir / "oracle_replay_trace_export"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reference_rows = load_reference_rows(args.source_dir / "comparison.csv")
    economics = EconomicParameters(carbon_price_eur_per_t=args.carbon_price_eur_per_t)

    replacing_seeds = set(args.seeds)
    all_trips = read_existing_csv(args.output_dir / "executable_trips_by_seed.csv", replacing_seeds)
    headroom_rows = read_existing_csv(
        args.output_dir / "greedy_vs_oracle_replay_headroom.csv",
        replacing_seeds,
    )
    validation_rows = read_existing_csv(
        args.output_dir / "oracle_replay_validation.csv",
        replacing_seeds,
    )
    start = time.perf_counter()
    with pruned_executable_options(set(args.load_hours), args.load_start_stride_h):
        for seed in args.seeds:
            seed_start = time.perf_counter()
            trip_rows, headroom_row, validation_row, native_actions, result_summary = run_seed(
                args,
                economics,
                seed,
                reference_rows,
            )
            all_trips.extend(trip_rows)
            headroom_rows.append(headroom_row)
            validation_rows.append(validation_row)
            seed_dir = args.output_dir / f"seed_{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)
            write_csv(seed_dir / "executable_trips.csv", trip_rows)
            (seed_dir / "native_actions_by_hour.json").write_text(
                json.dumps(native_actions, indent=2),
                encoding="utf-8",
            )
            (seed_dir / "executable_result_summary.json").write_text(
                json.dumps(
                    result_summary,
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            write_csv(args.output_dir / "executable_trips_by_seed.partial.csv", all_trips)
            write_csv(args.output_dir / "greedy_vs_oracle_replay_headroom.partial.csv", headroom_rows)
            write_csv(args.output_dir / "oracle_replay_validation.partial.csv", validation_rows)
            print(
                f"seed={seed}: stored_headroom={headroom_row['stored_headroom_t']:.1f} "
                f"objective_headroom={headroom_row['objective_headroom_eur']:.0f} "
                f"mask_violations={headroom_row['action_mask_violation_count']} "
                f"time={time.perf_counter() - seed_start:.1f}s",
                flush=True,
            )

    write_csv(args.output_dir / "executable_trips_by_seed.csv", all_trips)
    write_csv(args.output_dir / "greedy_vs_oracle_replay_headroom.csv", headroom_rows)
    write_csv(args.output_dir / "oracle_replay_validation.csv", validation_rows)
    write_csv(args.output_dir / "greedy_vs_oracle_replay_summary.csv", summarize(headroom_rows))
    (args.output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "source_dir": str(args.source_dir),
                "scenario": SCENARIO_ID,
                "hours": args.hours,
                "seeds": args.seeds,
                "yara_buffer_t": args.yara_buffer_t,
                "terminal_buffer_t": args.terminal_buffer_t,
                "speed_floor": args.speed_floor,
                "load_hours": args.load_hours,
                "load_start_stride_h": args.load_start_stride_h,
                "oracle_level": args.oracle_level,
                "warm_start_policy": args.warm_start_policy,
                "storage_reward_eur_per_t": args.storage_reward_eur_per_t,
                "carbon_price_eur_per_t": args.carbon_price_eur_per_t,
                "cplex_time_limit_s": args.cplex_time_limit_s,
                "cplex_mip_gap_rel": args.cplex_mip_gap_rel,
                "elapsed_s": time.perf_counter() - start,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(f"wrote {args.output_dir}")


if __name__ == "__main__":
    main()
