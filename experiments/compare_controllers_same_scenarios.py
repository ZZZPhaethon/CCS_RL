"""Compare controllers on identical disturbance scenarios.

Each controller gets a fresh environment for a seed, while ``ScenarioGenerator``
uses the same seed so disturbances are comparable across controllers. Static
MILP benchmarks are reported separately because they are perfect-foresight
optimizers, not episode controllers.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable

from sim.economics import CostModel, EconomicParameters
from sim.control.baselines import greedy_shuttle_policy, idle_policy
from sim.control.milp import solve_max_storage_fixed_horizon
from sim.control.rule_based import RuleBasedActionGenerator
from sim.control.rolling_milp import RollingMilpController
from sim.environment import (
    CCSEnv,
    CCSEnvConfig,
    MAX_WELL_RATE_MTPA,
    MIN_WELL_RATE_MTPA,
    VESSEL_GO_TERMINAL,
    VESSEL_WAIT,
)
from sim.metrics import EpisodeMetrics, run_episode
from sim.network_scenarios import (
    available_fixed_scenario_choices,
    build_fixed_scenario_demo,
    fixed_scenario_locations,
    resolve_fixed_scenario_id,
)
from sim.scenario_generation import Scenario, ScenarioConfig, ScenarioGenerator

ProgressLogger = Callable[[str], None]
PolicyFactory = Callable[[CCSEnv], Callable[[CCSEnv], dict[str, list]]]

def make_env(
    *,
    cap_hours: int,
    scenario_seed_config: ScenarioConfig,
    economics: EconomicParameters | None = None,
    fixed_scenario: str = "toy",
) -> CCSEnv:
    network, _state = build_fixed_scenario_demo(fixed_scenario)
    return CCSEnv(
        network,
        fixed_scenario_locations(fixed_scenario),
        scenario_generator=ScenarioGenerator(config=scenario_seed_config),
        cost_model=CostModel(economics),
        config=CCSEnvConfig(episode_hours=cap_hours),
    )


def scenario_signature(scenario: Scenario) -> str:
    payload = {
        "initial_inventory_t": scenario.initial_inventory_t,
        "emitter_availability": scenario.emitter_availability,
        "vessel_speed_factor": scenario.vessel_speed_factor,
        "well_available": scenario.well_available,
        "injectivity_factor": scenario.injectivity_factor,
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:12]


def rule_based_env_policy(env: CCSEnv) -> Callable[[CCSEnv], dict[str, list]]:
    generator = RuleBasedActionGenerator(env.network, env._routes)

    def policy(current_env: CCSEnv) -> dict[str, list]:
        frame = generator.next_action_frame(current_env.simulator.state)
        vessel_actions = [VESSEL_WAIT] * len(current_env.vessel_ids)
        well_rates = [MIN_WELL_RATE_MTPA] * len(current_env.well_ids)
        mask = current_env.vessel_action_mask()
        vessel_index = {vessel_id: i for i, vessel_id in enumerate(current_env.vessel_ids)}
        well_index = {well_id: i for i, well_id in enumerate(current_env.well_ids)}

        for proposal in frame.proposals:
            if proposal.verb == "sail_to" and proposal.entity_id in vessel_index:
                i = vessel_index[proposal.entity_id]
                route = current_env._routes[proposal.entity_id]
                destination = str(proposal.params["destination_id"])
                if destination in current_env.emitter_ids:
                    emitter_action = current_env.vessel_go_emitter_action(destination)
                    if mask[i][emitter_action]:
                        vessel_actions[i] = emitter_action
                elif destination == route["destination"] and mask[i][VESSEL_GO_TERMINAL]:
                    vessel_actions[i] = VESSEL_GO_TERMINAL
            elif proposal.verb == "set_well_split":
                for well_id, split in proposal.params["well_splits"].items():
                    if float(split) <= 0.0 or well_id not in well_index:
                        continue
                    well_rates[well_index[well_id]] = MAX_WELL_RATE_MTPA

        return {"vessels": vessel_actions, "wells": well_rates}

    return policy


def controller_factories(
    replan_every: int,
    progress: ProgressLogger,
    rolling_planning_horizon_h: int,
    rolling_time_limit_s: float,
    economics: EconomicParameters,
) -> dict[str, PolicyFactory]:
    return {
        "idle": lambda _env: idle_policy,
        "greedy_shuttle": lambda _env: greedy_shuttle_policy,
        "rule_based": rule_based_env_policy,
        "rolling_milp": lambda env: RollingMilpController(
            env,
            replan_every=replan_every,
            economics=economics,
            progress=progress,
            planning_horizon_h=rolling_planning_horizon_h,
            time_limit_s=rolling_time_limit_s,
        ),
    }


def metric_row(
    *,
    seed: int,
    controller: str,
    metrics: EpisodeMetrics,
    signature: str,
    solve_time_s: float,
    fixed_scenario: str = "toy",
) -> dict[str, object]:
    return {
        "seed": seed,
        "fixed_scenario": fixed_scenario,
        "controller": controller,
        "scenario_signature": signature,
        "solve_time_s": solve_time_s,
        **metrics.as_dict(),
    }


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_controller: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_controller[str(row["controller"])].append(row)

    metrics = [
        "elapsed_hours",
        "solve_time_s",
        "captured_t",
        "stored_t",
        "vented_t",
        "in_transit_growth_t",
        "loss_rate",
        "storage_rate",
        "vent_penalty",
        "storage_shortfall_penalty",
        "vessel_fuel",
        "conditioning",
        "reconditioning",
        "loading",
        "unloading",
        "operating_cost",
        "total_cost",
        "cost_per_stored_t",
        "total_cost_per_stored_t",
        "throttle_hours",
        "well_switch_count",
        "berth_wait_vessel_hours",
        "pressure_risk_hours",
        "longest_venting_streak_hours",
    ]
    summary_rows: list[dict[str, object]] = []
    for controller, controller_rows in sorted(by_controller.items()):
        out: dict[str, object] = {"controller": controller, "episodes": len(controller_rows)}
        for metric in metrics:
            values = [row[metric] for row in controller_rows if row.get(metric) is not None]
            if not values:
                out[f"{metric}_mean"] = ""
                out[f"{metric}_std"] = ""
                continue
            floats = [float(value) for value in values]
            mean = sum(floats) / len(floats)
            variance = sum((value - mean) ** 2 for value in floats) / len(floats)
            out[f"{metric}_mean"] = mean
            out[f"{metric}_std"] = variance**0.5
        summary_rows.append(out)
    return summary_rows


def static_fixed_horizon_milp_benchmark(
    cap_hours: int,
    time_limit_s: float | None,
    mip_gap_rel: float | None,
    mip_gap_abs: float | None,
    *,
    scenario_seed_config: ScenarioConfig,
    seed: int,
    economics: EconomicParameters,
    fixed_scenario: str = "toy",
) -> dict[str, object]:
    env = make_env(
        cap_hours=cap_hours,
        scenario_seed_config=scenario_seed_config,
        economics=economics,
        fixed_scenario=fixed_scenario,
    )
    scenario = env.scenario_generator.sample(env.network, seed=seed)
    solve_start = time.perf_counter()
    result = solve_max_storage_fixed_horizon(
        env,
        horizon_h=cap_hours,
        time_limit_s=time_limit_s,
        mip_gap_rel=mip_gap_rel,
        mip_gap_abs=mip_gap_abs,
        scenario=scenario,
        economics=economics,
    )
    solve_time_s = time.perf_counter() - solve_start
    is_valid = bool(getattr(result, "is_valid", result.status == "Optimal"))
    validation_error = str(getattr(result, "validation_error", ""))
    max_integrality = getattr(result, "max_binary_integrality_violation", "")
    row = {
        "case": "static_milp_fixed_horizon_scenario_oracle",
        "seed": seed,
        "fixed_scenario": fixed_scenario,
        "scenario_signature": scenario_signature(scenario),
        "horizon_h": cap_hours,
        "status": result.status,
        "solve_time_s": solve_time_s,
        "is_valid": is_valid,
        "validation_error": validation_error,
        "max_binary_integrality_violation": max_integrality,
        "time_limit_s": "" if time_limit_s is None else time_limit_s,
        "mip_gap_rel": "" if mip_gap_rel is None else mip_gap_rel,
        "mip_gap_abs": "" if mip_gap_abs is None else mip_gap_abs,
    }
    if not is_valid:
        row.update(
            {
                "stored_t": "",
                "deliveries": "",
                "vented_t": "",
                "in_transit_t": "",
                "in_transit_growth_t": "",
                "shortfall_t": "",
                "vessel_fuel": "",
                "conditioning": "",
                "reconditioning": "",
                "loading": "",
                "unloading": "",
                "operating_cost": "",
                "total_cost": "",
                "cost_per_stored_t": "",
                "total_cost_per_stored_t": "",
            }
        )
        return row

    row.update(
        {
            "stored_t": result.stored_t,
            "deliveries": result.deliveries,
            "vented_t": result.vented_t,
            "in_transit_t": result.in_transit_t,
            "in_transit_growth_t": result.in_transit_growth_t,
            "shortfall_t": result.shortfall_t,
            "vessel_fuel": getattr(result, "vessel_fuel", ""),
            "conditioning": getattr(result, "conditioning", ""),
            "reconditioning": getattr(result, "reconditioning", ""),
            "loading": getattr(result, "loading", ""),
            "unloading": getattr(result, "unloading", ""),
            "operating_cost": result.operating_cost,
            "total_cost": result.total_cost,
            "cost_per_stored_t": result.cost_per_stored_t,
            "total_cost_per_stored_t": getattr(result, "total_cost_per_stored_t", ""),
        }
    )
    return row


def _static_row_is_valid(row: dict[str, object]) -> bool:
    value = row.get("is_valid", True)
    if isinstance(value, str):
        return value.lower() in {"true", "1", "yes"}
    return bool(value)


def _blank_static_summary(rows: list[dict[str, object]], reason: str) -> dict[str, object]:
    return {
        "case": "static_milp_fixed_horizon_scenario_oracle_summary",
        "episodes": len(rows),
        "valid_episodes": 0,
        "horizon_h": rows[0].get("horizon_h", "") if rows else "",
        "status": ";".join(sorted({str(row.get("status", "")) for row in rows})),
        "solve_time_s": sum(float(row.get("solve_time_s") or 0.0) for row in rows) / len(rows) if rows else 0.0,
        "is_valid": False,
        "validation_error": reason,
        "max_binary_integrality_violation": max(
            float(row.get("max_binary_integrality_violation") or 0.0) for row in rows
        )
        if rows
        else "",
        "stored_t": "",
        "deliveries": "",
        "vented_t": "",
        "in_transit_t": "",
        "in_transit_growth_t": "",
        "shortfall_t": "",
        "vessel_fuel": "",
        "conditioning": "",
        "reconditioning": "",
        "loading": "",
        "unloading": "",
        "operating_cost": "",
        "total_cost": "",
        "cost_per_stored_t": "",
        "total_cost_per_stored_t": "",
        "time_limit_s": rows[0].get("time_limit_s", "") if rows else "",
        "mip_gap_rel": rows[0].get("mip_gap_rel", "") if rows else "",
        "mip_gap_abs": rows[0].get("mip_gap_abs", "") if rows else "",
    }


def _mean(rows: list[dict[str, object]], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows)


def _summarize_valid_static_rows(rows: list[dict[str, object]], all_rows: list[dict[str, object]]) -> dict[str, object]:
    stored_mean = _mean(rows, "stored_t")
    operating_cost_mean = _mean(rows, "operating_cost")
    total_cost_mean = _mean(rows, "total_cost")
    return {
        "case": "static_milp_fixed_horizon_scenario_oracle_summary",
        "episodes": len(all_rows),
        "valid_episodes": len(rows),
        "horizon_h": all_rows[0]["horizon_h"],
        "status": ";".join(sorted({str(row["status"]) for row in all_rows})),
        "solve_time_s": _mean(all_rows, "solve_time_s"),
        "is_valid": len(rows) == len(all_rows),
        "validation_error": "" if len(rows) == len(all_rows) else f"{len(all_rows) - len(rows)} invalid static MILP solves excluded",
        "max_binary_integrality_violation": max(float(row.get("max_binary_integrality_violation") or 0.0) for row in all_rows),
        "stored_t": stored_mean,
        "deliveries": _mean(rows, "deliveries"),
        "vented_t": _mean(rows, "vented_t"),
        "in_transit_t": _mean(rows, "in_transit_t"),
        "in_transit_growth_t": _mean(rows, "in_transit_growth_t"),
        "shortfall_t": _mean(rows, "shortfall_t"),
        "vessel_fuel": _mean(rows, "vessel_fuel"),
        "conditioning": _mean(rows, "conditioning"),
        "reconditioning": _mean(rows, "reconditioning"),
        "loading": _mean(rows, "loading"),
        "unloading": _mean(rows, "unloading"),
        "operating_cost": operating_cost_mean,
        "total_cost": total_cost_mean,
        "cost_per_stored_t": operating_cost_mean / stored_mean if stored_mean > 0 else float("nan"),
        "total_cost_per_stored_t": total_cost_mean / stored_mean if stored_mean > 0 else float("nan"),
        "time_limit_s": all_rows[0].get("time_limit_s", ""),
        "mip_gap_rel": all_rows[0].get("mip_gap_rel", ""),
        "mip_gap_abs": all_rows[0].get("mip_gap_abs", ""),
    }


def summarize_static_fixed_horizon_benchmarks(rows: list[dict[str, object]]) -> dict[str, object]:
    if len(rows) == 1:
        return rows[0]
    valid_rows = [row for row in rows if _static_row_is_valid(row)]
    if not valid_rows:
        return _blank_static_summary(rows, f"no valid static MILP solves ({len(rows)} invalid)")
    return _summarize_valid_static_rows(valid_rows, rows)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def format_cost_per_t(value: object) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(numeric):
        return "n/a"
    return f"EUR {numeric:,.2f}"


def write_report(path: Path, *, summary_rows: list[dict[str, object]], benchmark: dict[str, object]) -> None:
    lines = [
        "# Same-Scenario Controller Comparison",
        "",
        "All episode controllers are evaluated on identical disturbance trajectories for each seed.",
        "The static MILP row is a separate perfect-foresight benchmark, not an online episode controller.",
        "",
        "## Static MILP Benchmark",
        "",
    ]
    if benchmark.get("status") == "skipped":
        lines.append(f"- horizon: {float(benchmark['horizon_h']):.0f} h; static MILP solve skipped.")
    elif str(benchmark.get("case", "")).startswith("static_milp_fixed_horizon"):
        label = "same-scenario oracle" if "scenario_oracle" in str(benchmark.get("case", "")) else "nominal"
        episodes = benchmark.get("episodes")
        episodes_text = "" if episodes is None else f"; episodes: {episodes}"
        if not _static_row_is_valid(benchmark):
            reason = benchmark.get("validation_error") or "static MILP did not produce a validated integer solution"
            lines.append(
                f"- {label}; horizon: {float(benchmark['horizon_h']):.0f} h{episodes_text}; "
                f"status: {benchmark['status']}; "
                f"solve time: {float(benchmark.get('solve_time_s', 0.0)):,.1f}s; "
                f"not comparable: {reason}"
            )
        else:
            lines.append(
                f"- {label}; horizon: {float(benchmark['horizon_h']):.0f} h{episodes_text}; "
                f"status: {benchmark['status']}; "
                f"solve time: {float(benchmark.get('solve_time_s', 0.0)):,.1f}s; "
                f"stored: {float(benchmark['stored_t']):,.1f} t; "
                f"vented: {float(benchmark.get('vented_t', 0.0)):,.1f} t; "
                f"in-transit: {float(benchmark.get('in_transit_t', 0.0)):,.1f} t; "
                f"shortfall: {float(benchmark.get('shortfall_t', 0.0)):,.1f} t; "
                f"deliveries: {benchmark['deliveries']}; "
                f"operating cost: EUR {float(benchmark['operating_cost']):,.0f}; "
                f"total cost: EUR {float(benchmark.get('total_cost', benchmark['operating_cost'])):,.0f}; "
                f"op cost/t: {format_cost_per_t(benchmark['cost_per_stored_t'])}"
            )
    else:
        raise ValueError(f"Unsupported benchmark row: {benchmark}")
    lines += [
        "",
        "## Episode Controller Summary",
        "",
        "| Controller | Hours mean | Solve s mean | Stored t mean | Vented t mean | Shortfall penalty mean | Total cost mean | Op cost mean |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['controller']} | {float(row.get('elapsed_hours_mean', 0.0)):,.1f} | "
            f"{float(row.get('solve_time_s_mean', 0.0)):,.1f} | "
            f"{float(row.get('stored_t_mean', 0.0)):,.1f} | "
            f"{float(row.get('vented_t_mean', 0.0)):,.1f} | "
            f"{float(row.get('storage_shortfall_penalty_mean', 0.0)):,.0f} | "
            f"{float(row.get('total_cost_mean', row.get('operating_cost_mean', 0.0))):,.0f} | "
            f"{float(row.get('operating_cost_mean', 0.0)):,.0f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cap-hours", type=int, default=600)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument(
        "--scenario",
        choices=available_fixed_scenario_choices(),
        default="toy",
        help="Fixed network scenario from the scenarios folder. Disturbances are still controlled by --quiet-scenario and --seeds.",
    )
    parser.add_argument(
        "--controllers",
        nargs="+",
        choices=["idle", "greedy_shuttle", "rule_based", "rolling_milp"],
        default=["idle", "greedy_shuttle", "rolling_milp"],
    )
    parser.add_argument("--rolling-replan-every", type=int, default=24)
    parser.add_argument(
        "--rolling-planning-horizon-h",
        type=int,
        default=168,
        help="Lookahead horizon for the rolling fixed-horizon MILP controller.",
    )
    parser.add_argument(
        "--rolling-time-limit-s",
        type=float,
        default=30.0,
        help="Per-replan solver time limit for the rolling MILP controller.",
    )
    parser.add_argument("--skip-static-milp", action="store_true")
    parser.add_argument("--static-milp-time-limit-s", type=float, default=600.0)
    parser.add_argument(
        "--static-milp-gap-rel",
        type=float,
        default=None,
        help="Optional relative MIP gap tolerance for the fixed-horizon static MILP.",
    )
    parser.add_argument(
        "--static-milp-gap-abs",
        type=float,
        default=None,
        help="Optional absolute MIP gap tolerance for the fixed-horizon static MILP.",
    )
    parser.add_argument("--random-initial-inventory", action="store_true")
    parser.add_argument("--quiet-scenario", action="store_true")
    parser.add_argument(
        "--storage-shortfall-penalty-eur-per-t",
        type=float,
        default=100.0,
        help="Set to 0 for a vent-penalty-only objective; passed to env, rolling MILP, and static MILP.",
    )
    return parser.parse_args()


def _quiet_config(cap_hours: int, random_initial_inventory: bool) -> ScenarioConfig:
    return ScenarioConfig(
        episode_hours=cap_hours,
        randomize_initial_inventory=random_initial_inventory,
        capture_noise_std=0.0,
        capture_outage_rate_per_week=0.0,
        enable_weather=False,
        well_maintenance_rate_per_week=0.0,
        injectivity_max_decline=0.0,
        injectivity_noise_std=0.0,
    )


def main() -> None:
    args = parse_args()
    fixed_scenario = resolve_fixed_scenario_id(args.scenario)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = args.output_dir / "progress.log"
    progress_path.write_text("", encoding="utf-8")

    def log(message: str) -> None:
        print(message, flush=True)
        with progress_path.open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")

    scenario_config = (
        _quiet_config(args.cap_hours, args.random_initial_inventory)
        if args.quiet_scenario
        else ScenarioConfig(
            episode_hours=args.cap_hours,
            randomize_initial_inventory=args.random_initial_inventory,
        )
    )
    economics = EconomicParameters(
        storage_shortfall_eur_per_t=args.storage_shortfall_penalty_eur_per_t
    )

    rows: list[dict[str, object]] = []
    seed_to_signature: dict[int, str] = {}
    total_runs = len(args.seeds) * len(args.controllers)
    run_index = 0
    script_start = time.perf_counter()
    log(
        f"Running {total_runs} fixed-horizon episode rollouts: "
        f"scenario={fixed_scenario}, cap={args.cap_hours} h, seeds={args.seeds}, controllers={args.controllers}"
    )
    factories = controller_factories(
        args.rolling_replan_every,
        log,
        args.rolling_planning_horizon_h,
        args.rolling_time_limit_s,
        economics,
    )
    partial_path = args.output_dir / "controller_comparison_by_seed.partial.csv"
    for seed in args.seeds:
        seed_signatures = set()
        for controller_name in args.controllers:
            factory = factories[controller_name]
            run_index += 1
            run_start = time.perf_counter()
            log(f"[{run_index}/{total_runs}] seed={seed} controller={controller_name} ...")
            env = make_env(
                cap_hours=args.cap_hours,
                scenario_seed_config=scenario_config,
                economics=economics,
                fixed_scenario=fixed_scenario,
            )
            metrics = run_episode(env, factory(env), seed=seed)
            run_duration_s = time.perf_counter() - run_start
            signature = scenario_signature(env.scenario)
            seed_signatures.add(signature)
            rows.append(
                metric_row(
                    seed=seed,
                    controller=controller_name,
                    metrics=metrics,
                    signature=signature,
                    solve_time_s=run_duration_s,
                    fixed_scenario=fixed_scenario,
                )
            )
            write_csv(partial_path, rows)
            log(
                f"[{run_index}/{total_runs}] seed={seed} controller={controller_name} done "
                f"in {run_duration_s:.1f}s; "
                f"hours={metrics.elapsed_hours:.0f}; stored={metrics.stored_t:.1f} t; "
                f"vented={metrics.vented_t:.1f} t; total_cost={metrics.total_cost:,.0f}"
            )
        if len(seed_signatures) != 1:
            raise RuntimeError(f"Seed {seed} did not produce identical scenarios across controllers.")
        seed_to_signature[seed] = next(iter(seed_signatures))

    summary_rows = summarize(rows)
    benchmark_rows: list[dict[str, object]]
    if args.skip_static_milp:
        log("Skipping static MILP fixed-horizon same-scenario oracle.")
        benchmark = {
            "case": "static_milp_fixed_horizon_scenario_oracle",
            "episodes": len(args.seeds),
            "fixed_scenario": fixed_scenario,
            "horizon_h": args.cap_hours,
            "status": "skipped",
            "solve_time_s": "",
            "is_valid": False,
            "validation_error": "static MILP solve skipped",
            "max_binary_integrality_violation": "",
            "stored_t": "",
            "deliveries": "",
            "vented_t": "",
            "in_transit_t": "",
            "in_transit_growth_t": "",
            "shortfall_t": "",
            "vessel_fuel": "",
            "conditioning": "",
            "reconditioning": "",
            "loading": "",
            "unloading": "",
            "operating_cost": "",
            "total_cost": "",
            "cost_per_stored_t": "",
            "total_cost_per_stored_t": "",
            "time_limit_s": "",
            "mip_gap_rel": "",
            "mip_gap_abs": "",
        }
        benchmark_rows = [benchmark]
    else:
        benchmark_rows = []
        for seed in args.seeds:
            log(f"Solving static MILP fixed-horizon same-scenario oracle for seed={seed} ...")
            row = static_fixed_horizon_milp_benchmark(
                args.cap_hours,
                args.static_milp_time_limit_s,
                args.static_milp_gap_rel,
                args.static_milp_gap_abs,
                scenario_seed_config=scenario_config,
                seed=seed,
                economics=economics,
                fixed_scenario=fixed_scenario,
            )
            if row["scenario_signature"] != seed_to_signature[seed]:
                raise RuntimeError(f"Static MILP scenario signature mismatch for seed {seed}.")
            benchmark_rows.append(row)
        benchmark = summarize_static_fixed_horizon_benchmarks(benchmark_rows)

    write_csv(args.output_dir / "controller_comparison_by_seed.csv", rows)
    write_csv(args.output_dir / "controller_comparison_summary.csv", summary_rows)
    write_csv(args.output_dir / "static_milp_same_scenario_benchmark.csv", benchmark_rows)
    write_report(args.output_dir / "controller_comparison_report.md", summary_rows=summary_rows, benchmark=benchmark)
    log(f"Wrote outputs to {args.output_dir}")
    log(f"Total wall-clock time: {time.perf_counter() - script_start:.1f}s")


if __name__ == "__main__":
    main()
