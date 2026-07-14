"""Replay-validated native MPC headroom experiment with optional rolling MILP."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from sim.control.baselines import greedy_shuttle_policy
from sim.control.native_mpc import RollingNativeMpcController
from sim.control.replay import ReplayExpectation, replay_native_actions
from sim.control.rolling_milp import RollingMilpController
from sim.economics import CostModel, EconomicParameters
from sim.environment import CCSEnv, CCSEnvConfig
from sim.metrics import EpisodeMetrics, _MetricsRecorder
from sim.network_scenarios import build_fixed_scenario_demo, fixed_scenario_locations
from sim.scenario_generation import ScenarioConfig, ScenarioGenerator

SCENARIO_ID = "northern_lights_phase1_3vessels"
OBJECTIVE_MODE = "vent_end_unstored_operating_cost"


def make_env(args: argparse.Namespace, economics: EconomicParameters) -> CCSEnv:
    network, _state = build_fixed_scenario_demo(SCENARIO_ID)
    disturbed = args.disturbance_profile == "block"
    scenario_config = ScenarioConfig(
        episode_hours=args.hours,
        weather_process="block",
        weather_update_hours=args.weather_update_hours,
        weather_update_speed_factor_range=(
            (args.weather_speed_min, args.weather_speed_max) if disturbed else (1.0, 1.0)
        ),
        capture_noise_std=args.capture_noise_std if disturbed else 0.0,
        capture_outage_rate_per_week=0.0,
        capture_high_output_rate_per_week=(
            args.capture_high_output_rate_per_week if disturbed else 0.0
        ),
        capture_high_output_mean_hours=args.capture_high_output_mean_hours,
        capture_high_output_multiplier_range=(args.capture_multiplier_min, args.capture_multiplier_max),
        well_maintenance_rate_per_week=0.0,
        randomize_initial_inventory=disturbed,
    )
    return CCSEnv(
        network,
        fixed_scenario_locations(SCENARIO_ID),
        scenario_generator=ScenarioGenerator(config=scenario_config),
        cost_model=CostModel(economics),
        config=CCSEnvConfig(episode_hours=args.hours, reward_mode="vent_first"),
    )


def _validate_rl_executable_action(env: CCSEnv, action: dict[str, list[int]]) -> None:
    for kind, choices, masks, entity_ids in (
        ("vessel", action["vessels"], env.vessel_action_mask(), env.vessel_ids),
        ("well", action["wells"], env.well_rate_action_mask(), env.well_ids),
    ):
        if len(choices) != len(entity_ids):
            raise RuntimeError(f"{kind} action trace has the wrong action dimension")
        for entity_id, choice, mask in zip(entity_ids, choices, masks):
            if not (0 <= int(choice) < len(mask) and mask[int(choice)]):
                raise RuntimeError(
                    f"{kind} action trace is not RL-executable for {entity_id}: {choice}"
                )


def collect_actions(
    env: CCSEnv,
    policy,
    seed: int,
) -> tuple[list[dict[str, list[int]]], EpisodeMetrics, ReplayExpectation]:
    env.reset(seed=seed)
    recorder = _MetricsRecorder(env)
    actions: list[dict[str, list[int]]] = []
    injection_tph: list[float] = []
    overflow_risk_t = 0.0
    done = False
    while not done:
        action = policy(env)
        native = {
            "vessels": [int(value) for value in action["vessels"]],
            "wells": [int(value) for value in action["wells"]],
        }
        _validate_rl_executable_action(env, native)
        actions.append(native)
        before_stored_t = float(env.cumulative_stored_t)
        _obs, reward, terminated, truncated, info = env.step(native)
        injection_tph.append(float(env.cumulative_stored_t) - before_stored_t)
        overflow_risk_t += float(info.get("overflow_risk_t", 0.0))
        recorder.record_step(reward, info)
        done = terminated or truncated

    metrics = recorder.result()
    state = env.simulator.state
    expectation = ReplayExpectation(
        required_fields=frozenset(
            {
                "elapsed_hours",
                "stored_t",
                "vented_t",
                "captured_t",
                "in_transit_t",
                "vessel_fuel",
                "conditioning",
                "reconditioning",
                "loading",
                "unloading",
                "operating_cost",
                "total_cost",
                "total_reward",
                "objective_value",
                "overflow_risk_t",
                "injection_tph",
                "entity_inventory_t",
                "vessel_berths",
            }
        ),
        elapsed_hours=int(metrics.elapsed_hours),
        stored_t=metrics.stored_t,
        vented_t=metrics.vented_t,
        captured_t=metrics.captured_t,
        in_transit_t=metrics.in_transit_t,
        vessel_fuel=metrics.vessel_fuel,
        conditioning=metrics.conditioning,
        reconditioning=metrics.reconditioning,
        loading=metrics.loading,
        unloading=metrics.unloading,
        operating_cost=metrics.operating_cost,
        total_cost=metrics.total_cost,
        total_reward=metrics.total_reward,
        objective_value=-metrics.total_reward / float(env.config.reward_scale),
        overflow_risk_t=overflow_risk_t,
        injection_tph=tuple(injection_tph),
        entity_inventory_t={
            entity_id: float(state.entity_inventory_t.get(entity_id, 0.0))
            for entity_id in env.network.entities
        },
        vessel_berths={
            vessel_id: state.vessel_berths.get(vessel_id)
            for vessel_id in env.vessel_ids
        },
    )
    return actions, metrics, expectation


def run_seed(args: argparse.Namespace, seed: int, economics: EconomicParameters) -> tuple[dict[str, object], dict[str, list[dict[str, list[int]]]]]:
    greedy_actions, greedy_trace, greedy_expected = collect_actions(
        make_env(args, economics),
        greedy_shuttle_policy,
        seed,
    )
    greedy_replay_env = make_env(args, economics)
    greedy_replay_env.reset(seed=seed)
    greedy_replay = replay_native_actions(
        greedy_replay_env,
        greedy_actions,
        horizon_h=args.hours,
        expected=greedy_expected,
    )

    trace_env = make_env(args, economics)
    mpc = RollingNativeMpcController(
        trace_env,
        replan_every=args.replan_every,
        planning_horizon_h=args.planning_horizon_h,
    )
    mpc_start = time.perf_counter()
    mpc_actions, native_mpc_trace, native_mpc_expected = collect_actions(
        trace_env,
        mpc,
        seed,
    )
    mpc_wall_s = time.perf_counter() - mpc_start
    native_mpc_replay_env = make_env(args, economics)
    native_mpc_replay_env.reset(seed=seed)
    native_mpc_replay = replay_native_actions(
        native_mpc_replay_env,
        mpc_actions,
        horizon_h=args.hours,
        expected=native_mpc_expected,
    )
    greedy = greedy_trace
    native_mpc = native_mpc_trace
    greedy_unit_cost = greedy.total_cost / greedy.stored_t if greedy.stored_t > 0.0 else math.nan
    mpc_unit_cost = native_mpc.total_cost / native_mpc.stored_t if native_mpc.stored_t > 0.0 else math.nan
    replay_ok = (
        greedy_replay.is_executable
        and greedy_replay.is_exact
        and native_mpc_replay.is_executable
        and native_mpc_replay.is_exact
        and mpc.last_trace_replay_is_valid
        and mpc.last_trace_replay_is_exact
        and len(greedy_actions) == args.hours
        and len(mpc_actions) == args.hours
    )
    row: dict[str, object] = {
        "seed": seed,
        "objective_mode": OBJECTIVE_MODE,
        "replay_ok": replay_ok,
        "greedy_trace_replay_matches": greedy_replay.is_exact,
        "native_mpc_trace_replay_matches": native_mpc_replay.is_exact,
        "greedy_replay_is_executable": greedy_replay.is_executable,
        "greedy_replay_is_exact": greedy_replay.is_exact,
        "greedy_replay_mismatches": ";".join(greedy_replay.mismatches),
        "native_mpc_replay_is_executable": native_mpc_replay.is_executable,
        "native_mpc_replay_is_exact": native_mpc_replay.is_exact,
        "native_mpc_replay_mismatches": ";".join(native_mpc_replay.mismatches),
        "candidate_evaluations": mpc.candidate_evaluations,
        "native_mpc_wall_s": mpc_wall_s,
        "last_candidate": mpc.last_candidate_name,
        "greedy_vented_t": greedy.vented_t,
        "native_mpc_vented_t": native_mpc.vented_t,
        "vented_reduction_t": greedy.vented_t - native_mpc.vented_t if replay_ok else "",
        "vented_reduction_pct": (
            100.0 * (greedy.vented_t - native_mpc.vented_t) / greedy.vented_t
            if replay_ok and greedy.vented_t > 0.0
            else (0.0 if replay_ok else "")
        ),
        "greedy_stored_t": greedy.stored_t,
        "native_mpc_stored_t": native_mpc.stored_t,
        "greedy_total_cost_eur": greedy.total_cost,
        "native_mpc_total_cost_eur": native_mpc.total_cost,
        "total_cost_saving_eur": greedy.total_cost - native_mpc.total_cost if replay_ok else "",
        "greedy_unit_cost_eur_per_t": greedy_unit_cost,
        "native_mpc_unit_cost_eur_per_t": mpc_unit_cost,
        "unit_cost_saving_eur_per_t": greedy_unit_cost - mpc_unit_cost if replay_ok else "",
    }
    actions = {"greedy": greedy_actions, "native_mpc": mpc_actions}

    if getattr(args, "include_rolling_milp", False):
        milp_env = make_env(args, economics)
        rolling_milp = RollingMilpController(
            milp_env,
            replan_every=args.replan_every,
            planning_horizon_h=args.planning_horizon_h,
            time_limit_s=args.rolling_milp_time_limit_s,
            solver=getattr(args, "rolling_milp_solver", "cbc"),
            economics=economics,
        )
        milp_start = time.perf_counter()
        milp_actions, milp_trace, milp_expected = collect_actions(
            milp_env,
            rolling_milp,
            seed,
        )
        milp_wall_s = time.perf_counter() - milp_start
        milp_replay_env = make_env(args, economics)
        milp_replay_env.reset(seed=seed)
        milp_replay = replay_native_actions(
            milp_replay_env,
            milp_actions,
            horizon_h=args.hours,
            expected=milp_expected,
        )
        milp_replay_ok = (
            rolling_milp.last_plan_valid
            and milp_replay.is_executable
            and milp_replay.is_exact
            and len(milp_actions) == args.hours
        )
        milp_unit_cost = (
            milp_trace.total_cost / milp_trace.stored_t
            if milp_trace.stored_t > 0.0
            else math.nan
        )
        row.update(
            {
                "rolling_milp_replay_ok": milp_replay_ok,
                "rolling_milp_replay_is_executable": milp_replay.is_executable,
                "rolling_milp_replay_is_exact": milp_replay.is_exact,
                "rolling_milp_replay_mismatches": ";".join(milp_replay.mismatches),
                "rolling_milp_solver": rolling_milp.solver,
                "rolling_milp_wall_s": milp_wall_s,
                "rolling_milp_replans": rolling_milp.replan_count,
                "rolling_milp_model_inexact_replans": rolling_milp.model_inexact_replan_count,
                "rolling_milp_vented_t": milp_trace.vented_t,
                "rolling_milp_stored_t": milp_trace.stored_t,
                "rolling_milp_total_cost_eur": milp_trace.total_cost,
                "rolling_milp_unit_cost_eur_per_t": milp_unit_cost,
            }
        )
        actions["rolling_milp"] = milp_actions

    return row, actions


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hours", type=int, default=720)
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--disturbance-profile", choices=("none", "block"), default="block")
    parser.add_argument("--replan-every", type=int, default=24)
    parser.add_argument("--planning-horizon-h", type=int, default=168)
    parser.add_argument("--weather-update-hours", type=float, default=24.0)
    parser.add_argument("--weather-speed-min", type=float, default=0.75)
    parser.add_argument("--weather-speed-max", type=float, default=1.0)
    parser.add_argument("--capture-noise-std", type=float, default=0.10)
    parser.add_argument("--capture-high-output-rate-per-week", type=float, default=0.5)
    parser.add_argument("--capture-high-output-mean-hours", type=float, default=48.0)
    parser.add_argument("--capture-multiplier-min", type=float, default=1.25)
    parser.add_argument("--capture-multiplier-max", type=float, default=1.75)
    parser.add_argument(
        "--include-rolling-milp",
        action="store_true",
        help="Also evaluate the replay-grounded rolling MILP on the same scenario.",
    )
    parser.add_argument("--rolling-milp-time-limit-s", type=float, default=30.0)
    parser.add_argument(
        "--rolling-milp-solver",
        choices=("cbc", "cplex", "cplex_native"),
        default="cbc",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("output/rolling_native_mpc_block_24h_720h"))
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    economics = EconomicParameters()
    rows: list[dict[str, object]] = []
    for seed_text in args.seeds.split(","):
        seed = int(seed_text)
        row, actions = run_seed(args, seed, economics)
        rows.append(row)
        with (args.output_dir / f"seed_{seed}_native_actions.json").open("w", encoding="utf-8") as handle:
            json.dump(actions, handle, indent=2)
        if row["replay_ok"]:
            message = (
                f"seed={seed} greedy_vent={row['greedy_vented_t']:.1f} "
                f"native_mpc_vent={row['native_mpc_vented_t']:.1f} "
                f"reduction={row['vented_reduction_pct']:.1f}% replay=True"
            )
            if args.include_rolling_milp:
                message += (
                    f" rolling_milp_vent={row['rolling_milp_vented_t']:.1f} "
                    f"milp_replay={row['rolling_milp_replay_ok']}"
                )
        else:
            message = (
                f"seed={seed} replay=False; greedy_mismatches="
                f"{row['greedy_replay_mismatches']}; native_mpc_mismatches="
                f"{row['native_mpc_replay_mismatches']}"
            )
        print(message, flush=True)
    write_csv(args.output_dir / "by_seed.csv", rows)
    with (args.output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2, default=str)


if __name__ == "__main__":
    main()
