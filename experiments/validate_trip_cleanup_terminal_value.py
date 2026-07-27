"""Structural and coordinated-replay validation for the compact cleanup value."""

from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
import statistics
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.evaluate_trip_cleanup_terminal_value import (
    CAPACITY_T,
    LOAD_RATE_TPH,
    _nominal_fuel_hours,
    _predict_one,
    _solver,
)
from experiments.rolling_native_mpc_headroom import make_env
from sim.economics import EconomicParameters
from sim.environment import VESSEL_GO_TERMINAL, VESSEL_WAIT


SOURCES = ("brevik", "celsio", "yara_sluiskil")
VESSELS = ("northern_pathfinder", "northern_phoenix", "northern_pioneer")
TERMINAL = "oygarden_terminal"


def _trace_paths(seed: int) -> list[tuple[str, Path]]:
    main = Path("output/rolling_economic_objective_mpc_720h_5seeds_2026-07-15")
    return [
        ("lexicographic", main / f"seed_{seed}_mpc_lexicographic_actions.json"),
        ("economic", main / f"seed_{seed}_mpc_economic_actions.json"),
        ("economic_safe", main / f"seed_{seed}_mpc_economic_safe_actions.json"),
        (
            "economic_safe_strict",
            Path("output/rolling_economic_objective_mpc_strict_720h_5seeds_2026-07-15")
            / f"seed_{seed}_mpc_economic_safe_strict_actions.json",
        ),
        (
            "economic_lex_guard",
            Path("output/rolling_economic_objective_mpc_lex_guard_720h_5seeds_2026-07-15")
            / f"seed_{seed}_mpc_economic_lex_guard_actions.json",
        ),
    ]


def _disable_new_capture(env) -> None:
    for emitter_id in env.emitter_ids:
        emitter = env.network.entities[emitter_id]
        profile = emitter.hourly_capture_profile_tph
        env.network.entities[emitter_id] = replace(
            emitter,
            nominal_capture_tph=0.0,
            hourly_capture_profile_tph=(
                tuple(0.0 for _ in profile) if profile is not None else None
            ),
        )


def _disable_cleanup_disturbances(env) -> None:
    scenario = env.scenario
    if scenario is not None:
        scenario.emitter_availability = {
            key: [1.0] * len(values) for key, values in scenario.emitter_availability.items()
        }
        scenario.vessel_speed_factor = {
            key: [1.0] * len(values) for key, values in scenario.vessel_speed_factor.items()
        }
        scenario.leg_speed_factor = {
            key: [1.0] * len(values) for key, values in scenario.leg_speed_factor.items()
        }
        scenario.well_available = {
            key: [True] * len(values) for key, values in scenario.well_available.items()
        }
        scenario.injectivity_factor = {
            key: [1.0] * len(values) for key, values in scenario.injectivity_factor.items()
        }
    state = env.simulator.state
    state.emitter_availability = {emitter_id: 1.0 for emitter_id in env.emitter_ids}
    state.vessel_speed_factor = {vessel_id: 1.0 for vessel_id in env.vessel_ids}
    state.leg_speed_factor = {key: 1.0 for key in state.leg_speed_factor}
    state.well_available = {well_id: True for well_id in env.well_ids}
    state.injectivity_factor = {well_id: 1.0 for well_id in env.well_ids}


def _queues_empty(env) -> bool:
    return not any(env.simulator.state.terminal_unload_queues.values())


def _vessels_ready_at_sources(env, tolerance_t: float) -> bool:
    state = env.simulator.state
    for vessel_id in env.vessel_ids:
        vessel_state = env.simulator.vessel_states[vessel_id]
        if vessel_state["mode"] != "berthed" or vessel_state["berth"] not in env.emitter_ids:
            return False
        if float(state.entity_inventory_t.get(vessel_id, 0.0)) > tolerance_t:
            return False
    return True


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path(
            "output/trip_cleanup_terminal_value_500_v5_2026-07-16/predictions.csv"
        ),
    )
    parser.add_argument(
        "--run-config",
        type=Path,
        default=Path(
            "output/rolling_economic_objective_mpc_720h_5seeds_2026-07-15/run_config.json"
        ),
    )
    parser.add_argument("--max-cleanup-h", type=int, default=720)
    parser.add_argument("--tolerance-t", type=float, default=1.0)
    parser.add_argument("--solver", choices=("auto", "cplex", "cbc"), default="cplex")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/trip_cleanup_terminal_value_validation_v2_2026-07-16"),
    )
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _structural_audit(rows: list[dict[str, str]], params) -> dict[str, object]:
    max_mass_error_t = 0.0
    max_capacity_violation_t = 0.0
    max_integer_violation = 0.0
    max_component_error_eur = 0.0
    nonoptimal = 0
    for row in rows:
        shipped = json.loads(row["trip_shipped_json"])
        trips = json.loads(row["trip_counts_json"])
        topup = json.loads(row["trip_topup_json"])
        nonoptimal += row["trip_status"] != "Optimal"
        for source in SOURCES:
            assigned_t = sum(
                float(shipped[f"{vessel}->{source}"])
                + float(topup[f"{vessel}->{source}"])
                for vessel in VESSELS
            )
            max_mass_error_t = max(
                max_mass_error_t,
                abs(assigned_t - float(row[f"f_source_t__{source}"])),
            )
        for key, amount_t in shipped.items():
            trip_count = float(trips[key])
            max_capacity_violation_t = max(
                max_capacity_violation_t,
                float(amount_t) - CAPACITY_T * trip_count,
            )
            max_integer_violation = max(
                max_integer_violation, abs(trip_count - round(trip_count))
            )
        source_total_t = float(row["f_source_total_t"])
        cargo_total_t = float(row["f_cargo_total_t"])
        unstored_total_t = float(row["f_unstored_total_t"])
        expected = {
            "trip_conditioning_eur": params.conditioning_eur_per_t * source_total_t,
            "trip_reconditioning_eur": params.reconditioning_eur_per_t * unstored_total_t,
            "trip_loading_eur": params.hoteling_fuel_eur_per_h * source_total_t / LOAD_RATE_TPH,
            "trip_unloading_eur": (
                params.hoteling_fuel_eur_per_h
                * (source_total_t + cargo_total_t)
                / LOAD_RATE_TPH
            ),
            "trip_vessel_fuel_eur": (
                params.vessel_fuel_eur_per_h_sailing * float(row["trip_fuel_h"])
            ),
        }
        max_component_error_eur = max(
            max_component_error_eur,
            *(abs(float(row[key]) - value) for key, value in expected.items()),
        )
        expected_total = sum(expected.values())
        max_component_error_eur = max(
            max_component_error_eur,
            abs(float(row["trip_cleanup_cost_eur"]) - expected_total),
        )
    return {
        "sample_count": len(rows),
        "nonoptimal_count": nonoptimal,
        "max_source_mass_balance_error_t": max_mass_error_t,
        "max_trip_capacity_violation_t": max(0.0, max_capacity_violation_t),
        "max_trip_integer_violation": max_integer_violation,
        "max_cost_component_error_eur": max_component_error_eur,
        "passed": (
            nonoptimal == 0
            and max_mass_error_t <= 1e-5
            and max_capacity_violation_t <= 1e-7
            and max_integer_violation <= 1e-9
            and max_component_error_eur <= 1e-5
        ),
    }


def _synthetic_row(template: dict[str, str], source: str | None, amount_t: float):
    row = dict(template)
    homes = {
        "northern_pathfinder": "celsio",
        "northern_phoenix": "yara_sluiskil",
        "northern_pioneer": "brevik",
    }
    for emitter in SOURCES:
        row[f"f_source_t__{emitter}"] = str(amount_t if emitter == source else 0.0)
    for vessel in VESSELS:
        row[f"f_cargo_t__{vessel}"] = "0"
        row[f"f_cargo_positive__{vessel}"] = "0"
        row[f"f_vessel_remaining_sail_h__{vessel}"] = "0"
        row[f"f_vessel_loaded_remaining_sail_h__{vessel}"] = "0"
        for node in (*SOURCES, TERMINAL):
            row[f"f_vessel_at__{vessel}__{node}"] = str(float(node == homes[vessel]))
            row[f"f_vessel_sailing_to__{vessel}__{node}"] = "0"
    row[f"f_terminal_t__{TERMINAL}"] = str(amount_t if source is None else 0.0)
    row["f_source_total_t"] = str(amount_t if source is not None else 0.0)
    row["f_cargo_total_t"] = "0"
    row["f_terminal_total_t"] = str(amount_t if source is None else 0.0)
    row["f_downstream_total_t"] = "0"
    row["f_unstored_total_t"] = str(amount_t)
    return row


def _ordering_tests(
    template,
    *,
    sources,
    vessels,
    terminal,
    terminal_leg,
    direct_leg,
    params,
    solver,
) -> list[dict[str, object]]:
    cases = {
        "brevik_7500": _synthetic_row(template, "brevik", 7_500.0),
        "yara_7500": _synthetic_row(template, "yara_sluiskil", 7_500.0),
        "brevik_7400": _synthetic_row(template, "brevik", 7_400.0),
        "brevik_7600": _synthetic_row(template, "brevik", 7_600.0),
        "terminal_7500": _synthetic_row(template, None, 7_500.0),
    }
    values = {
        name: float(
            _predict_one(
                row,
                sources=sources,
                vessels=vessels,
                terminal=terminal,
                terminal_leg=terminal_leg,
                direct_leg=direct_leg,
                params=params,
                solver=solver,
            )["trip_cleanup_cost_eur"]
        )
        for name, row in cases.items()
    }
    definitions = [
        ("far_source_costs_more", "yara_7500", "brevik_7500"),
        ("capacity_boundary_costs_more", "brevik_7600", "brevik_7400"),
        ("source_stock_costs_more_than_terminal", "brevik_7500", "terminal_7500"),
    ]
    return [
        {
            "test": name,
            "higher_case": higher,
            "higher_cost_eur": values[higher],
            "lower_case": lower,
            "lower_cost_eur": values[lower],
            "difference_eur": values[higher] - values[lower],
            "passed": values[higher] > values[lower] + 1e-6,
        }
        for name, higher, lower in definitions
    ]


@dataclass
class _ExecutionPlan:
    initial_active: bool
    initial_target_cargo_t: float
    trips: list[tuple[str, float]]
    active_trip: tuple[str, float] | None = None


def _execution_plans(env, prediction: dict[str, object]) -> dict[str, _ExecutionPlan]:
    shipped = json.loads(str(prediction["trip_shipped_json"]))
    trip_counts = json.loads(str(prediction["trip_counts_json"]))
    topup = json.loads(str(prediction["trip_topup_json"]))
    first = json.loads(str(prediction["trip_first_json"]))
    state = env.simulator.state
    plans = {}
    for vessel in env.vessel_ids:
        cargo_t = float(state.entity_inventory_t.get(vessel, 0.0))
        vessel_state = env.simulator.vessel_states[vessel]
        node = (
            vessel_state.get("berth")
            if vessel_state["mode"] == "berthed"
            else vessel_state.get("destination")
        )
        initial_topup_t = (
            float(topup.get(f"{vessel}->{node}", 0.0)) if node in env.emitter_ids else 0.0
        )
        items: list[tuple[str, float]] = []
        for source in env.emitter_ids:
            key = f"{vessel}->{source}"
            amount_t = float(shipped[key])
            count = int(trip_counts[key])
            for _ in range(count):
                trip_amount_t = min(CAPACITY_T, amount_t)
                if trip_amount_t > 1e-7:
                    items.append((source, trip_amount_t))
                    amount_t -= trip_amount_t
        first_source = next(
            (
                source
                for source in env.emitter_ids
                if int(first[f"{vessel}->{source}"]) == 1
            ),
            None,
        )
        items.sort(key=lambda item: (item[0] != first_source, env.emitter_ids.index(item[0])))
        plans[vessel] = _ExecutionPlan(
            initial_active=cargo_t > 1.0,
            initial_target_cargo_t=min(CAPACITY_T, cargo_t + initial_topup_t),
            trips=items,
        )
    return plans


def _allowed_action(env, vessel_index: int, desired: int) -> int:
    return desired if env.vessel_action_mask()[vessel_index][desired] else VESSEL_WAIT


def _coordinated_action(env, plans, tolerance_t: float, ready_source: str):
    state = env.simulator.state
    actions = []
    source_total_t = sum(
        float(state.entity_inventory_t.get(source, 0.0)) for source in env.emitter_ids
    )

    def steal_local_trip(current_vessel: str, source: str):
        for other_vessel, other_plan in plans.items():
            if other_vessel == current_vessel:
                continue
            for item_index, item in enumerate(other_plan.trips):
                if item[0] == source:
                    return other_plan.trips.pop(item_index)
        return None

    for index, vessel in enumerate(env.vessel_ids):
        plan = plans[vessel]
        vessel_state = env.simulator.vessel_states[vessel]
        mode = vessel_state["mode"]
        berth = vessel_state.get("berth")
        cargo_t = float(state.entity_inventory_t.get(vessel, 0.0))
        if mode != "berthed":
            actions.append(VESSEL_WAIT)
            continue

        if plan.initial_active:
            if berth in env.terminal_ids and cargo_t <= tolerance_t:
                plan.initial_active = False
            elif berth in env.terminal_ids:
                actions.append(VESSEL_WAIT)
                continue
            elif berth in env.emitter_ids:
                source_t = float(state.entity_inventory_t.get(str(berth), 0.0))
                if cargo_t + tolerance_t >= plan.initial_target_cargo_t or source_t <= tolerance_t:
                    actions.append(_allowed_action(env, index, VESSEL_GO_TERMINAL))
                else:
                    actions.append(VESSEL_WAIT)
                continue

        if plan.active_trip is not None and berth in env.terminal_ids and cargo_t <= tolerance_t:
            plan.active_trip = None
        if plan.active_trip is None:
            while plan.trips:
                candidate = plan.trips.pop(0)
                if float(state.entity_inventory_t.get(candidate[0], 0.0)) > tolerance_t:
                    plan.active_trip = candidate
                    break
        if (
            plan.active_trip is None
            and berth in env.emitter_ids
            and cargo_t <= tolerance_t
            and float(state.entity_inventory_t.get(str(berth), 0.0)) > tolerance_t
        ):
            plan.active_trip = steal_local_trip(vessel, str(berth))

        if plan.active_trip is not None:
            source, target_t = plan.active_trip
            if berth in env.terminal_ids:
                if cargo_t > tolerance_t:
                    actions.append(VESSEL_WAIT)
                else:
                    actions.append(
                        _allowed_action(env, index, env.vessel_go_emitter_action(source))
                    )
                continue
            if berth == source:
                source_t = float(state.entity_inventory_t.get(source, 0.0))
                if cargo_t + tolerance_t >= target_t or source_t <= tolerance_t:
                    if cargo_t > tolerance_t:
                        actions.append(_allowed_action(env, index, VESSEL_GO_TERMINAL))
                    else:
                        plan.active_trip = None
                        actions.append(VESSEL_WAIT)
                else:
                    actions.append(VESSEL_WAIT)
                continue
            if cargo_t > tolerance_t:
                actions.append(_allowed_action(env, index, VESSEL_GO_TERMINAL))
            else:
                actions.append(
                    _allowed_action(env, index, env.vessel_go_emitter_action(source))
                )
            continue

        if berth in env.terminal_ids:
            if cargo_t > tolerance_t:
                actions.append(VESSEL_WAIT)
            elif source_total_t > tolerance_t:
                # Stay at the terminal as a staging vessel until a reserved trip
                # becomes available; returning early can trigger unwanted loading.
                actions.append(VESSEL_WAIT)
            else:
                actions.append(
                    _allowed_action(env, index, env.vessel_go_emitter_action(ready_source))
                )
        elif cargo_t > tolerance_t:
            actions.append(_allowed_action(env, index, VESSEL_GO_TERMINAL))
        elif float(state.entity_inventory_t.get(str(berth), 0.0)) > tolerance_t:
            # WAIT would auto-load an unreserved vessel.  Leave the source empty
            # instead of stealing cargo reserved for an en-route vessel.
            actions.append(_allowed_action(env, index, VESSEL_GO_TERMINAL))
        else:
            actions.append(VESSEL_WAIT)
    return {
        "vessels": actions,
        "wells": [env.highest_feasible_well_rate_index(well) for well in env.well_ids],
    }


def _coordinated_replay(
    env,
    prediction,
    max_cleanup_h: int,
    tolerance_t: float,
    *,
    disable_new_capture: bool = True,
    disable_disturbances: bool = True,
):
    cleanup_env = copy.deepcopy(env)
    if disable_new_capture:
        _disable_new_capture(cleanup_env)
    if disable_disturbances:
        _disable_cleanup_disturbances(cleanup_env)
    cleanup_env.n_steps = cleanup_env.t + max_cleanup_h + 1
    cleanup_env.config.episode_hours = cleanup_env.n_steps
    plans = _execution_plans(cleanup_env, prediction)
    ready_source = min(
        cleanup_env.emitter_ids,
        key=lambda source: next(
            float(cleanup_env._routes[vessel]["distance_km"])
            for vessel in cleanup_env.vessel_ids
            if str(cleanup_env._routes[vessel]["origin"]) == source
        ),
    )
    start = {
        name: float(getattr(cleanup_env.ledger, name))
        for name in ("vessel_fuel", "conditioning", "reconditioning", "loading", "unloading", "vented_t")
    }
    inventory_clear_h = None
    ready_h = None
    for hour in range(max_cleanup_h + 1):
        if (
            inventory_clear_h is None
            and cleanup_env._in_transit_inventory() <= tolerance_t
            and _queues_empty(cleanup_env)
        ):
            inventory_clear_h = hour
        if (
            inventory_clear_h is not None
            and _vessels_ready_at_sources(cleanup_env, tolerance_t)
            and _queues_empty(cleanup_env)
        ):
            ready_h = hour
            break
        if hour == max_cleanup_h:
            break
        cleanup_env.step(
            _coordinated_action(cleanup_env, plans, tolerance_t, ready_source)
        )
    components = {
        name: float(getattr(cleanup_env.ledger, name)) - start[name]
        for name in ("vessel_fuel", "conditioning", "reconditioning", "loading", "unloading")
    }
    actual_cost = sum(components.values())
    return {
        "coordinated_completed": ready_h is not None,
        "coordinated_inventory_clear_h": inventory_clear_h,
        "coordinated_ready_h": ready_h,
        "coordinated_end_unstored_t": float(cleanup_env._in_transit_inventory()),
        "coordinated_vented_t": float(cleanup_env.ledger.vented_t) - start["vented_t"],
        **{f"coordinated_{name}_eur": value for name, value in components.items()},
        "coordinated_cleanup_cost_eur": actual_cost,
        "cost_difference_eur": actual_cost - float(prediction["trip_cleanup_cost_eur"]),
    }


def _sample_rows(rows):
    policies = [
        "lexicographic",
        "economic",
        "economic_safe",
        "economic_safe_strict",
        "economic_lex_guard",
    ]
    indexed = {
        (int(row["seed"]), row["policy"], int(row["end_hour"])): row for row in rows
    }
    selected = []
    for seed in range(1, 6):
        for offset, hour in enumerate((168, 312, 456, 600)):
            policy = policies[(seed + offset - 1) % len(policies)]
            selected.append(indexed[(seed, policy, hour)])
    return selected


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = _read_csv(args.predictions)
    run_config = json.loads(args.run_config.read_text(encoding="utf-8"))
    economics = EconomicParameters()
    env, sources, vessels, terminal, terminal_leg, direct_leg = _nominal_fuel_hours(args)
    solver_name, solver = _solver(args.solver, 5.0)
    audit = _structural_audit(rows, env.cost_model.parameters)
    ordering = _ordering_tests(
        rows[0],
        sources=sources,
        vessels=vessels,
        terminal=terminal,
        terminal_leg=terminal_leg,
        direct_leg=direct_leg,
        params=env.cost_model.parameters,
        solver=solver,
    )

    replay_rows = []
    for index, row in enumerate(_sample_rows(rows), start=1):
        seed = int(row["seed"])
        policy = str(row["policy"])
        end_hour = int(row["end_hour"])
        action_path = dict(_trace_paths(seed))[policy]
        actions = json.loads(action_path.read_text(encoding="utf-8"))
        replay_env = make_env(argparse.Namespace(**run_config), economics)
        replay_env.reset(seed=seed)
        for action in actions[:end_hour]:
            replay_env.step(action)
        replay = _coordinated_replay(
            replay_env,
            row,
            max_cleanup_h=args.max_cleanup_h,
            tolerance_t=args.tolerance_t,
        )
        replay_rows.append(
            {
                "seed": seed,
                "policy": policy,
                "end_hour": end_hour,
                "trip_cleanup_cost_eur": float(row["trip_cleanup_cost_eur"]),
                "trip_fuel_h": float(row["trip_fuel_h"]),
                **replay,
            }
        )
        print(
            f"replay {index}/20 seed={seed} policy={policy} h={end_hour} "
            f"completed={replay['coordinated_completed']} diff={replay['cost_difference_eur']:.1f}",
            flush=True,
        )
    differences = [float(row["cost_difference_eur"]) for row in replay_rows]
    summary = {
        "solver": solver_name,
        "structural_audit": audit,
        "ordering_tests_passed": sum(bool(row["passed"]) for row in ordering),
        "ordering_tests_total": len(ordering),
        "coordinated_replay_count": len(replay_rows),
        "coordinated_replay_completed": sum(
            bool(row["coordinated_completed"]) for row in replay_rows
        ),
        "coordinated_replay_zero_vent": sum(
            abs(float(row["coordinated_vented_t"])) <= 1e-8 for row in replay_rows
        ),
        "mean_cost_difference_eur": statistics.mean(differences),
        "mae_cost_difference_eur": statistics.mean(abs(value) for value in differences),
        "max_abs_cost_difference_eur": max(abs(value) for value in differences),
    }
    _write_csv(args.output_dir / "ordering_tests.csv", ordering)
    _write_csv(args.output_dir / "coordinated_replay_20states.csv", replay_rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
