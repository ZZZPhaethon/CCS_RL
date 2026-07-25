"""Evaluate a compact trip-level MILP as a rolling-horizon terminal value.

The model does not expand the cleanup tail hour by hour.  It assigns residual
source stock to integer vessel trips, accounts for cargo already on each
vessel, and prices nominal sailing, conditioning, reconditioning, loading and
unloading. Its predictions are evaluated against the retained deterministic
CCSEnv cleanup labels in ``evaluation_states.csv``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import sys
import time

import pulp

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.rolling_native_mpc_headroom import make_env
from sim.control.cplex_milp import _dynamic_leg_distance_km
from sim.economics import EconomicParameters


CAPACITY_T = 7_500.0
LOAD_RATE_TPH = 800.0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(
            "output/trip_cleanup_terminal_value_500_v5_2026-07-16/evaluation_states.csv"
        ),
    )
    parser.add_argument(
        "--run-config",
        type=Path,
        default=Path(
            "output/rolling_economic_objective_mpc_720h_5seeds_2026-07-15/run_config.json"
        ),
    )
    parser.add_argument("--solver", choices=("auto", "cplex", "cbc"), default="auto")
    parser.add_argument("--time-limit-s", type=float, default=5.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/trip_cleanup_terminal_value_500_v5_2026-07-16"),
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


def _solver(name: str, time_limit_s: float):
    cplex = pulp.CPLEX_CMD(msg=False, timeLimit=time_limit_s)
    if name in {"auto", "cplex"} and cplex.available():
        return "cplex", cplex
    if name == "cplex":
        raise RuntimeError("CPLEX_CMD is not available")
    return "cbc", pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit_s, threads=1)


def _nominal_fuel_hours(args: argparse.Namespace):
    run_config = json.loads(args.run_config.read_text(encoding="utf-8"))
    env = make_env(argparse.Namespace(**run_config), EconomicParameters())
    env.reset(seed=1)
    sources = list(env.emitter_ids)
    vessels = list(env.vessel_ids)
    terminal = env.terminal_ids[0]
    # Cleanup replay uses each vessel's configured cruise speed.  The generic
    # environment default is only an observation fallback and is 12 kn in this
    # experiment, whereas all three vessels are configured at 14 kn.
    speed_kmh = float(env._routes[vessels[0]]["speed_knots"]) * 1.852

    def fuel_hours(distance_km: float) -> int:
        if distance_km <= 1e-9:
            return 0
        return max(0, math.ceil(distance_km / speed_kmh) - 1)

    terminal_leg = {
        source: fuel_hours(float(env._routes[vessels[0]]["distance_km"]))
        for source in sources
    }
    # Each vessel's configured home route supplies the source-terminal distance.
    for vessel_id in vessels:
        route = env._routes[vessel_id]
        terminal_leg[str(route["origin"])] = fuel_hours(float(route["distance_km"]))
    direct_leg = {}
    route = env._routes[vessels[0]]
    for origin in sources:
        for destination in sources:
            distance_km = (
                0.0
                if origin == destination
                else _dynamic_leg_distance_km(env, route, origin, destination)
            )
            direct_leg[(origin, destination)] = fuel_hours(float(distance_km))
    return env, sources, vessels, terminal, terminal_leg, direct_leg


def _terminal_features(env) -> dict[str, float]:
    """Extract the physical terminal-state inputs required by the trip MILP."""
    state = env.simulator.state
    features: dict[str, float] = {}
    source_total_t = 0.0
    cargo_total_t = 0.0
    terminal_total_t = 0.0
    for source in env.emitter_ids:
        stock_t = float(state.entity_inventory_t.get(source, 0.0))
        features[f"f_source_t__{source}"] = stock_t
        source_total_t += stock_t
    for vessel in env.vessel_ids:
        cargo_t = float(state.entity_inventory_t.get(vessel, 0.0))
        vessel_state = env.simulator.vessel_states[vessel]
        mode = str(vessel_state["mode"])
        remaining_h = 0.0
        if mode == "sailing":
            remaining_km = max(
                0.0,
                (1.0 - float(vessel_state.get("progress", 0.0)))
                * float(vessel_state.get("distance_km", 0.0)),
            )
            speed_knots = float(env._routes[vessel]["speed_knots"])
            remaining_h = remaining_km / max(1e-9, speed_knots * 1.852)
        features[f"f_cargo_t__{vessel}"] = cargo_t
        features[f"f_vessel_remaining_sail_h__{vessel}"] = remaining_h
        for node in (*env.emitter_ids, *env.terminal_ids):
            features[f"f_vessel_at__{vessel}__{node}"] = float(
                mode == "berthed" and vessel_state.get("berth") == node
            )
            features[f"f_vessel_sailing_to__{vessel}__{node}"] = float(
                mode == "sailing" and vessel_state.get("destination") == node
            )
        cargo_total_t += cargo_t
    for terminal in env.terminal_ids:
        terminal_total_t += float(state.entity_inventory_t.get(terminal, 0.0))
    features["f_source_total_t"] = source_total_t
    features["f_cargo_total_t"] = cargo_total_t
    features["f_terminal_total_t"] = terminal_total_t
    features["f_unstored_total_t"] = float(env._in_transit_inventory())
    return features


def _state_node(
    row: dict[str, str], vessel_id: str, nodes: list[str]
) -> tuple[str, str]:
    for node_id in nodes:
        if float(row[f"f_vessel_at__{vessel_id}__{node_id}"]) > 0.5:
            return "berthed", node_id
    for node_id in nodes:
        if float(row[f"f_vessel_sailing_to__{vessel_id}__{node_id}"]) > 0.5:
            return "sailing", node_id
    raise ValueError(f"Cannot locate {vessel_id} in terminal-state row")


def _predict_one(
    row: dict[str, str],
    *,
    sources: list[str],
    vessels: list[str],
    terminal: str,
    terminal_leg: dict[str, int],
    direct_leg: dict[tuple[str, str], int],
    params: EconomicParameters,
    solver,
) -> dict[str, object]:
    nodes = [*sources, terminal]
    source_stock = {source: float(row[f"f_source_t__{source}"]) for source in sources}
    cargo = {vessel: float(row[f"f_cargo_t__{vessel}"]) for vessel in vessels}
    vessel_state = {
        vessel: _state_node(row, vessel, nodes) for vessel in vessels
    }
    remaining_fuel_h = {
        vessel: (
            max(0, math.ceil(float(row[f"f_vessel_remaining_sail_h__{vessel}"]) - 1e-9))
            if vessel_state[vessel][0] == "sailing"
            else 0
        )
        for vessel in vessels
    }
    max_trips = max(1, math.ceil(sum(source_stock.values()) / CAPACITY_T) + 1)

    problem = pulp.LpProblem("compact_trip_cleanup", pulp.LpMinimize)
    shipped = {
        (vessel, source): pulp.LpVariable(
            f"shipped__{vessel}__{source}", lowBound=0.0
        )
        for vessel in vessels
        for source in sources
    }
    trips = {
        (vessel, source): pulp.LpVariable(
            f"trips__{vessel}__{source}", lowBound=0, upBound=max_trips, cat="Integer"
        )
        for vessel in vessels
        for source in sources
    }
    use = {
        vessel: pulp.LpVariable(f"use__{vessel}", cat="Binary") for vessel in vessels
    }
    first = {
        (vessel, source): pulp.LpVariable(
            f"first__{vessel}__{source}", cat="Binary"
        )
        for vessel in vessels
        for source in sources
    }
    topup = {}
    for vessel in vessels:
        mode, node = vessel_state[vessel]
        for source in sources:
            upper = 0.0
            if cargo[vessel] > 1.0 and node == source:
                upper = max(0.0, CAPACITY_T - cargo[vessel])
            topup[(vessel, source)] = pulp.LpVariable(
                f"topup__{vessel}__{source}", lowBound=0.0, upBound=upper
            )

    for source in sources:
        problem += (
            pulp.lpSum(topup[(vessel, source)] + shipped[(vessel, source)] for vessel in vessels)
            == source_stock[source]
        )
    for vessel in vessels:
        for source in sources:
            problem += shipped[(vessel, source)] <= CAPACITY_T * trips[(vessel, source)]
            problem += first[(vessel, source)] <= trips[(vessel, source)]
        total_trips = pulp.lpSum(trips[(vessel, source)] for source in sources)
        problem += total_trips <= max_trips * use[vessel]
        problem += total_trips >= use[vessel]
        mode, node = vessel_state[vessel]
        can_start_from_source = cargo[vessel] <= 1.0 and node in sources
        if can_start_from_source:
            problem += pulp.lpSum(first[(vessel, source)] for source in sources) == use[vessel]
        else:
            problem += pulp.lpSum(first[(vessel, source)] for source in sources) == 0

    fixed_fuel_h = 0.0
    final_ready_source = min(sources, key=lambda source: terminal_leg[source])
    ready_fuel_h = terminal_leg[final_ready_source]
    first_adjustment = []
    final_return = []
    for vessel in vessels:
        mode, node = vessel_state[vessel]
        fixed_fuel_h += remaining_fuel_h[vessel]
        if cargo[vessel] > 1.0:
            if node in sources:
                fixed_fuel_h += terminal_leg[node]
            fixed_fuel_h += ready_fuel_h
        elif node == terminal:
            fixed_fuel_h += ready_fuel_h
        else:
            final_return.append(ready_fuel_h * use[vessel])
            for source in sources:
                first_adjustment.append(
                    (direct_leg[(node, source)] - terminal_leg[source])
                    * first[(vessel, source)]
                )

    trip_fuel_h = pulp.lpSum(
        2.0 * terminal_leg[source] * trips[(vessel, source)]
        for vessel in vessels
        for source in sources
    )
    fuel_h_expr = fixed_fuel_h + trip_fuel_h + pulp.lpSum(first_adjustment + final_return)
    # The tiny secondary term forces maximum use of already-scheduled cargo capacity
    # when several allocations have identical sailing hours.
    problem += (
        fuel_h_expr
        - 1e-8 * pulp.lpSum(topup.values())
        - 1e-4 * pulp.lpSum(first.values())
    )
    started = time.perf_counter()
    problem.solve(solver)
    solve_time_s = time.perf_counter() - started
    status = pulp.LpStatus.get(problem.status, str(problem.status))
    if status != "Optimal":
        raise RuntimeError(f"Compact trip cleanup returned {status}")

    fuel_h = float(pulp.value(fuel_h_expr))
    source_total_t = sum(source_stock.values())
    cargo_total_t = sum(cargo.values())
    unstored_total_t = float(row["f_unstored_total_t"])
    conditioning = params.conditioning_eur_per_t * source_total_t
    reconditioning = params.reconditioning_eur_per_t * unstored_total_t
    loading = params.hoteling_fuel_eur_per_h * source_total_t / LOAD_RATE_TPH
    unloading = (
        params.hoteling_fuel_eur_per_h * (source_total_t + cargo_total_t) / LOAD_RATE_TPH
    )
    vessel_fuel = fuel_h * params.vessel_fuel_eur_per_h_sailing
    operating_cost = vessel_fuel + conditioning + reconditioning + loading + unloading
    shipped_values = {
        f"{vessel}->{source}": float(pulp.value(shipped[(vessel, source)]))
        for vessel in vessels
        for source in sources
    }
    trip_values = {
        f"{vessel}->{source}": int(round(float(pulp.value(trips[(vessel, source)]))))
        for vessel in vessels
        for source in sources
    }
    topup_values = {
        f"{vessel}->{source}": float(pulp.value(topup[(vessel, source)]))
        for vessel in vessels
        for source in sources
    }
    first_values = {
        f"{vessel}->{source}": int(round(float(pulp.value(first[(vessel, source)]))))
        for vessel in vessels
        for source in sources
    }
    return {
        "trip_status": status,
        "trip_solve_time_s": solve_time_s,
        "trip_variable_count": len(problem.variables()),
        "trip_constraint_count": len(problem.constraints),
        "trip_fuel_h": fuel_h,
        "trip_vessel_fuel_eur": vessel_fuel,
        "trip_conditioning_eur": conditioning,
        "trip_reconditioning_eur": reconditioning,
        "trip_loading_eur": loading,
        "trip_unloading_eur": unloading,
        "trip_cleanup_cost_eur": operating_cost,
        "trip_shipped_json": json.dumps(shipped_values, sort_keys=True),
        "trip_counts_json": json.dumps(trip_values, sort_keys=True),
        "trip_topup_json": json.dumps(topup_values, sort_keys=True),
        "trip_first_json": json.dumps(first_values, sort_keys=True),
    }


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = list(_read_csv(args.dataset))
    env, sources, vessels, terminal, terminal_leg, direct_leg = _nominal_fuel_hours(args)
    solver_name, solver = _solver(args.solver, args.time_limit_s)
    params = env.cost_model.parameters
    for index, row in enumerate(rows, start=1):
        row.update(
            _predict_one(
                row,
                sources=sources,
                vessels=vessels,
                terminal=terminal,
                terminal_leg=terminal_leg,
                direct_leg=direct_leg,
                params=params,
                solver=solver,
            )
        )
        if index % 25 == 0 or index == len(rows):
            print(f"solved {index}/{len(rows)}", flush=True)
    _write_csv(args.output_dir / "predictions.csv", rows)
    metadata = {
        "solver": solver_name,
        "time_limit_s": args.time_limit_s,
        "terminal_fuel_hours": terminal_leg,
        "direct_source_fuel_hours": {
            f"{origin}->{destination}": value
            for (origin, destination), value in direct_leg.items()
        },
        "mean_solve_time_s": statistics.mean(
            float(row["trip_solve_time_s"]) for row in rows
        ),
        "max_solve_time_s": max(float(row["trip_solve_time_s"]) for row in rows),
        "variable_count": int(rows[0]["trip_variable_count"]),
        "constraint_count": int(rows[0]["trip_constraint_count"]),
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
