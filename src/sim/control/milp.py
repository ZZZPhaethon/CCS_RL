"""Fixed-horizon MILP benchmarks for CCS shipping and injection.

This is a perfect-information benchmark for the same cost objective used by the
online controllers: variable operating cost plus venting and storage-shortfall
penalties. It is an idealized oracle: the terminal is modelled as a fluid buffer
and unloads as scheduled deliveries, so the schedule is optimistic and not
necessarily executable step-for-step in the simulator.

Binding constraints captured (all derived from the same network the env uses):
- injection capacity (pipeline / manifold / wells), the sustained-throughput cap;
- terminal berth unloading capacity;
- per-vessel round-trip cadence and first-arrival startup delay;
- terminal buffer capacity;
- aggregate emitter capture inflow under flexible vessel-to-emitter assignment.

Solved with PuLP + CBC.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..economics import EconomicParameters
from ..entities.emitter import Emitter
from ..entities.manifold import SubseaManifold
from ..entities.pipeline import Pipeline
from ..entities.storage import InjectionWell
from ..entities.terminal import Terminal
from ..entities.vessel import Vessel
from ..scenario_generation import Scenario

KNOTS_TO_KMH = 1.852


@dataclass
class VesselParams:
    vessel_id: str
    source_id: str
    capacity_t: float
    load_dur_h: int
    unload_dur_h: int
    unloading_rate_tph: float
    sail_h: int

    @property
    def round_trip_h(self) -> int:
        return self.load_dur_h + 2 * self.sail_h + self.unload_dur_h

    @property
    def startup_h(self) -> int:
        return self.load_dur_h + self.sail_h


@dataclass
class FixedHorizonMilpResult:
    status: str
    horizon_h: int
    stored_t: float
    deliveries: int
    operating_cost: float
    vented_t: float
    in_transit_t: float
    in_transit_growth_t: float
    shortfall_t: float
    total_cost: float
    cost_per_stored_t: float
    schedule: dict[str, list[int]]
    total_cost_per_stored_t: float = float("nan")
    vessel_fuel: float = 0.0
    conditioning: float = 0.0
    reconditioning: float = 0.0
    loading: float = 0.0
    unloading: float = 0.0
    is_valid: bool = True
    validation_error: str = ""
    max_binary_integrality_violation: float = 0.0


@dataclass(frozen=True)
class StaticSolutionValidation:
    is_valid: bool
    validation_error: str = ""
    max_binary_integrality_violation: float = 0.0


def extract_params(env) -> tuple[list[VesselParams], float, float, float]:
    """Pull MILP parameters from a CCSEnv (injection cap, capture rate, term cap)."""
    network = env.network

    pipelines = [e.max_flow_tph for e in network._entities_of_type(Pipeline).values()]
    manifolds = [e.max_flow_tph for e in network._entities_of_type(SubseaManifold).values()]
    well_sum = sum(e.max_injection_tph for e in network._entities_of_type(InjectionWell).values())
    inj_cap = min([well_sum] + pipelines + manifolds)

    capture_rate = sum(e.nominal_capture_tph for e in network._entities_of_type(Emitter).values())
    term_cap = sum(e.storage_capacity_t for e in network._entities_of_type(Terminal).values())

    vessels: list[VesselParams] = []
    for vid, route in env._routes.items():
        vessel = network.entities[vid]
        assert isinstance(vessel, Vessel)
        sail_h = math.ceil(route["distance_km"] / (route["speed_knots"] * KNOTS_TO_KMH))
        vessels.append(
            VesselParams(
                vessel_id=vid,
                source_id=str(route["origin"]),
                capacity_t=vessel.capacity_t,
                load_dur_h=math.ceil(vessel.capacity_t / vessel.loading_rate_tph),
                unload_dur_h=math.ceil(vessel.capacity_t / vessel.unloading_rate_tph),
                unloading_rate_tph=vessel.unloading_rate_tph,
                sail_h=sail_h,
            )
        )
    return vessels, inj_cap, capture_rate, term_cap


def solve_max_storage_fixed_horizon(
    env,
    horizon_h: int,
    economics: EconomicParameters | None = None,
    initial_buffer_t: float = 0.0,
    time_limit_s: float | None = None,
    mip_gap_rel: float | None = None,
    mip_gap_abs: float | None = None,
    scenario: Scenario | None = None,
    msg: bool = False,
) -> FixedHorizonMilpResult:
    """Cost-minimizing fixed-horizon oracle. The function name is legacy."""
    import pulp

    params = economics or EconomicParameters()
    if scenario is not None:
        return _solve_max_storage_fixed_horizon_with_scenario(
            env,
            horizon_h=int(horizon_h),
            scenario=scenario,
            params=params,
            time_limit_s=time_limit_s,
            mip_gap_rel=mip_gap_rel,
            mip_gap_abs=mip_gap_abs,
            msg=msg,
        )

    vessels, inj_cap, capture_rate, term_cap = extract_params(env)
    source_cap = _source_buffer_capacity(env)
    H = int(horizon_h)
    hours = range(H)

    prob = pulp.LpProblem("cost_fixed_horizon", pulp.LpMinimize)
    d = {
        (v.vessel_id, t): pulp.LpVariable(f"d_{v.vessel_id}_{t}", cat="Binary")
        for v in vessels
        for t in hours
    }
    inj = {t: pulp.LpVariable(f"inj_{t}", lowBound=0, upBound=inj_cap) for t in hours}
    vent = {t: pulp.LpVariable(f"vent_{t}", lowBound=0) for t in hours}
    source_stock = {
        t: pulp.LpVariable(f"source_stock_{t}", lowBound=0, upBound=source_cap)
        for t in hours
    }
    shortfall = pulp.LpVariable("storage_shortfall", lowBound=0)

    for v in vessels:
        for t in hours:
            if t < v.startup_h:
                prob += d[(v.vessel_id, t)] == 0

        for t in hours:
            window = [d[(v.vessel_id, k)] for k in range(t, min(t + v.round_trip_h, H))]
            prob += pulp.lpSum(window) <= 1

    terminal_berths = _terminal_berth_count(env)
    unload_dur = max(v.unload_dur_h for v in vessels)
    for t in hours:
        window = [d[(v.vessel_id, k)] for v in vessels for k in range(t, min(t + unload_dur, H))]
        prob += pulp.lpSum(window) <= terminal_berths

    for t in hours:
        cum_deliv = pulp.lpSum(v.capacity_t * d[(v.vessel_id, k)] for v in vessels for k in range(t + 1))
        cum_inj = pulp.lpSum(inj[k] for k in range(t + 1))
        cum_vent = pulp.lpSum(vent[k] for k in range(t + 1))
        supply_t = initial_buffer_t + capture_rate * (t + 1)
        prob += cum_inj <= cum_deliv
        prob += cum_deliv - cum_inj <= term_cap
        prob += source_stock[t] == supply_t - cum_deliv - cum_vent

    stored_expr = pulp.lpSum(inj.values())
    captured_total_t = initial_buffer_t + capture_rate * H
    prob += shortfall >= env.config.storage_target_rate * captured_total_t - stored_expr
    prob += (
        _schedule_cost_expression(vessels, d, stored_expr, params)
        + pulp.lpSum(vent.values()) * params.carbon_price_eur_per_t
        + shortfall * params.storage_shortfall_eur_per_t
    )
    prob.solve(
        pulp.PULP_CBC_CMD(
            msg=1 if msg else 0,
            timeLimit=time_limit_s,
            gapRel=mip_gap_rel,
            gapAbs=mip_gap_abs,
        )
    )

    status = pulp.LpStatus[prob.status]
    schedule = {
        v.vessel_id: [t for t in hours if round(d[(v.vessel_id, t)].value() or 0) == 1]
        for v in vessels
    }
    n_deliveries = sum(len(s) for s in schedule.values())
    stored_t = float(sum(inj[t].value() or 0.0 for t in hours))
    vented_t = float(sum(vent[t].value() or 0.0 for t in hours))
    captured_from_operations_t = capture_rate * H
    in_transit_growth_t = captured_from_operations_t - stored_t - vented_t
    in_transit_t = max(0.0, initial_buffer_t + in_transit_growth_t)
    shortfall_t = float(shortfall.value() or 0.0)
    cost_breakdown = _schedule_cost_breakdown(vessels, schedule, float(H), stored_t, params)
    cost = sum(cost_breakdown.values())
    total_cost = cost + vented_t * params.carbon_price_eur_per_t + shortfall_t * params.storage_shortfall_eur_per_t
    max_storable_from_deliveries_t = sum(v.capacity_t * len(schedule[v.vessel_id]) for v in vessels)
    validation = _validate_static_solution(
        status=status,
        binary_values=[d[(v.vessel_id, t)].value() for v in vessels for t in hours],
        stored_t=stored_t,
        vented_t=vented_t,
        in_transit_t=in_transit_t,
        captured_from_operations_t=captured_from_operations_t,
        initial_in_transit_t=initial_buffer_t,
        max_storable_from_deliveries_t=max_storable_from_deliveries_t,
    )
    return FixedHorizonMilpResult(
        status=status,
        horizon_h=H,
        stored_t=stored_t,
        deliveries=n_deliveries,
        operating_cost=cost,
        vented_t=vented_t,
        in_transit_t=in_transit_t,
        in_transit_growth_t=in_transit_growth_t,
        shortfall_t=shortfall_t,
        total_cost=total_cost,
        cost_per_stored_t=cost / stored_t if stored_t > 0 else float("nan"),
        schedule=schedule,
        total_cost_per_stored_t=total_cost / stored_t if stored_t > 0 else float("nan"),
        **cost_breakdown,
        is_valid=validation.is_valid,
        validation_error=validation.validation_error,
        max_binary_integrality_violation=validation.max_binary_integrality_violation,
    )


def _solve_max_storage_fixed_horizon_with_scenario(
    env,
    *,
    horizon_h: int,
    scenario: Scenario,
    params: EconomicParameters,
    time_limit_s: float | None,
    mip_gap_rel: float | None,
    mip_gap_abs: float | None,
    msg: bool,
) -> FixedHorizonMilpResult:
    """Cost-minimizing perfect-foresight MILP for one sampled scenario."""
    import pulp

    vessels, nominal_inj_cap, _capture_rate, term_cap = extract_params(env)
    H = int(horizon_h)
    hours = range(H)
    if H <= 0:
        return FixedHorizonMilpResult(
            status="Empty horizon",
            horizon_h=H,
            stored_t=0.0,
            deliveries=0,
            operating_cost=0.0,
            vented_t=0.0,
            in_transit_t=0.0,
            in_transit_growth_t=0.0,
            shortfall_t=0.0,
            total_cost=0.0,
            cost_per_stored_t=float("nan"),
            schedule={v.vessel_id: [] for v in vessels},
            is_valid=True,
        )

    initial_inventory = scenario.initial_inventory_t
    term_init_t = sum(float(initial_inventory.get(tid, 0.0)) for tid in env.terminal_ids)
    source_initial_t = {eid: float(initial_inventory.get(eid, 0.0)) for eid in env.emitter_ids}
    source_supply = {
        eid: _cumulative_capture_supply(env, scenario, eid, H, source_initial_t[eid])
        for eid in env.emitter_ids
    }
    total_supply = [
        sum(source_supply[eid][t] for eid in env.emitter_ids)
        for t in hours
    ]
    source_cap = _source_buffer_capacity(env)
    inj_cap_by_hour = [_scenario_injection_cap(env, scenario, t, nominal_inj_cap) for t in hours]
    berth_count_by_hour = [_scenario_berth_count(env, scenario, t) for t in hours]

    departures = _scenario_departure_options(env, vessels, scenario, H)

    prob = pulp.LpProblem("cost_fixed_horizon_scenario", pulp.LpMinimize)
    d = {
        key: pulp.LpVariable(f"depart_{key[0]}_{key[1]}", cat="Binary")
        for key in departures
    }
    inj = {t: pulp.LpVariable(f"inj_{t}", lowBound=0, upBound=nominal_inj_cap) for t in hours}
    vent = {t: pulp.LpVariable(f"vent_{t}", lowBound=0) for t in hours}
    source_stock = {
        t: pulp.LpVariable(f"source_stock_{t}", lowBound=0, upBound=source_cap)
        for t in hours
    }
    shortfall = pulp.LpVariable("storage_shortfall", lowBound=0)

    for t in hours:
        prob += inj[t] <= inj_cap_by_hour[t]

    for v in vessels:
        vessel_keys = [(vid, depart_t) for (vid, depart_t) in departures if vid == v.vessel_id]
        for left_index, left_key in enumerate(vessel_keys):
            left_ready = departures[left_key]["next_depart_h"]
            for right_key in vessel_keys[left_index + 1:]:
                if int(right_key[1]) < left_ready:
                    prob += d[left_key] + d[right_key] <= 1
                else:
                    break

    unload_dur = max(v.unload_dur_h for v in vessels)
    for t in hours:
        active_unloads = [
            d[key]
            for key, option in departures.items()
            if option["arrival_h"] <= t < option["arrival_h"] + unload_dur
        ]
        if active_unloads:
            prob += pulp.lpSum(active_unloads) <= berth_count_by_hour[t]

    by_id = {v.vessel_id: v for v in vessels}
    unload_profiles = {v.vessel_id: _unload_profile(v) for v in vessels}
    for t in hours:
        unloaded_by_t = [
            _cumulative_unloaded_at(unload_profiles[key[0]], t - option["arrival_h"]) * d[key]
            for key, option in departures.items()
            if t >= option["arrival_h"]
        ]
        loaded_by_t = [
            by_id[key[0]].capacity_t * d[key]
            for key in departures
            if key[1] <= t
        ]
        cum_unload = pulp.lpSum(unloaded_by_t)
        cum_inj = pulp.lpSum(inj[k] for k in range(t + 1))
        cum_loaded = pulp.lpSum(loaded_by_t)
        cum_vent = pulp.lpSum(vent[k] for k in range(t + 1))
        prob += cum_inj <= term_init_t + cum_unload
        prob += term_init_t + cum_unload - cum_inj <= term_cap
        prob += source_stock[t] == total_supply[t] - cum_loaded - cum_vent

    stored_expr = pulp.lpSum(inj.values())
    captured_total_t = total_supply[-1] if total_supply else 0.0
    prob += shortfall >= env.config.storage_target_rate * captured_total_t - stored_expr
    prob += (
        _departure_cost_expression(vessels, d, stored_expr, params)
        + pulp.lpSum(vent.values()) * params.carbon_price_eur_per_t
        + shortfall * params.storage_shortfall_eur_per_t
    )
    prob.solve(
        pulp.PULP_CBC_CMD(
            msg=1 if msg else 0,
            timeLimit=time_limit_s,
            gapRel=mip_gap_rel,
            gapAbs=mip_gap_abs,
        )
    )

    status = pulp.LpStatus[prob.status]
    schedule = {v.vessel_id: [] for v in vessels}
    for key, option in departures.items():
        if round(d[key].value() or 0) == 1:
            schedule[key[0]].append(int(option["arrival_h"]))
    for times in schedule.values():
        times.sort()
    stored_t = float(sum(inj[t].value() or 0.0 for t in hours))
    vented_t = float(sum(vent[t].value() or 0.0 for t in hours))
    source_initial_total_t = sum(source_initial_t.values())
    initial_in_transit_t = source_initial_total_t + term_init_t
    captured_from_operations_t = captured_total_t - source_initial_total_t
    in_transit_growth_t = captured_from_operations_t - stored_t - vented_t
    in_transit_t = max(0.0, initial_in_transit_t + in_transit_growth_t)
    shortfall_t = float(shortfall.value() or 0.0)
    n_deliveries = sum(len(times) for times in schedule.values())
    cost_breakdown = _schedule_cost_breakdown(vessels, schedule, float(H), stored_t, params)
    cost = sum(cost_breakdown.values())
    total_cost = cost + vented_t * params.carbon_price_eur_per_t + shortfall_t * params.storage_shortfall_eur_per_t
    by_id = {v.vessel_id: v for v in vessels}
    max_storable_from_deliveries_t = term_init_t + sum(
        by_id[vessel_id].capacity_t * len(times)
        for vessel_id, times in schedule.items()
    )
    validation = _validate_static_solution(
        status=status,
        binary_values=[d[key].value() for key in departures],
        stored_t=stored_t,
        vented_t=vented_t,
        in_transit_t=in_transit_t,
        captured_from_operations_t=captured_from_operations_t,
        initial_in_transit_t=initial_in_transit_t,
        max_storable_from_deliveries_t=max_storable_from_deliveries_t,
    )
    return FixedHorizonMilpResult(
        status=status,
        horizon_h=H,
        stored_t=stored_t,
        deliveries=n_deliveries,
        operating_cost=cost,
        vented_t=vented_t,
        in_transit_t=in_transit_t,
        in_transit_growth_t=in_transit_growth_t,
        shortfall_t=shortfall_t,
        total_cost=total_cost,
        cost_per_stored_t=cost / stored_t if stored_t > 0 else float("nan"),
        schedule=schedule,
        total_cost_per_stored_t=total_cost / stored_t if stored_t > 0 else float("nan"),
        **cost_breakdown,
        is_valid=validation.is_valid,
        validation_error=validation.validation_error,
        max_binary_integrality_violation=validation.max_binary_integrality_violation,
    )


def _cumulative_capture_supply(
    env,
    scenario: Scenario,
    emitter_id: str,
    horizon_h: int,
    initial_t: float,
) -> list[float]:
    emitter = env.network.entities[emitter_id]
    assert isinstance(emitter, Emitter)
    cumulative = initial_t
    out: list[float] = []
    for t in range(horizon_h):
        availability = _scenario_series_value(scenario.emitter_availability, emitter_id, t, emitter.availability)
        cumulative += emitter.capture_rate_tph_at(float(t)) * max(0.0, float(availability))
        out.append(cumulative)
    return out


def _scenario_injection_cap(env, scenario: Scenario, t: int, nominal_inj_cap: float) -> float:
    well_sum = 0.0
    for well_id, well in env.network._entities_of_type(InjectionWell).items():
        available = bool(_scenario_series_value(scenario.well_available, well_id, t, well.available))
        if not available:
            continue
        injectivity = max(0.0, float(_scenario_series_value(scenario.injectivity_factor, well_id, t, 1.0)))
        well_sum += well.max_injection_tph * injectivity
    return min(nominal_inj_cap, well_sum)


def _scenario_berth_count(env, scenario: Scenario, t: int) -> int:
    return sum(
        max(0, int(terminal.berth_count))
        for terminal in env.network._entities_of_type(Terminal).values()
    )


def _scenario_departure_options(
    env,
    vessels: list[VesselParams],
    scenario: Scenario,
    horizon_h: int,
) -> dict[tuple[str, int], dict[str, int]]:
    options: dict[tuple[str, int], dict[str, int]] = {}
    for v in vessels:
        earliest_depart = v.load_dur_h
        for depart_h in range(earliest_depart, horizon_h):
            arrival_h = _arrival_hour(env, scenario, v.vessel_id, depart_h, horizon_h)
            if arrival_h is None or arrival_h >= horizon_h:
                continue
            return_depart_h = arrival_h + v.unload_dur_h
            home_h = _arrival_hour(env, scenario, v.vessel_id, return_depart_h, horizon_h)
            next_depart_h = horizon_h if home_h is None else home_h + v.load_dur_h
            options[(v.vessel_id, depart_h)] = {
                "arrival_h": arrival_h,
                "next_depart_h": next_depart_h,
            }
    return options


def _validate_static_solution(
    *,
    status: str,
    binary_values,
    stored_t: float,
    vented_t: float,
    in_transit_t: float,
    captured_from_operations_t: float,
    initial_in_transit_t: float,
    max_storable_from_deliveries_t: float,
    integrality_tol: float = 1e-6,
    mass_tol: float = 1e-4,
) -> StaticSolutionValidation:
    values = list(binary_values)
    numeric_values = [float(value) for value in values if value is not None]
    if len(numeric_values) != len(values):
        return StaticSolutionValidation(False, "missing binary variable value", float("inf"))

    max_integrality = 0.0
    if numeric_values:
        max_integrality = max(abs(value - round(value)) for value in numeric_values)

    if status != "Optimal":
        return StaticSolutionValidation(
            False,
            f"solver status {status} is not a validated integer solution",
            max_integrality,
        )
    if max_integrality > integrality_tol:
        return StaticSolutionValidation(
            False,
            f"binary integrality violation {max_integrality:.3g} exceeds tolerance {integrality_tol:g}",
            max_integrality,
        )

    if min(stored_t, vented_t, in_transit_t) < -mass_tol:
        return StaticSolutionValidation(False, "negative mass variable in static MILP result", max_integrality)

    lhs = initial_in_transit_t + captured_from_operations_t
    rhs = stored_t + vented_t + in_transit_t
    balance_tol = max(mass_tol, mass_tol * max(1.0, abs(lhs)))
    if abs(lhs - rhs) > balance_tol:
        return StaticSolutionValidation(
            False,
            f"mass balance violation: input {lhs:.6g} t != output {rhs:.6g} t",
            max_integrality,
        )

    capacity_tol = max(mass_tol, mass_tol * max(1.0, max_storable_from_deliveries_t))
    if stored_t - max_storable_from_deliveries_t > capacity_tol:
        return StaticSolutionValidation(
            False,
            (
                "stored mass exceeds delivered/initial-terminal capacity: "
                f"{stored_t:.6g} t > {max_storable_from_deliveries_t:.6g} t"
            ),
            max_integrality,
        )

    return StaticSolutionValidation(True, "", max_integrality)


def _unload_profile(vessel: VesselParams) -> list[float]:
    profile = []
    remaining = vessel.capacity_t
    for _ in range(vessel.unload_dur_h):
        amount = min(vessel.unloading_rate_tph, remaining)
        profile.append(amount)
        remaining -= amount
    return profile


def _cumulative_unloaded_at(profile: list[float], offset_h: int) -> float:
    if offset_h < 0:
        return 0.0
    return sum(profile[: min(offset_h + 1, len(profile))])


def _arrival_hour(env, scenario: Scenario, vessel_id: str, depart_h: int, horizon_h: int) -> int | None:
    route = env._routes[vessel_id]
    speed_knots = float(route["speed_knots"])
    distance_km = float(route["distance_km"])
    remaining_km = distance_km
    for t in range(max(0, depart_h), horizon_h):
        factor = max(0.0, float(_scenario_series_value(scenario.vessel_speed_factor, vessel_id, t, 1.0)))
        remaining_km -= speed_knots * KNOTS_TO_KMH * factor
        if remaining_km <= 1e-9:
            return t + 1
    return None


def _scenario_series_value(series_by_id, entity_id: str, t: int, default):
    series = series_by_id.get(entity_id)
    if not series:
        return default
    index = max(0, min(len(series) - 1, t))
    return series[index]


def _source_buffer_capacity(env) -> float:
    return sum(
        emitter.buffer_capacity_t
        for emitter in env.network._entities_of_type(Emitter).values()
    )


def _terminal_berth_count(env) -> int:
    return max(
        0,
        sum(int(terminal.berth_count) for terminal in env.network._entities_of_type(Terminal).values()),
    )


def _schedule_cost_expression(
    vessels: list[VesselParams],
    delivery_vars,
    stored_expr,
    params: EconomicParameters,
):
    import pulp

    by_id = {v.vessel_id: v for v in vessels}
    per_delivery = pulp.lpSum(
        (
            2 * by_id[vid].sail_h * params.vessel_fuel_eur_per_h_sailing
            + by_id[vid].capacity_t * params.conditioning_eur_per_t
            + (by_id[vid].load_dur_h + by_id[vid].unload_dur_h) * params.hoteling_fuel_eur_per_h
        )
        * var
        for (vid, _delivery_h), var in delivery_vars.items()
    )
    return per_delivery + stored_expr * params.reconditioning_eur_per_t


def _departure_cost_expression(
    vessels: list[VesselParams],
    departure_vars,
    stored_expr,
    params: EconomicParameters,
):
    import pulp

    by_id = {v.vessel_id: v for v in vessels}
    per_departure = pulp.lpSum(
        (
            2 * by_id[vid].sail_h * params.vessel_fuel_eur_per_h_sailing
            + by_id[vid].capacity_t * params.conditioning_eur_per_t
            + (by_id[vid].load_dur_h + by_id[vid].unload_dur_h) * params.hoteling_fuel_eur_per_h
        )
        * var
        for (vid, _depart_h), var in departure_vars.items()
    )
    return per_departure + stored_expr * params.reconditioning_eur_per_t


def _schedule_cost(
    vessels: list[VesselParams],
    schedule: dict[str, list[int]],
    horizon_h: float,
    stored_t: float,
    params: EconomicParameters,
) -> float:
    return sum(_schedule_cost_breakdown(vessels, schedule, horizon_h, stored_t, params).values())


def _schedule_cost_breakdown(
    vessels: list[VesselParams],
    schedule: dict[str, list[int]],
    horizon_h: float,
    stored_t: float,
    params: EconomicParameters,
) -> dict[str, float]:
    sail_hours = sum(2 * v.sail_h * len(schedule[v.vessel_id]) for v in vessels)
    vessel_fuel = sail_hours * params.vessel_fuel_eur_per_h_sailing
    loaded_t = sum(v.capacity_t * len(schedule[v.vessel_id]) for v in vessels)
    conditioning = loaded_t * params.conditioning_eur_per_t
    reconditioning = stored_t * params.reconditioning_eur_per_t
    loading_h = sum(v.load_dur_h * len(schedule[v.vessel_id]) for v in vessels)
    unloading_h = sum(v.unload_dur_h * len(schedule[v.vessel_id]) for v in vessels)
    loading = loading_h * params.hoteling_fuel_eur_per_h
    unloading = unloading_h * params.hoteling_fuel_eur_per_h
    return {
        "vessel_fuel": vessel_fuel,
        "conditioning": conditioning,
        "reconditioning": reconditioning,
        "loading": loading,
        "unloading": unloading,
    }
