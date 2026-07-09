"""Trip-based CPLEX MILP oracles for CCS shipping schedules.

This module keeps the trip formulations separate from the hourly native-action
MILP in :mod:`sim.control.cplex_milp`.

Two levels are exposed:

* ``solve_relaxed_trip_milp_with_cplex`` treats each selected trip as an
  aggregate shipped mass. It is an optimistic theoretical oracle.
* ``solve_executable_trip_milp_with_cplex`` selects path-continuous executable
  trip templates and expands them into the same hourly native actions used by RL.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

try:
    import pulp
except ImportError:  # pragma: no cover - exercised only in minimal installs
    pulp = None

from ..economics import EconomicParameters
from ..entities.emitter import Emitter
from ..entities.manifold import SubseaManifold
from ..entities.pipeline import Pipeline
from ..entities.storage import InjectionWell
from ..entities.terminal import Terminal
from ..entities.vessel import Vessel
from ..environment import VESSEL_GO_TERMINAL, VESSEL_WAIT, WELL_RATE_LEVELS_MTPA
from ..operations.pressure_limits import mtpa_to_tph
from ..scenario_generation import Scenario
from .cplex_milp import (
    CplexMilpReplayResult,
    _add_single_well_dynamic_bhp_constraints,
    _capture_tonnes,
    _current_start_hour,
    _make_cplex_cmd,
    _path_start,
    _scenario_series_value,
    _sail_hours_between,
    _solution_status_label,
    _terminal_berth_counts,
    _value,
    _well_rate_options_by_hour,
    replay_full_scenario_cplex_plan,
)

@dataclass(frozen=True)
class TripRecord:
    vessel_id: str
    emitter_id: str
    load_start_h: int
    depart_h: int
    arrival_h: int
    return_start_h: int
    end_h: int
    amount_t: float


@dataclass(frozen=True)
class TripMilpResult:
    level: str
    status: str
    horizon_h: int
    stored_t: float
    vented_t: float
    in_transit_t: float
    in_transit_growth_t: float
    shortfall_t: float
    deliveries: int
    trips: list[TripRecord]
    injection_tph: list[float]
    vessel_actions_by_hour: dict[str, list[int]]
    well_rate_indices_by_hour: dict[str, list[int]]
    native_actions_by_hour: list[dict[str, list[int]]]
    operating_cost: float
    total_cost: float
    cost_per_stored_t: float
    total_cost_per_stored_t: float
    storage_reward_eur_per_t: float = 0.0
    net_reward: float = 0.0
    objective_value: float = 0.0
    vessel_fuel: float = 0.0
    conditioning: float = 0.0
    reconditioning: float = 0.0
    loading: float = 0.0
    unloading: float = 0.0
    is_valid: bool = True
    validation_error: str = ""
    max_binary_integrality_violation: float = 0.0
    binary_count: int = 0
    variable_count: int = 0
    constraint_count: int = 0


@dataclass(frozen=True)
class _TripOption:
    vessel_id: str
    emitter_id: str
    load_start_h: int
    depart_h: int
    arrival_h: int
    return_start_h: int
    end_h: int
    capacity_t: float
    load_rate_tph: float
    unload_rate_tph: float
    outbound_sail_h: int
    return_sail_h: int
    load_profile_t: tuple[float, ...] = ()
    unload_profile_t: tuple[float, ...] = ()

    @property
    def amount_t(self) -> float:
        return sum(self.load_profile_t) if self.load_profile_t else self.capacity_t

    @property
    def sail_fuel_h(self) -> int:
        return max(0, self.outbound_sail_h - 1) + max(0, self.return_sail_h - 1)


@dataclass(frozen=True)
class _Validation:
    is_valid: bool
    validation_error: str
    max_binary_integrality_violation: float


def solve_relaxed_trip_milp_with_cplex(
    env,
    *,
    scenario: Scenario | None = None,
    horizon_h: int | None = None,
    economics: EconomicParameters | None = None,
    storage_reward_eur_per_t: float | None = None,
    cplex_path: str | None = None,
    time_limit_s: float | None = None,
    mip_gap_rel: float | None = None,
    mip_gap_abs: float | None = None,
    threads: int | None = None,
    msg: bool = False,
) -> TripMilpResult:
    """Solve an optimistic aggregate-flow trip MILP with external CPLEX."""

    _require_ready(env, scenario)
    scenario = scenario or env.scenario
    params = economics or EconomicParameters()
    reward_per_t = _storage_reward_eur_per_t(env, storage_reward_eur_per_t)
    H = _horizon(env, scenario, horizon_h)
    if H <= 0:
        return _empty_result("relaxed_trip", H)

    start_step = scenario.step_index(_current_start_hour(env))
    hours = range(H)
    state = env.simulator.state
    terminal_capacity_t = _terminal_capacity_t(env)
    options = _relaxed_trip_options(env, scenario, start_step, H)

    prob = pulp.LpProblem("relaxed_trip_cplex_milp", pulp.LpMinimize)
    choose = {i: pulp.LpVariable(f"trip_{i}", cat="Binary") for i in range(len(options))}
    shipped = {
        i: pulp.LpVariable(f"trip_mass_{i}", lowBound=0.0, upBound=options[i].capacity_t)
        for i in range(len(options))
    }
    source_stock, source_ready, source_overflow, vent = _source_variables(env, H)
    terminal_stock = _terminal_stock_variables(H, terminal_capacity_t)
    well_inj = _well_injection_variables(env, H)

    for i, option in enumerate(options):
        prob += shipped[i] <= option.capacity_t * choose[i]

    _add_relaxed_vessel_occupancy_constraints(prob, options, choose, env, H)
    _add_relaxed_berth_constraints(prob, options, choose, env, scenario, start_step, H)
    _add_source_balance_constraints(
        prob,
        env,
        scenario,
        start_step,
        H,
        source_stock,
        source_ready,
        source_overflow,
        vent,
        load_at=lambda emitter_id, t: pulp.lpSum(
            shipped[i]
            for i, option in enumerate(options)
            if option.emitter_id == emitter_id and option.load_start_h == t
        ),
    )
    _add_relaxed_terminal_and_injection_constraints(
        prob,
        env,
        scenario,
        start_step,
        H,
        terminal_stock,
        well_inj,
        delivered_at=lambda t: pulp.lpSum(
            shipped[i] for i, option in enumerate(options) if option.arrival_h == t
        ),
    )
    _add_single_well_dynamic_bhp_constraints(prob, env, well_inj, H)

    captured_from_operations_t = _captured_from_operations(env, scenario, start_step, H)
    stored_expr = pulp.lpSum(well_inj[(well_id, t)] for well_id in env.well_ids for t in hours)

    prob += (
        _relaxed_trip_cost_expression(env, options, choose, shipped, stored_expr, params)
        + pulp.lpSum(vent[(emitter_id, t)] for emitter_id in env.emitter_ids for t in hours)
        * params.carbon_price_eur_per_t
        - stored_expr * reward_per_t
    )
    _solve(prob, cplex_path, time_limit_s, mip_gap_rel, mip_gap_abs, threads, msg)
    status = _solution_status_label(prob.status, getattr(prob, "sol_status", None))

    selected = [
        _record_from_option(option, _value(shipped[i]))
        for i, option in enumerate(options)
        if round(_value(choose[i])) == 1 and _value(shipped[i]) > 1e-7
    ]
    injection_tph = [
        sum(_value(well_inj[(well_id, t)]) for well_id in env.well_ids)
        for t in hours
    ]
    stored_t = sum(injection_tph)
    vented_t = sum(_value(vent[(emitter_id, t)]) for emitter_id in env.emitter_ids for t in hours)
    shortfall_t = _storage_shortfall_t(env, captured_from_operations_t, stored_t)
    initial_in_transit_t = _initial_in_transit_t(env)
    in_transit_t = initial_in_transit_t + captured_from_operations_t - stored_t - vented_t
    cost = _relaxed_trip_cost_breakdown(env, options, choose, shipped, stored_t, params)
    total_cost = cost["operating_cost"] + vented_t * params.carbon_price_eur_per_t
    net_reward = reward_per_t * stored_t - total_cost
    validation = _validate_trip_solution(
        status,
        [choose[i].value() for i in choose],
        stored_t=stored_t,
        vented_t=vented_t,
        in_transit_t=in_transit_t,
        captured_from_operations_t=captured_from_operations_t,
        initial_in_transit_t=initial_in_transit_t,
        max_storable_from_deliveries_t=_initial_terminal_t(env) + sum(trip.amount_t for trip in selected),
    )
    return _result(
        level="relaxed_trip",
        status=status,
        horizon_h=H,
        stored_t=stored_t,
        vented_t=vented_t,
        in_transit_t=in_transit_t,
        initial_in_transit_t=initial_in_transit_t,
        captured_from_operations_t=captured_from_operations_t,
        shortfall_t=shortfall_t,
        trips=selected,
        injection_tph=injection_tph,
        vessel_actions_by_hour={},
        well_rate_indices_by_hour={},
        native_actions_by_hour=[],
        cost=cost,
        total_cost=total_cost,
        storage_reward_eur_per_t=reward_per_t,
        net_reward=net_reward,
        validation=validation,
        binary_count=len(choose),
        variable_count=len(prob.variables()),
        constraint_count=len(prob.constraints),
    )


def solve_executable_trip_milp_with_cplex(
    env,
    *,
    scenario: Scenario | None = None,
    horizon_h: int | None = None,
    economics: EconomicParameters | None = None,
    storage_reward_eur_per_t: float | None = None,
    cplex_path: str | None = None,
    time_limit_s: float | None = None,
    mip_gap_rel: float | None = None,
    mip_gap_abs: float | None = None,
    threads: int | None = None,
    msg: bool = False,
) -> TripMilpResult:
    """Solve path-continuous executable trip templates and expand them to RL actions."""

    _require_ready(env, scenario)
    scenario = scenario or env.scenario
    params = economics or EconomicParameters()
    reward_per_t = _storage_reward_eur_per_t(env, storage_reward_eur_per_t)
    H = _horizon(env, scenario, horizon_h)
    if H <= 0:
        return _empty_result("executable_trip", H)

    start_step = scenario.step_index(_current_start_hour(env))
    hours = range(H)
    terminal_capacity_t = _terminal_capacity_t(env)
    options = _executable_trip_options(env, scenario, start_step, H)
    well_rate_options = _well_rate_options_by_hour(env, scenario, H)

    prob = pulp.LpProblem("executable_trip_cplex_milp", pulp.LpMinimize)
    choose = {i: pulp.LpVariable(f"trip_{i}", cat="Binary") for i in range(len(options))}
    source_stock, source_ready, source_overflow, vent = _source_variables(env, H)
    terminal_stock = _terminal_stock_variables(H, terminal_capacity_t)
    well_choice = {
        (well_id, t, rate_index): pulp.LpVariable(f"well_{well_id}_{t}_{rate_index}", cat="Binary")
        for well_id in env.well_ids
        for t in hours
        for rate_index in well_rate_options[(well_id, t)]
    }
    well_inj = _well_injection_variables(env, H)
    injection_limit_choice = {
        (t, limit): pulp.LpVariable(f"injection_limit_{t}_{limit}", cat="Binary")
        for t in hours
        for limit in range(2)
    }

    _add_executable_vessel_and_berth_constraints(prob, options, choose, env, scenario, start_step, H)
    _add_executable_trip_path_constraints(prob, options, choose, env, scenario, start_step, H)
    _add_source_balance_constraints(
        prob,
        env,
        scenario,
        start_step,
        H,
        source_stock,
        source_ready,
        source_overflow,
        vent,
        load_at=lambda emitter_id, t: pulp.lpSum(
            amount * choose[i]
            for i, option in enumerate(options)
            for offset, amount in enumerate(option.load_profile_t)
            if option.emitter_id == emitter_id and option.load_start_h + offset == t
        ),
    )
    _add_executable_terminal_and_injection_constraints(
        prob,
        env,
        scenario,
        start_step,
        H,
        terminal_stock,
        options,
        choose,
        well_rate_options,
        well_choice,
        well_inj,
        injection_limit_choice,
    )
    _add_single_well_dynamic_bhp_constraints(prob, env, well_inj, H)

    captured_from_operations_t = _captured_from_operations(env, scenario, start_step, H)
    stored_expr = pulp.lpSum(well_inj[(well_id, t)] for well_id in env.well_ids for t in hours)
    prob += (
        _executable_trip_cost_expression(options, choose, stored_expr, params)
        + pulp.lpSum(vent[(emitter_id, t)] for emitter_id in env.emitter_ids for t in hours)
        * params.carbon_price_eur_per_t
        - stored_expr * reward_per_t
    )

    _solve(prob, cplex_path, time_limit_s, mip_gap_rel, mip_gap_abs, threads, msg)
    status = _solution_status_label(prob.status, getattr(prob, "sol_status", None))
    selected = [
        _record_from_option(option, option.amount_t)
        for i, option in enumerate(options)
        if round(_value(choose[i])) == 1
    ]
    selected.sort(key=lambda trip: (trip.vessel_id, trip.load_start_h))
    injection_tph = [
        sum(_value(well_inj[(well_id, t)]) for well_id in env.well_ids)
        for t in hours
    ]
    well_rate_indices_by_hour = _extract_well_rate_indices(env, H, well_rate_options, well_choice)
    vessel_actions_by_hour = _expand_executable_trips_to_actions(
        env,
        scenario,
        start_step,
        H,
        options,
        choose,
    )
    native_actions_by_hour = [
        {
            "vessels": [vessel_actions_by_hour[vessel_id][t] for vessel_id in env.vessel_ids],
            "wells": [well_rate_indices_by_hour[well_id][t] for well_id in env.well_ids],
        }
        for t in hours
    ]
    stored_t = sum(injection_tph)
    vented_t = sum(_value(vent[(emitter_id, t)]) for emitter_id in env.emitter_ids for t in hours)
    shortfall_t = _storage_shortfall_t(env, captured_from_operations_t, stored_t)
    initial_in_transit_t = _initial_in_transit_t(env)
    in_transit_t = initial_in_transit_t + captured_from_operations_t - stored_t - vented_t
    cost = _executable_trip_cost_breakdown(env, scenario, start_step, H, options, choose, stored_t, params)
    total_cost = cost["operating_cost"] + vented_t * params.carbon_price_eur_per_t
    net_reward = reward_per_t * stored_t - total_cost
    validation = _validate_trip_solution(
        status,
        [
            *[choose[i].value() for i in choose],
            *[var.value() for var in source_overflow.values()],
            *[var.value() for var in well_choice.values()],
            *[var.value() for var in injection_limit_choice.values()],
        ],
        stored_t=stored_t,
        vented_t=vented_t,
        in_transit_t=in_transit_t,
        captured_from_operations_t=captured_from_operations_t,
        initial_in_transit_t=initial_in_transit_t,
        max_storable_from_deliveries_t=_initial_terminal_t(env)
        + sum(
            sum(
                amount
                for offset, amount in enumerate(option.unload_profile_t)
                if option.arrival_h + offset < H
            )
            for i, option in enumerate(options)
            if round(_value(choose[i])) == 1
        ),
    )
    return _result(
        level="executable_trip",
        status=status,
        horizon_h=H,
        stored_t=stored_t,
        vented_t=vented_t,
        in_transit_t=in_transit_t,
        initial_in_transit_t=initial_in_transit_t,
        captured_from_operations_t=captured_from_operations_t,
        shortfall_t=shortfall_t,
        trips=selected,
        injection_tph=injection_tph,
        vessel_actions_by_hour=vessel_actions_by_hour,
        well_rate_indices_by_hour=well_rate_indices_by_hour,
        native_actions_by_hour=native_actions_by_hour,
        cost=cost,
        total_cost=total_cost,
        storage_reward_eur_per_t=reward_per_t,
        net_reward=net_reward,
        validation=validation,
        binary_count=len(choose) + len(source_overflow) + len(well_choice) + len(injection_limit_choice),
        variable_count=len(prob.variables()),
        constraint_count=len(prob.constraints),
    )


def replay_trip_milp_plan(env, result: TripMilpResult, *, stored_tol_t: float = 1e-6) -> CplexMilpReplayResult:
    return replay_full_scenario_cplex_plan(env, result, stored_tol_t=stored_tol_t)


def _relaxed_trip_options(env, scenario: Scenario, start_step: int, horizon_h: int) -> list[_TripOption]:
    options: list[_TripOption] = []
    for vessel_id in env.vessel_ids:
        vessel = env.network.entities[vessel_id]
        assert isinstance(vessel, Vessel)
        terminal_id = str(env._routes[vessel_id]["destination"])
        for emitter_id in env.emitter_ids:
            load_rate = min(vessel.loading_rate_tph, env.network.entities[emitter_id].loading_rate_tph)
            for load_start_h in range(horizon_h):
                outbound_h = _sail_hours_between(
                    env,
                    emitter_id,
                    terminal_id,
                    vessel_id,
                    scenario=scenario,
                    start_step=start_step + load_start_h,
                    max_horizon_h=horizon_h - load_start_h,
                )
                arrival_h = load_start_h + outbound_h
                if arrival_h >= horizon_h:
                    continue
                return_start_h = arrival_h + 1
                return_h = _sail_hours_between(
                    env,
                    terminal_id,
                    emitter_id,
                    vessel_id,
                    scenario=scenario,
                    start_step=start_step + return_start_h,
                    max_horizon_h=max(1, horizon_h - return_start_h),
                )
                end_h = min(horizon_h, return_start_h + return_h)
                options.append(
                    _TripOption(
                        vessel_id=vessel_id,
                        emitter_id=emitter_id,
                        load_start_h=load_start_h,
                        depart_h=load_start_h,
                        arrival_h=arrival_h,
                        return_start_h=return_start_h,
                        end_h=max(return_start_h, end_h),
                        capacity_t=float(vessel.capacity_t),
                        load_rate_tph=float(load_rate),
                        unload_rate_tph=float(vessel.unloading_rate_tph),
                        outbound_sail_h=outbound_h,
                        return_sail_h=return_h,
                    )
                )
    return options


def _executable_trip_options(env, scenario: Scenario, start_step: int, horizon_h: int) -> list[_TripOption]:
    options: list[_TripOption] = []
    for vessel_id in env.vessel_ids:
        vessel = env.network.entities[vessel_id]
        assert isinstance(vessel, Vessel)
        terminal_id = str(env._routes[vessel_id]["destination"])
        for emitter_id in env.emitter_ids:
            load_rate = min(vessel.loading_rate_tph, env.network.entities[emitter_id].loading_rate_tph)
            max_load_hours = max(1, math.ceil(vessel.capacity_t / max(1e-9, load_rate)))
            for load_hours in range(1, max_load_hours + 1):
                load_profile = _bounded_profile(vessel.capacity_t, load_rate, load_hours)
                amount_t = sum(load_profile)
                unload_hours = max(1, math.ceil(amount_t / max(1e-9, vessel.unloading_rate_tph)))
                unload_profile = _bounded_profile(amount_t, vessel.unloading_rate_tph, unload_hours)
                for load_start_h in range(horizon_h):
                    depart_h = load_start_h + load_hours
                    if depart_h >= horizon_h:
                        continue
                    outbound_h = _sail_hours_between(
                        env,
                        emitter_id,
                        terminal_id,
                        vessel_id,
                        scenario=scenario,
                        start_step=start_step + depart_h,
                        max_horizon_h=horizon_h - depart_h,
                    )
                    arrival_h = depart_h + outbound_h
                    if arrival_h >= horizon_h:
                        continue
                    return_start_h = arrival_h + unload_hours
                    if return_start_h > horizon_h:
                        continue
                    options.append(
                        _TripOption(
                            vessel_id=vessel_id,
                            emitter_id=emitter_id,
                            load_start_h=load_start_h,
                            depart_h=depart_h,
                            arrival_h=arrival_h,
                            return_start_h=return_start_h,
                            end_h=return_start_h,
                            capacity_t=float(vessel.capacity_t),
                            load_rate_tph=float(load_rate),
                            unload_rate_tph=float(vessel.unloading_rate_tph),
                            outbound_sail_h=outbound_h,
                            return_sail_h=0,
                            load_profile_t=tuple(load_profile),
                            unload_profile_t=tuple(unload_profile),
                        )
                    )
    return options


def _bounded_profile(total_t: float, rate_tph: float, hours: int) -> list[float]:
    remaining = max(0.0, float(total_t))
    profile: list[float] = []
    for _ in range(hours):
        amount = min(float(rate_tph), remaining)
        profile.append(amount)
        remaining -= amount
    return profile


def _add_relaxed_vessel_occupancy_constraints(prob, options, choose, env, horizon_h: int) -> None:
    for vessel_id in env.vessel_ids:
        vessel_options = [
            (i, option)
            for i, option in enumerate(options)
            if option.vessel_id == vessel_id
        ]
        for t in range(horizon_h):
            active = [choose[i] for i, option in vessel_options if option.load_start_h <= t < option.end_h]
            if active:
                prob += pulp.lpSum(active) <= 1


def _add_relaxed_berth_constraints(
    prob,
    options,
    choose,
    env,
    scenario: Scenario,
    start_step: int,
    horizon_h: int,
) -> None:
    for t in range(horizon_h):
        berth_counts = _terminal_berth_counts(env, scenario, start_step + t)
        for terminal_id, berth_count in berth_counts.items():
            vessels_for_terminal = [
                vessel_id
                for vessel_id in env.vessel_ids
                if str(env._routes[vessel_id]["destination"]) == terminal_id
            ]
            arrivals = [
                choose[i]
                for i, option in enumerate(options)
                if option.vessel_id in vessels_for_terminal and option.arrival_h == t
            ]
            if arrivals:
                prob += pulp.lpSum(arrivals) <= min(1, berth_count)


def _add_executable_vessel_and_berth_constraints(
    prob,
    options,
    choose,
    env,
    scenario: Scenario,
    start_step: int,
    horizon_h: int,
) -> None:
    for vessel_id in env.vessel_ids:
        vessel_options = [(i, option) for i, option in enumerate(options) if option.vessel_id == vessel_id]
        for t in range(horizon_h):
            active = [choose[i] for i, option in vessel_options if option.load_start_h <= t < option.end_h]
            if active:
                prob += pulp.lpSum(active) <= 1

    for emitter_id in env.emitter_ids:
        for t in range(horizon_h):
            active_loads = [
                choose[i]
                for i, option in enumerate(options)
                if option.emitter_id == emitter_id
                and option.load_start_h <= t < option.depart_h
            ]
            if active_loads:
                prob += pulp.lpSum(active_loads) <= 1

    for t in range(horizon_h):
        berth_counts = _terminal_berth_counts(env, scenario, start_step + t)
        for terminal_id, berth_count in berth_counts.items():
            vessels_for_terminal = [
                vessel_id
                for vessel_id in env.vessel_ids
                if str(env._routes[vessel_id]["destination"]) == terminal_id
            ]
            active_unloads = [
                choose[i]
                for i, option in enumerate(options)
                if option.vessel_id in vessels_for_terminal
                and option.arrival_h <= t < option.return_start_h
            ]
            if active_unloads:
                prob += pulp.lpSum(active_unloads) <= min(1, berth_count)


def _add_executable_trip_path_constraints(
    prob,
    options,
    choose,
    env,
    scenario: Scenario,
    start_step: int,
    horizon_h: int,
) -> None:
    for vessel_id in env.vessel_ids:
        start = _path_start(env, scenario, start_step, vessel_id, horizon_h)
        vessel_options = sorted(
            [i for i, option in enumerate(options) if option.vessel_id == vessel_id],
            key=lambda i: (options[i].load_start_h, options[i].return_start_h, options[i].emitter_id),
        )
        for position, j in enumerate(vessel_options):
            option_j = options[j]
            predecessor_flags = [
                (i, _trip_can_precede(env, scenario, start_step, options[i], option_j))
                for i in vessel_options[:position]
            ]
            initial_reachable = _trip_reachable_from_start(
                env,
                scenario,
                start_step,
                vessel_id,
                start,
                option_j,
            )
            feasible_predecessors = [i for i, can_precede in predecessor_flags if can_precede]
            if not initial_reachable:
                if feasible_predecessors:
                    prob += choose[j] <= pulp.lpSum(choose[i] for i in feasible_predecessors)
                else:
                    prob += choose[j] == 0

            for i, can_precede in predecessor_flags:
                if not can_precede:
                    prob += choose[i] + choose[j] <= 1


def _trip_reachable_from_start(
    env,
    scenario: Scenario,
    start_step: int,
    vessel_id: str,
    start,
    option: _TripOption,
) -> bool:
    if start.node_id is None or start.start_h > option.load_start_h:
        return False
    return _can_sail_between(
        env,
        scenario,
        vessel_id,
        str(start.node_id),
        option.emitter_id,
        start_step + start.start_h,
        option.load_start_h - start.start_h,
    )


def _trip_can_precede(env, scenario: Scenario, start_step: int, before: _TripOption, after: _TripOption) -> bool:
    if before.vessel_id != after.vessel_id or before.return_start_h > after.load_start_h:
        return False
    terminal_id = str(env._routes[before.vessel_id]["destination"])
    return _can_sail_between(
        env,
        scenario,
        before.vessel_id,
        terminal_id,
        after.emitter_id,
        start_step + before.return_start_h,
        after.load_start_h - before.return_start_h,
    )


def _can_sail_between(
    env,
    scenario: Scenario,
    vessel_id: str,
    origin_id: str,
    destination_id: str,
    start_step: int,
    available_h: int,
) -> bool:
    if origin_id == destination_id:
        return True
    if available_h <= 0:
        return False
    sail_h = _sail_hours_between(
        env,
        origin_id,
        destination_id,
        vessel_id,
        scenario=scenario,
        start_step=start_step,
        max_horizon_h=available_h,
    )
    return sail_h <= available_h


def _add_source_balance_constraints(
    prob,
    env,
    scenario: Scenario,
    start_step: int,
    horizon_h: int,
    source_stock,
    source_ready,
    source_overflow,
    vent,
    *,
    load_at,
) -> None:
    for emitter_id in env.emitter_ids:
        emitter = env.network.entities[emitter_id]
        assert isinstance(emitter, Emitter)
        initial_source_t = float(env.simulator.state.entity_inventory_t.get(emitter_id, 0.0))
        prob += source_stock[(emitter_id, 0)] == initial_source_t
        for t in range(horizon_h):
            capture_t = _capture_tonnes(env, scenario, emitter_id, start_step + t)
            pre_load = source_stock[(emitter_id, t)] + capture_t
            source_m = emitter.buffer_capacity_t + max(0.0, capture_t)
            load_expr = load_at(emitter_id, t)
            prob += source_ready[(emitter_id, t)] <= pre_load
            prob += source_ready[(emitter_id, t)] <= emitter.buffer_capacity_t
            prob += source_ready[(emitter_id, t)] >= pre_load - source_m * source_overflow[(emitter_id, t)]
            prob += source_ready[(emitter_id, t)] >= emitter.buffer_capacity_t - emitter.buffer_capacity_t * (
                1 - source_overflow[(emitter_id, t)]
            )
            prob += vent[(emitter_id, t)] == pre_load - source_ready[(emitter_id, t)]
            prob += source_stock[(emitter_id, t + 1)] == source_ready[(emitter_id, t)] - load_expr


def _add_relaxed_terminal_and_injection_constraints(
    prob,
    env,
    scenario: Scenario,
    start_step: int,
    horizon_h: int,
    terminal_stock,
    well_inj,
    *,
    delivered_at,
) -> None:
    prob += terminal_stock[0] == _initial_terminal_t(env)
    well_caps = _continuous_well_caps_by_hour(env, scenario, horizon_h)
    for t in range(horizon_h):
        total_inj = pulp.lpSum(well_inj[(well_id, t)] for well_id in env.well_ids)
        for well_id in env.well_ids:
            prob += well_inj[(well_id, t)] <= well_caps[(well_id, t)]
        _add_pipeline_and_manifold_constraints(prob, env, well_inj, t)
        prob += total_inj <= terminal_stock[t] + delivered_at(t)
        prob += terminal_stock[t + 1] == terminal_stock[t] + delivered_at(t) - total_inj
        for terminal_id, berth_count in _terminal_berth_counts(env, scenario, start_step + t).items():
            active_arrivals = delivered_at(t)
            if active_arrivals is not None:
                # Aggregate relaxed trips unload instantly, so the berth is used
                # only on the arrival hour.
                arrival_count = 0
                for _ in range(min(1, berth_count)):
                    arrival_count += 1
                if arrival_count <= 0:
                    prob += active_arrivals <= 0


def _add_executable_terminal_and_injection_constraints(
    prob,
    env,
    scenario: Scenario,
    start_step: int,
    horizon_h: int,
    terminal_stock,
    options,
    choose,
    well_rate_options,
    well_choice,
    well_inj,
    injection_limit_choice,
) -> None:
    prob += terminal_stock[0] == _initial_terminal_t(env)
    max_hourly_injection_t = 0.0
    for t in range(horizon_h):
        for well_id in env.well_ids:
            max_hourly_injection_t += max(
                mtpa_to_tph(WELL_RATE_LEVELS_MTPA[rate_index])
                for rate_index in well_rate_options[(well_id, t)]
            )
    max_hourly_supply_t = _terminal_capacity_t(env) + sum(
        float(env.network.entities[vessel_id].unloading_rate_tph)
        for vessel_id in env.vessel_ids
    )
    for t in range(horizon_h):
        total_unload = pulp.lpSum(
            amount * choose[i]
            for i, option in enumerate(options)
            for offset, amount in enumerate(option.unload_profile_t)
            if option.arrival_h + offset == t
        )
        total_request = pulp.lpSum(
            mtpa_to_tph(WELL_RATE_LEVELS_MTPA[rate_index]) * well_choice[(well_id, t, rate_index)]
            for well_id in env.well_ids
            for rate_index in well_rate_options[(well_id, t)]
        )
        total_inj = pulp.lpSum(well_inj[(well_id, t)] for well_id in env.well_ids)
        available_for_injection = terminal_stock[t] + total_unload
        for well_id in env.well_ids:
            prob += pulp.lpSum(
                well_choice[(well_id, t, rate_index)]
                for rate_index in well_rate_options[(well_id, t)]
            ) == 1
            well_request = pulp.lpSum(
                mtpa_to_tph(WELL_RATE_LEVELS_MTPA[rate_index]) * well_choice[(well_id, t, rate_index)]
                for rate_index in well_rate_options[(well_id, t)]
            )
            prob += well_inj[(well_id, t)] <= well_request
        prob += total_inj <= total_request
        prob += total_inj <= available_for_injection
        prob += pulp.lpSum(injection_limit_choice[(t, limit)] for limit in range(2)) == 1
        prob += total_inj >= total_request - max_hourly_injection_t * (1 - injection_limit_choice[(t, 0)])
        prob += total_inj >= available_for_injection - max_hourly_supply_t * (1 - injection_limit_choice[(t, 1)])
        _add_pipeline_and_manifold_constraints(prob, env, well_inj, t)
        prob += terminal_stock[t + 1] == terminal_stock[t] + total_unload - total_inj


def _add_pipeline_and_manifold_constraints(prob, env, well_inj, t: int) -> None:
    for pipeline_id, pipeline in env.network._entities_of_type(Pipeline).items():
        prob += pulp.lpSum(
            well_inj[(well_id, t)] for well_id in _pipeline_wells(env, pipeline_id)
        ) <= pipeline.max_flow_tph
    for manifold_id, manifold in env.network._entities_of_type(SubseaManifold).items():
        prob += pulp.lpSum(
            well_inj[(well_id, t)]
            for well_id in env.network._downstream_of_type(manifold_id, InjectionWell)
        ) <= manifold.max_flow_tph


def _continuous_well_caps_by_hour(env, scenario: Scenario, horizon_h: int) -> dict[tuple[str, int], float]:
    options = _well_rate_options_by_hour(env, scenario, horizon_h)
    return {
        (well_id, t): max(mtpa_to_tph(WELL_RATE_LEVELS_MTPA[index]) for index in options[(well_id, t)])
        for well_id in env.well_ids
        for t in range(horizon_h)
    }


def _source_variables(env, horizon_h: int):
    source_stock = {
        (emitter_id, t): pulp.LpVariable(
            f"source_stock_{emitter_id}_{t}",
            lowBound=0.0,
            upBound=env.network.entities[emitter_id].buffer_capacity_t,
        )
        for emitter_id in env.emitter_ids
        for t in range(horizon_h + 1)
    }
    source_ready = {
        (emitter_id, t): pulp.LpVariable(
            f"source_ready_{emitter_id}_{t}",
            lowBound=0.0,
            upBound=env.network.entities[emitter_id].buffer_capacity_t,
        )
        for emitter_id in env.emitter_ids
        for t in range(horizon_h)
    }
    source_overflow = {
        (emitter_id, t): pulp.LpVariable(f"source_overflow_{emitter_id}_{t}", cat="Binary")
        for emitter_id in env.emitter_ids
        for t in range(horizon_h)
    }
    vent = {
        (emitter_id, t): pulp.LpVariable(f"vent_{emitter_id}_{t}", lowBound=0.0)
        for emitter_id in env.emitter_ids
        for t in range(horizon_h)
    }
    return source_stock, source_ready, source_overflow, vent


def _terminal_stock_variables(horizon_h: int, terminal_capacity_t: float):
    return {
        t: pulp.LpVariable(f"terminal_stock_{t}", lowBound=0.0, upBound=terminal_capacity_t)
        for t in range(horizon_h + 1)
    }


def _well_injection_variables(env, horizon_h: int):
    return {
        (well_id, t): pulp.LpVariable(f"well_actual_{well_id}_{t}", lowBound=0.0)
        for well_id in env.well_ids
        for t in range(horizon_h)
    }


def _relaxed_trip_cost_expression(env, options, choose, shipped, stored_expr, params: EconomicParameters):
    return (
        pulp.lpSum(option.sail_fuel_h * params.vessel_fuel_eur_per_h_sailing * choose[i] for i, option in enumerate(options))
        + pulp.lpSum(shipped[i] * params.conditioning_eur_per_t for i in shipped)
        + pulp.lpSum(
            shipped[i] * params.hoteling_fuel_eur_per_h / max(1e-9, options[i].load_rate_tph)
            for i in shipped
        )
        + pulp.lpSum(
            shipped[i] * params.hoteling_fuel_eur_per_h / max(1e-9, options[i].unload_rate_tph)
            for i in shipped
        )
        + stored_expr * params.reconditioning_eur_per_t
    )


def _executable_trip_cost_expression(options, choose, stored_expr, params: EconomicParameters):
    return (
        pulp.lpSum(option.sail_fuel_h * params.vessel_fuel_eur_per_h_sailing * choose[i] for i, option in enumerate(options))
        + pulp.lpSum(option.amount_t * params.conditioning_eur_per_t * choose[i] for i, option in enumerate(options))
        + pulp.lpSum(len(option.load_profile_t) * params.hoteling_fuel_eur_per_h * choose[i] for i, option in enumerate(options))
        + pulp.lpSum(len(option.unload_profile_t) * params.hoteling_fuel_eur_per_h * choose[i] for i, option in enumerate(options))
        + stored_expr * params.reconditioning_eur_per_t
    )


def _relaxed_trip_cost_breakdown(env, options, choose, shipped, stored_t: float, params: EconomicParameters) -> dict[str, float]:
    vessel_fuel = sum(
        option.sail_fuel_h * params.vessel_fuel_eur_per_h_sailing * round(_value(choose[i]))
        for i, option in enumerate(options)
    )
    loaded_t = sum(_value(var) for var in shipped.values())
    loading = sum(
        _value(shipped[i]) * params.hoteling_fuel_eur_per_h / max(1e-9, options[i].load_rate_tph)
        for i in shipped
    )
    unloading = sum(
        _value(shipped[i]) * params.hoteling_fuel_eur_per_h / max(1e-9, options[i].unload_rate_tph)
        for i in shipped
    )
    conditioning = loaded_t * params.conditioning_eur_per_t
    reconditioning = stored_t * params.reconditioning_eur_per_t
    operating_cost = vessel_fuel + conditioning + reconditioning + loading + unloading
    return {
        "vessel_fuel": vessel_fuel,
        "conditioning": conditioning,
        "reconditioning": reconditioning,
        "loading": loading,
        "unloading": unloading,
        "operating_cost": operating_cost,
    }


def _executable_trip_cost_breakdown(
    env,
    scenario: Scenario,
    start_step: int,
    horizon_h: int,
    options,
    choose,
    stored_t: float,
    params: EconomicParameters,
) -> dict[str, float]:
    selected = [option for i, option in enumerate(options) if round(_value(choose[i])) == 1]
    vessel_fuel_h = sum(option.sail_fuel_h for option in selected)
    vessel_fuel_h += _reposition_fuel_hours_for_selected_trips(
        env,
        scenario,
        start_step,
        horizon_h,
        selected,
    )
    vessel_fuel = vessel_fuel_h * params.vessel_fuel_eur_per_h_sailing
    loaded_t = sum(option.amount_t for option in selected)
    loading = sum(len(option.load_profile_t) for option in selected) * params.hoteling_fuel_eur_per_h
    unloading = sum(len(option.unload_profile_t) for option in selected) * params.hoteling_fuel_eur_per_h
    conditioning = loaded_t * params.conditioning_eur_per_t
    reconditioning = stored_t * params.reconditioning_eur_per_t
    operating_cost = vessel_fuel + conditioning + reconditioning + loading + unloading
    return {
        "vessel_fuel": vessel_fuel,
        "conditioning": conditioning,
        "reconditioning": reconditioning,
        "loading": loading,
        "unloading": unloading,
        "operating_cost": operating_cost,
    }


def _expand_executable_trips_to_actions(
    env,
    scenario: Scenario,
    start_step: int,
    horizon_h: int,
    options,
    choose,
) -> dict[str, list[int]]:
    actions = {vessel_id: [VESSEL_WAIT] * horizon_h for vessel_id in env.vessel_ids}
    for vessel_id, selected in _selected_options_by_vessel(env, options, choose).items():
        start = _path_start(env, scenario, start_step, vessel_id, horizon_h)
        current_node = None if start.node_id is None else str(start.node_id)
        current_h = int(start.start_h)
        for option in selected:
            if current_node is not None and current_node != option.emitter_id and current_h < horizon_h:
                actions[vessel_id][current_h] = env.vessel_go_emitter_action(option.emitter_id)
            if option.depart_h < horizon_h:
                actions[vessel_id][option.depart_h] = VESSEL_GO_TERMINAL
            current_node = str(env._routes[vessel_id]["destination"])
            current_h = option.return_start_h
    return actions


def _selected_options_by_vessel(env, options, choose) -> dict[str, list[_TripOption]]:
    selected = {vessel_id: [] for vessel_id in env.vessel_ids}
    for i, option in enumerate(options):
        if round(_value(choose[i])) == 1:
            selected[option.vessel_id].append(option)
    for vessel_id in env.vessel_ids:
        selected[vessel_id].sort(key=lambda option: (option.load_start_h, option.return_start_h, option.emitter_id))
    return selected


def _reposition_fuel_hours_for_selected_trips(
    env,
    scenario: Scenario,
    start_step: int,
    horizon_h: int,
    selected_options: list[_TripOption],
) -> int:
    total = 0
    by_vessel = {vessel_id: [] for vessel_id in env.vessel_ids}
    for option in selected_options:
        by_vessel[option.vessel_id].append(option)
    for vessel_id, selected in by_vessel.items():
        selected.sort(key=lambda option: (option.load_start_h, option.return_start_h, option.emitter_id))
        start = _path_start(env, scenario, start_step, vessel_id, horizon_h)
        current_node = None if start.node_id is None else str(start.node_id)
        current_h = int(start.start_h)
        for option in selected:
            if current_node is not None and current_node != option.emitter_id:
                sail_h = _sail_hours_between(
                    env,
                    current_node,
                    option.emitter_id,
                    vessel_id,
                    scenario=scenario,
                    start_step=start_step + current_h,
                    max_horizon_h=max(1, option.load_start_h - current_h),
                )
                total += max(0, sail_h - 1)
            current_node = str(env._routes[vessel_id]["destination"])
            current_h = option.return_start_h
    return total


def _extract_well_rate_indices(env, horizon_h: int, well_rate_options, well_choice) -> dict[str, list[int]]:
    indices = {well_id: [0] * horizon_h for well_id in env.well_ids}
    for well_id in env.well_ids:
        for t in range(horizon_h):
            selected = [
                rate_index
                for rate_index in well_rate_options[(well_id, t)]
                if round(_value(well_choice[(well_id, t, rate_index)])) == 1
            ]
            indices[well_id][t] = selected[0] if selected else 0
    return indices


def _record_from_option(option: _TripOption, amount_t: float) -> TripRecord:
    return TripRecord(
        vessel_id=option.vessel_id,
        emitter_id=option.emitter_id,
        load_start_h=option.load_start_h,
        depart_h=option.depart_h,
        arrival_h=option.arrival_h,
        return_start_h=option.return_start_h,
        end_h=option.end_h,
        amount_t=float(amount_t),
    )


def _result(
    *,
    level: str,
    status: str,
    horizon_h: int,
    stored_t: float,
    vented_t: float,
    in_transit_t: float,
    initial_in_transit_t: float,
    captured_from_operations_t: float,
    shortfall_t: float,
    trips: list[TripRecord],
    injection_tph: list[float],
    vessel_actions_by_hour: dict[str, list[int]],
    well_rate_indices_by_hour: dict[str, list[int]],
    native_actions_by_hour: list[dict[str, list[int]]],
    cost: dict[str, float],
    total_cost: float,
    storage_reward_eur_per_t: float,
    net_reward: float,
    validation: _Validation,
    binary_count: int,
    variable_count: int,
    constraint_count: int,
) -> TripMilpResult:
    return TripMilpResult(
        level=level,
        status=status,
        horizon_h=horizon_h,
        stored_t=stored_t,
        vented_t=vented_t,
        in_transit_t=in_transit_t,
        in_transit_growth_t=in_transit_t - initial_in_transit_t,
        shortfall_t=shortfall_t,
        deliveries=len(trips),
        trips=trips,
        injection_tph=injection_tph,
        vessel_actions_by_hour=vessel_actions_by_hour,
        well_rate_indices_by_hour=well_rate_indices_by_hour,
        native_actions_by_hour=native_actions_by_hour,
        operating_cost=cost["operating_cost"],
        total_cost=total_cost,
        cost_per_stored_t=cost["operating_cost"] / stored_t if stored_t > 0.0 else float("nan"),
        total_cost_per_stored_t=total_cost / stored_t if stored_t > 0.0 else float("nan"),
        storage_reward_eur_per_t=storage_reward_eur_per_t,
        net_reward=net_reward,
        objective_value=-net_reward,
        vessel_fuel=cost["vessel_fuel"],
        conditioning=cost["conditioning"],
        reconditioning=cost["reconditioning"],
        loading=cost["loading"],
        unloading=cost["unloading"],
        is_valid=validation.is_valid,
        validation_error=validation.validation_error,
        max_binary_integrality_violation=validation.max_binary_integrality_violation,
        binary_count=binary_count,
        variable_count=variable_count,
        constraint_count=constraint_count,
    )


def _validate_trip_solution(
    status: str,
    binary_values,
    *,
    stored_t: float,
    vented_t: float,
    in_transit_t: float,
    captured_from_operations_t: float,
    initial_in_transit_t: float,
    max_storable_from_deliveries_t: float,
    integrality_tol: float = 1e-6,
    mass_tol: float = 1e-4,
) -> _Validation:
    values = list(binary_values)
    numeric = [float(value) for value in values if value is not None]
    if len(numeric) != len(values):
        return _Validation(False, "missing binary variable value", float("inf"))
    max_integrality = max((abs(value - round(value)) for value in numeric), default=0.0)
    if status not in {"Optimal", "Integer Feasible"}:
        return _Validation(False, f"solver status {status} is not a validated integer solution", max_integrality)
    if max_integrality > integrality_tol:
        return _Validation(
            False,
            f"binary integrality violation {max_integrality:.3g} exceeds tolerance {integrality_tol:g}",
            max_integrality,
        )
    lhs = initial_in_transit_t + captured_from_operations_t
    rhs = stored_t + vented_t + in_transit_t
    balance_tol = max(mass_tol, mass_tol * max(1.0, abs(lhs)))
    if abs(lhs - rhs) > balance_tol:
        return _Validation(False, f"mass balance violation: input {lhs:.6g} t != output {rhs:.6g} t", max_integrality)
    capacity_tol = max(mass_tol, mass_tol * max(1.0, max_storable_from_deliveries_t))
    if stored_t - max_storable_from_deliveries_t > capacity_tol:
        return _Validation(
            False,
            (
                "stored mass exceeds delivered/initial-terminal capacity: "
                f"{stored_t:.6g} t > {max_storable_from_deliveries_t:.6g} t"
            ),
            max_integrality,
        )
    return _Validation(True, "", max_integrality)


def _solve(
    prob,
    cplex_path: str | None,
    time_limit_s: float | None,
    mip_gap_rel: float | None,
    mip_gap_abs: float | None,
    threads: int | None,
    msg: bool,
) -> None:
    solver = _make_cplex_cmd(
        cplex_path=cplex_path,
        time_limit_s=time_limit_s,
        mip_gap_rel=mip_gap_rel,
        mip_gap_abs=mip_gap_abs,
        threads=threads,
        msg=msg,
    )
    prob.solve(solver)


def _require_ready(env, scenario: Scenario | None) -> None:
    if pulp is None:
        raise ImportError("The trip MILP requires PuLP. Install it with `pip install pulp`.")
    if scenario is None and getattr(env, "scenario", None) is None:
        raise ValueError("Pass a Scenario or call env.reset(seed=...) before solving.")
    if getattr(env, "simulator", None) is None:
        raise ValueError("Call env.reset(seed=...) before solving the trip MILP.")
    if abs(float(env.network.time_step_hours) - 1.0) > 1e-9:
        raise ValueError("The trip MILP currently expects 1-hour network time steps.")


def _horizon(env, scenario: Scenario, horizon_h: int | None) -> int:
    if horizon_h is not None:
        return int(horizon_h)
    start_step = scenario.step_index(_current_start_hour(env))
    return max(0, int(scenario.n_steps - start_step))


def _empty_result(level: str, horizon_h: int) -> TripMilpResult:
    return TripMilpResult(
        level=level,
        status="Empty horizon",
        horizon_h=horizon_h,
        stored_t=0.0,
        vented_t=0.0,
        in_transit_t=0.0,
        in_transit_growth_t=0.0,
        shortfall_t=0.0,
        deliveries=0,
        trips=[],
        injection_tph=[],
        vessel_actions_by_hour={},
        well_rate_indices_by_hour={},
        native_actions_by_hour=[],
        operating_cost=0.0,
        total_cost=0.0,
        cost_per_stored_t=float("nan"),
        total_cost_per_stored_t=float("nan"),
    )


def _initial_terminal_t(env) -> float:
    return sum(float(env.simulator.state.entity_inventory_t.get(terminal_id, 0.0)) for terminal_id in env.terminal_ids)


def _initial_in_transit_t(env) -> float:
    reservoirs = set(env.reservoir_ids)
    return sum(
        float(value)
        for entity_id, value in env.simulator.state.entity_inventory_t.items()
        if entity_id not in reservoirs
    )


def _captured_from_operations(env, scenario: Scenario, start_step: int, horizon_h: int) -> float:
    return sum(
        _capture_tonnes(env, scenario, emitter_id, start_step + t)
        for emitter_id in env.emitter_ids
        for t in range(horizon_h)
    )


def _storage_shortfall_t(env, captured_from_operations_t: float, stored_t: float) -> float:
    required_storage_t = env.config.storage_target_rate * (
        float(env.cumulative_captured_t) + captured_from_operations_t
    )
    return max(0.0, required_storage_t - (float(env.cumulative_stored_t) + stored_t))


def _storage_reward_eur_per_t(env, override: float | None) -> float:
    if override is not None:
        return float(override)
    store_reward = getattr(env.config, "store_reward_eur_per_t", None)
    if store_reward is None:
        store_reward = getattr(env.config, "injection_reward_eur_per_t", 0.0)
    return float(store_reward)


def _terminal_capacity_t(env) -> float:
    return sum(terminal.storage_capacity_t for terminal in env.network._entities_of_type(Terminal).values())


def _pipeline_wells(env, pipeline_id: str) -> list[str]:
    wells = list(env.network._downstream_of_type(pipeline_id, InjectionWell))
    for manifold_id in env.network._downstream_of_type(pipeline_id, SubseaManifold):
        wells.extend(env.network._downstream_of_type(manifold_id, InjectionWell))
    return wells
