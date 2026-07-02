"""Rolling-horizon MILP controller over the native CCS action space.

The rolling controller is an MPC baseline: every few simulated hours it solves a
finite-horizon MILP from the current simulator state, executes the first slice of
that plan, and then re-plans later. Unlike the fixed-horizon oracle, this module
plans the same action objects the environment accepts:

- each berthed vessel chooses WAIT / GO_TERMINAL / GO_EMITTER[id];
- vessels already sailing are forced to WAIT until they arrive;
- injection is a continuous total rate that is mapped back to per-well Mt/y.

Loading and unloading remain automatic in the environment, so the MILP models
them as continuous flows that are only possible while a vessel waits at the
corresponding emitter or terminal.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Callable

from ..economics import EconomicParameters
from ..entities.storage import InjectionWell
from ..entities.terminal import Terminal
from ..environment import (
    MAX_WELL_RATE_MTPA,
    VESSEL_GO_TERMINAL,
    VESSEL_WAIT,
    CCSEnv,
)
from ..routes import route_distance_km, sea_route
from .baselines import greedy_shuttle_policy
from .milp import KNOTS_TO_KMH, _validate_static_solution, extract_params

_MTPA_TO_TPH = 1_000_000.0 / (365.25 * 24.0)
Policy = Callable[[CCSEnv], dict[str, list]]


@dataclass(frozen=True)
class RollingMilpPlan:
    vessel_actions_by_hour: dict[str, list[int]]
    injection_tph: list[float]
    vented_t: float
    shortfall_t: float
    total_cost: float
    status: str
    is_valid: bool
    validation_error: str = ""
    max_binary_integrality_violation: float = 0.0


@dataclass(frozen=True)
class _PathStart:
    start_h: int
    node_id: str | None


@dataclass(frozen=True)
class _ActionArc:
    vessel_id: str
    start_h: int
    end_h: int
    origin_id: str
    destination_id: str
    action: int
    is_sailing: bool

    @property
    def duration_h(self) -> int:
        return self.end_h - self.start_h


def _plan_explicit_actions(
    env: CCSEnv,
    planning_horizon_h: int,
    economics: EconomicParameters,
    time_limit_s: float = 30.0,
) -> RollingMilpPlan:
    """Plan hourly vessel actions and continuous injection over a lookahead."""
    import pulp

    H = max(1, int(planning_horizon_h))
    hours = range(H)
    state = env.simulator.state
    terminal_capacity_t = _terminal_capacity_t(env)
    injection_cap_by_hour = [_current_injection_cap_tph(env, t) for t in hours]
    arcs, starts = _build_action_arcs(env, H)

    prob = pulp.LpProblem("rolling_explicit_action_plan", pulp.LpMinimize)
    arc_vars = {
        index: pulp.LpVariable(f"x_arc_{index}", cat="Binary")
        for index in range(len(arcs))
    }
    cargo = {
        (vessel_id, t): pulp.LpVariable(
            f"cargo_{vessel_id}_{t}",
            lowBound=0,
            upBound=env.network.entities[vessel_id].capacity_t,
        )
        for vessel_id in env.vessel_ids
        for t in range(H + 1)
    }
    load = {
        (vessel_id, emitter_id, t): pulp.LpVariable(f"load_{vessel_id}_{emitter_id}_{t}", lowBound=0)
        for vessel_id in env.vessel_ids
        for emitter_id in env.emitter_ids
        for t in hours
    }
    unload = {
        (vessel_id, t): pulp.LpVariable(f"unload_{vessel_id}_{t}", lowBound=0)
        for vessel_id in env.vessel_ids
        for t in hours
    }
    source_stock = {
        (emitter_id, t): pulp.LpVariable(
            f"source_stock_{emitter_id}_{t}",
            lowBound=0,
            upBound=env.network.entities[emitter_id].buffer_capacity_t,
        )
        for emitter_id in env.emitter_ids
        for t in range(H + 1)
    }
    terminal_stock = {
        t: pulp.LpVariable(f"terminal_stock_{t}", lowBound=0, upBound=terminal_capacity_t)
        for t in range(H + 1)
    }
    inj = {
        t: pulp.LpVariable(f"inj_{t}", lowBound=0, upBound=injection_cap_by_hour[t])
        for t in hours
    }
    vent = {
        (emitter_id, t): pulp.LpVariable(f"vent_{emitter_id}_{t}", lowBound=0)
        for emitter_id in env.emitter_ids
        for t in hours
    }
    shortfall = pulp.LpVariable("storage_shortfall", lowBound=0)

    incoming, outgoing, wait_arc = _index_arcs(arcs)
    for vessel_id in env.vessel_ids:
        start = starts[vessel_id]
        if start.node_id is None or start.start_h >= H:
            continue
        nodes = _nodes_for_vessel(env, vessel_id)
        for t in range(start.start_h, H):
            for node_id in nodes:
                supply = 1 if t == start.start_h and node_id == start.node_id else 0
                prob += (
                    pulp.lpSum(arc_vars[i] for i in outgoing.get((vessel_id, t, node_id), []))
                    == pulp.lpSum(arc_vars[i] for i in incoming.get((vessel_id, t, node_id), [])) + supply
                )
        prob += (
            pulp.lpSum(
                arc_vars[i]
                for node_id in nodes
                for i in incoming.get((vessel_id, H, node_id), [])
            )
            == 1
        )

    for vessel_id in env.vessel_ids:
        vessel = env.network.entities[vessel_id]
        initial_cargo_t = float(state.entity_inventory_t.get(vessel_id, 0.0))
        prob += cargo[(vessel_id, 0)] == initial_cargo_t
        terminal_id = str(env._routes[vessel_id]["destination"])
        for t in hours:
            prob += (
                cargo[(vessel_id, t + 1)]
                == cargo[(vessel_id, t)]
                + pulp.lpSum(load[(vessel_id, emitter_id, t)] for emitter_id in env.emitter_ids)
                - unload[(vessel_id, t)]
            )
            for emitter_id in env.emitter_ids:
                emitter = env.network.entities[emitter_id]
                load_cap_tph = min(emitter.loading_rate_tph, vessel.loading_rate_tph)
                prob += (
                    load[(vessel_id, emitter_id, t)]
                    <= load_cap_tph * _wait_expr(arc_vars, wait_arc, vessel_id, emitter_id, t)
                )
            prob += (
                unload[(vessel_id, t)]
                <= vessel.unloading_rate_tph * _wait_expr(arc_vars, wait_arc, vessel_id, terminal_id, t)
            )

    for emitter_id in env.emitter_ids:
        initial_source_t = float(state.entity_inventory_t.get(emitter_id, 0.0))
        prob += source_stock[(emitter_id, 0)] == initial_source_t
        emitter = env.network.entities[emitter_id]
        for t in hours:
            capture_t = _capture_tonnes(env, emitter_id, t)
            prob += (
                source_stock[(emitter_id, t + 1)]
                == source_stock[(emitter_id, t)]
                + capture_t
                - pulp.lpSum(load[(vessel_id, emitter_id, t)] for vessel_id in env.vessel_ids)
                - vent[(emitter_id, t)]
            )
            prob += (
                pulp.lpSum(load[(vessel_id, emitter_id, t)] for vessel_id in env.vessel_ids)
                <= emitter.loading_rate_tph
            )

    initial_terminal_t = sum(float(state.entity_inventory_t.get(tid, 0.0)) for tid in env.terminal_ids)
    prob += terminal_stock[0] == initial_terminal_t
    for t in hours:
        prob += terminal_stock[t + 1] == terminal_stock[t] + pulp.lpSum(unload[(vessel_id, t)] for vessel_id in env.vessel_ids) - inj[t]
        for terminal_id, berth_count in _terminal_berth_counts(env).items():
            vessels_for_terminal = [
                vessel_id
                for vessel_id in env.vessel_ids
                if str(env._routes[vessel_id]["destination"]) == terminal_id
            ]
            prob += (
                pulp.lpSum(
                    _wait_expr(arc_vars, wait_arc, vessel_id, terminal_id, t)
                    for vessel_id in vessels_for_terminal
                )
                <= berth_count
            )

    stored_expr = pulp.lpSum(inj[t] for t in hours)
    initial_source_total_t = sum(float(state.entity_inventory_t.get(eid, 0.0)) for eid in env.emitter_ids)
    captured_from_operations_t = sum(_capture_tonnes(env, emitter_id, t) for emitter_id in env.emitter_ids for t in hours)
    captured_total_t = initial_source_total_t + captured_from_operations_t
    prob += shortfall >= env.config.storage_target_rate * captured_total_t - stored_expr
    prob += (
        _sailing_cost_expression(arcs, arc_vars, economics)
        + _loading_cost_expression(env, load, economics)
        + _unloading_cost_expression(env, unload, economics)
        + stored_expr * economics.reconditioning_eur_per_t
        + pulp.lpSum(vent[(emitter_id, t)] for emitter_id in env.emitter_ids for t in hours)
        * economics.carbon_price_eur_per_t
        + shortfall * economics.storage_shortfall_eur_per_t
    )
    prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit_s))

    status = pulp.LpStatus[prob.status]
    vessel_actions_by_hour = _extract_actions(env, H, arcs, arc_vars)
    injection_tph = [max(0.0, _value(inj[t])) for t in hours]
    vented_t = sum(_value(vent[(emitter_id, t)]) for emitter_id in env.emitter_ids for t in hours)
    shortfall_t = _value(shortfall)
    stored_t = sum(injection_tph)
    final_source_t = sum(_value(source_stock[(emitter_id, H)]) for emitter_id in env.emitter_ids)
    final_terminal_t = _value(terminal_stock[H])
    final_cargo_t = sum(_value(cargo[(vessel_id, H)]) for vessel_id in env.vessel_ids)
    initial_cargo_t = sum(float(state.entity_inventory_t.get(vessel_id, 0.0)) for vessel_id in env.vessel_ids)
    initial_in_transit_t = initial_source_total_t + initial_terminal_t + initial_cargo_t
    in_transit_t = final_source_t + final_terminal_t + final_cargo_t
    unloaded_t = sum(_value(unload[(vessel_id, t)]) for vessel_id in env.vessel_ids for t in hours)
    validation = _validate_static_solution(
        status=status,
        binary_values=[arc_vars[index].value() for index in arc_vars],
        stored_t=stored_t,
        vented_t=vented_t,
        in_transit_t=in_transit_t,
        captured_from_operations_t=captured_from_operations_t,
        initial_in_transit_t=initial_in_transit_t,
        max_storable_from_deliveries_t=initial_terminal_t + unloaded_t,
    )
    return RollingMilpPlan(
        vessel_actions_by_hour=vessel_actions_by_hour,
        injection_tph=injection_tph,
        vented_t=vented_t,
        shortfall_t=shortfall_t,
        total_cost=_value(prob.objective),
        status=status,
        is_valid=validation.is_valid,
        validation_error=validation.validation_error,
        max_binary_integrality_violation=validation.max_binary_integrality_violation,
    )


class RollingMilpController:
    """Re-planning MILP controller, usable as a metrics ``policy(env)``."""

    def __init__(
        self,
        env: CCSEnv,
        replan_every: int = 24,
        economics: EconomicParameters | None = None,
        progress: Callable[[str], None] | None = None,
        planning_horizon_h: int = 168,
        time_limit_s: float = 30.0,
        fallback_policy: Policy | None = None,
    ):
        self.replan_every = replan_every
        self.economics = economics or EconomicParameters()
        self.progress = progress
        self.planning_horizon_h = int(planning_horizon_h)
        self.time_limit_s = float(time_limit_s)
        self.fallback_policy = fallback_policy or greedy_shuttle_policy
        self.fallback_policy_name = getattr(self.fallback_policy, "__name__", "fallback_policy")
        self._vessel_actions_by_hour: dict[str, list[int]] = {}
        self._planned_injection_tph: list[float] = []
        self._plan_origin_h: float = -1e9
        self._has_active_plan = False
        self._using_fallback = False
        self.last_plan_status = ""
        self.last_plan_valid = False
        self.last_validation_error = ""
        self.fallback_count = 0

    def __call__(self, env: CCSEnv) -> dict[str, list]:
        return self.policy(env)

    def policy(self, env: CCSEnv) -> dict[str, list]:
        now = env.simulator.state.time_h
        new_episode = now < self._plan_origin_h
        if new_episode or now - self._plan_origin_h >= self.replan_every or not self._has_active_plan:
            self._replan(env, now)

        if self._using_fallback:
            return self.fallback_policy(env)

        masks = env.vessel_action_mask()
        vessel_actions = [
            self._planned_vessel_action(env, vessel_id, now, masks[index])
            for index, vessel_id in enumerate(env.vessel_ids)
        ]
        return {"vessels": vessel_actions, "wells": self._well_rates_for_plan(env, now)}

    def _replan(self, env: CCSEnv, now: float) -> None:
        state = env.simulator.state
        term_init = sum(state.entity_inventory_t.get(t, 0.0) for t in env.terminal_ids)
        source_buffer = sum(state.entity_inventory_t.get(e, 0.0) for e in env.emitter_ids)
        start = time.perf_counter()
        if self.progress is not None:
            self.progress(
                f"  rolling_milp replan at t={now:.0f} h; "
                f"lookahead={self.planning_horizon_h} h; "
                f"terminal={term_init:,.1f} t; source_buffer={source_buffer:,.1f} t"
            )
        plan = _plan_explicit_actions(
            env,
            self.planning_horizon_h,
            self.economics,
            time_limit_s=self.time_limit_s,
        )
        self.last_plan_status = plan.status
        self.last_plan_valid = plan.is_valid
        self.last_validation_error = plan.validation_error
        self._plan_origin_h = now
        self._has_active_plan = True
        if not plan.is_valid:
            self._vessel_actions_by_hour = {}
            self._planned_injection_tph = []
            self._using_fallback = True
            self.fallback_count += 1
            if self.progress is not None:
                self.progress(
                    f"  rolling_milp plan invalid in {time.perf_counter() - start:.1f}s; "
                    f"status={plan.status}; fallback={self.fallback_policy_name}; reason={plan.validation_error}"
                )
            return

        self._vessel_actions_by_hour = plan.vessel_actions_by_hour
        self._planned_injection_tph = plan.injection_tph
        self._using_fallback = False
        planned_departures = sum(
            1
            for actions in self._vessel_actions_by_hour.values()
            for action in actions
            if action != VESSEL_WAIT
        )
        if self.progress is not None:
            self.progress(
                f"  rolling_milp plan ready in {time.perf_counter() - start:.1f}s; "
                f"planned_departures={planned_departures}; "
                f"vented={plan.vented_t:,.1f} t; shortfall={plan.shortfall_t:,.1f} t"
            )

    def _planned_vessel_action(self, env: CCSEnv, vessel_id: str, now: float, mask: list[bool]) -> int:
        actions = self._vessel_actions_by_hour.get(vessel_id)
        if not actions:
            return VESSEL_WAIT
        elapsed = int(max(0.0, math.floor(now - self._plan_origin_h)))
        if elapsed >= len(actions):
            return VESSEL_WAIT
        choice = int(actions[elapsed])
        if 0 <= choice < len(mask) and mask[choice]:
            return choice
        return VESSEL_WAIT

    def _well_rates_for_plan(self, env: CCSEnv, now: float) -> list[float]:
        if not self._planned_injection_tph:
            return [MAX_WELL_RATE_MTPA if upper > 0.0 else 0.0 for _lower, upper in env.well_rate_bounds()]
        elapsed = int(max(0.0, math.floor(now - self._plan_origin_h)))
        index = min(elapsed, len(self._planned_injection_tph) - 1)
        return self._well_rates_from_total_tph(env, self._planned_injection_tph[index])

    def _well_rates_from_total_tph(self, env: CCSEnv, target_tph: float) -> list[float]:
        bounds = env.well_rate_bounds()
        rates = [0.0] * len(bounds)
        available = [(i, lower, upper) for i, (lower, upper) in enumerate(bounds) if upper > 0.0]
        if not available:
            return rates

        min_total_tph = sum(lower * _MTPA_TO_TPH for _i, lower, _upper in available)
        max_total_tph = sum(upper * _MTPA_TO_TPH for _i, _lower, upper in available)
        target_tph = min(max(float(target_tph), min_total_tph), max_total_tph)

        for i, lower, _upper in available:
            rates[i] = lower
        remaining_tph = target_tph - min_total_tph
        for i, lower, upper in available:
            extra_tph = min(remaining_tph, (upper - lower) * _MTPA_TO_TPH)
            rates[i] += extra_tph / _MTPA_TO_TPH
            remaining_tph -= extra_tph
            if remaining_tph <= 1e-9:
                break
        return rates


def _build_action_arcs(env: CCSEnv, horizon_h: int) -> tuple[list[_ActionArc], dict[str, _PathStart]]:
    arcs: list[_ActionArc] = []
    starts = {vessel_id: _path_start(env, vessel_id, horizon_h) for vessel_id in env.vessel_ids}
    for vessel_id in env.vessel_ids:
        start = starts[vessel_id]
        if start.node_id is None or start.start_h >= horizon_h:
            continue
        nodes = _nodes_for_vessel(env, vessel_id)
        for t in range(start.start_h, horizon_h):
            for origin_id in nodes:
                arcs.append(
                    _ActionArc(
                        vessel_id=vessel_id,
                        start_h=t,
                        end_h=t + 1,
                        origin_id=origin_id,
                        destination_id=origin_id,
                        action=VESSEL_WAIT,
                        is_sailing=False,
                    )
                )
                for destination_id in nodes:
                    if destination_id == origin_id:
                        continue
                    duration_h = _sail_hours_between(
                        env,
                        origin_id,
                        destination_id,
                        vessel_id,
                        start_h=t,
                        max_horizon_h=horizon_h - t,
                    )
                    if t + duration_h > horizon_h:
                        continue
                    arcs.append(
                        _ActionArc(
                            vessel_id=vessel_id,
                            start_h=t,
                            end_h=t + duration_h,
                            origin_id=origin_id,
                            destination_id=destination_id,
                            action=_action_to_destination(env, vessel_id, destination_id),
                            is_sailing=True,
                        )
                    )
    return arcs, starts


def _index_arcs(arcs: list[_ActionArc]):
    incoming: dict[tuple[str, int, str], list[int]] = {}
    outgoing: dict[tuple[str, int, str], list[int]] = {}
    wait_arc: dict[tuple[str, str, int], int] = {}
    for index, arc in enumerate(arcs):
        outgoing.setdefault((arc.vessel_id, arc.start_h, arc.origin_id), []).append(index)
        incoming.setdefault((arc.vessel_id, arc.end_h, arc.destination_id), []).append(index)
        if not arc.is_sailing:
            wait_arc[(arc.vessel_id, arc.origin_id, arc.start_h)] = index
    return incoming, outgoing, wait_arc


def _nodes_for_vessel(env: CCSEnv, vessel_id: str) -> list[str]:
    terminal_id = str(env._routes[vessel_id]["destination"])
    return list(dict.fromkeys([*env.emitter_ids, terminal_id]))


def _path_start(env: CCSEnv, vessel_id: str, horizon_h: int) -> _PathStart:
    vstate = env.simulator.vessel_states[vessel_id]
    if vstate["mode"] == "berthed":
        return _PathStart(0, str(vstate["berth"]))
    remaining_h = _remaining_sailing_hours(env, vessel_id, max_horizon_h=horizon_h)
    if remaining_h >= horizon_h:
        return _PathStart(horizon_h, None)
    return _PathStart(remaining_h, str(vstate["destination"]))


def _remaining_sailing_hours(env: CCSEnv, vessel_id: str, max_horizon_h: int | None = None) -> int:
    route = env._routes[vessel_id]
    vstate = env.simulator.vessel_states[vessel_id]
    distance_km = float(vstate.get("distance_km") or route["distance_km"])
    remaining_km = max(0.0, distance_km * (1.0 - float(vstate["progress"])))
    return _sailing_duration_h(
        env,
        vessel_id,
        distance_km=remaining_km,
        start_h=0,
        max_horizon_h=max_horizon_h,
    )


def _sail_hours_between(
    env: CCSEnv,
    origin_id: str,
    destination_id: str,
    vessel_id: str,
    *,
    start_h: int = 0,
    max_horizon_h: int | None = None,
) -> int:
    route = env._routes[vessel_id]
    if {origin_id, destination_id} == {str(route["origin"]), str(route["destination"])}:
        distance_km = float(route["distance_km"])
    else:
        distance_km = _dynamic_leg_distance_km(env, route, origin_id, destination_id)
    return _sailing_duration_h(
        env,
        vessel_id,
        distance_km=distance_km,
        start_h=start_h,
        max_horizon_h=max_horizon_h,
    )


def _sailing_duration_h(
    env: CCSEnv,
    vessel_id: str,
    *,
    distance_km: float,
    start_h: int,
    max_horizon_h: int | None,
) -> int:
    route = env._routes[vessel_id]
    speed_kmh = max(0.0, float(route["speed_knots"])) * KNOTS_TO_KMH
    if distance_km <= 1e-9:
        return 0
    if speed_kmh <= 1e-9:
        return 1 if max_horizon_h is None else max_horizon_h + 1

    nominal_h = max(1, math.ceil(distance_km / speed_kmh))
    max_search_h = max(1, int(max_horizon_h)) if max_horizon_h is not None else nominal_h * 10 + 24
    covered_km = 0.0
    for elapsed_h in range(1, max_search_h + 1):
        offset_h = start_h + elapsed_h - 1
        speed_factor = _forecast_vessel_speed_factor(env, vessel_id, offset_h)
        covered_km += speed_kmh * speed_factor
        if covered_km >= distance_km - 1e-9:
            return elapsed_h
    return max_search_h + 1


def _dynamic_leg_distance_km(env: CCSEnv, route: dict, origin_id: str, destination_id: str) -> float:
    leg_routes = route.setdefault("dynamic_leg_routes", {})
    leg_id = f"{origin_id}->{destination_id}"
    if leg_id not in leg_routes:
        maritime_route = sea_route(env.locations[origin_id], env.locations[destination_id])
        coordinates = _connect_route_to_endpoints(
            maritime_route.coordinates,
            env.locations[origin_id],
            env.locations[destination_id],
        )
        leg_routes[leg_id] = {
            "id": leg_id,
            "origin": origin_id,
            "destination": destination_id,
            "provider": maritime_route.provider,
            "distance_km": round(route_distance_km(coordinates), 2),
            "coordinates": coordinates,
        }
    return float(leg_routes[leg_id]["distance_km"])


def _connect_route_to_endpoints(coordinates, origin, destination):
    connected = list(coordinates)
    if not connected:
        return [origin, destination]
    if connected[0] != origin:
        connected.insert(0, origin)
    if connected[-1] != destination:
        connected.append(destination)
    return connected


def _action_to_destination(env: CCSEnv, vessel_id: str, destination_id: str) -> int:
    if destination_id == str(env._routes[vessel_id]["destination"]):
        return VESSEL_GO_TERMINAL
    return env.vessel_go_emitter_action(destination_id)


def _wait_expr(arc_vars, wait_arc: dict[tuple[str, str, int], int], vessel_id: str, node_id: str, t: int):
    index = wait_arc.get((vessel_id, node_id, t))
    return 0 if index is None else arc_vars[index]


def _capture_tonnes(env: CCSEnv, emitter_id: str, offset_h: int) -> float:
    state = env.simulator.state
    emitter = env.network.entities[emitter_id]
    availability = _forecast_series_value(
        env,
        env.scenario.emitter_availability if env.scenario is not None else {},
        emitter_id,
        offset_h,
        state.emitter_availability.get(emitter_id, emitter.availability),
    )
    return emitter.capture_rate_tph_at(state.time_h + offset_h) * max(0.0, float(availability))


def _terminal_capacity_t(env: CCSEnv) -> float:
    return sum(env.network.entities[terminal_id].storage_capacity_t for terminal_id in env.terminal_ids)


def _terminal_berth_counts(env: CCSEnv) -> dict[str, int]:
    return {
        terminal_id: max(0, int(terminal.berth_count))
        for terminal_id, terminal in env.network._entities_of_type(Terminal).items()
    }


def _current_injection_cap_tph(env: CCSEnv, offset_h: int = 0) -> float:
    _vessels, nominal_injection_cap_tph, _capture_rate, _terminal_capacity = extract_params(env)
    state = env.simulator.state
    well_cap_tph = 0.0
    for well_id, well in env.network._entities_of_type(InjectionWell).items():
        available = _forecast_series_value(
            env,
            env.scenario.well_available if env.scenario is not None else {},
            well_id,
            offset_h,
            state.well_available.get(well_id, well.available),
        )
        if not bool(available):
            continue
        injectivity = _forecast_series_value(
            env,
            env.scenario.injectivity_factor if env.scenario is not None else {},
            well_id,
            offset_h,
            state.injectivity_factor.get(well_id, 1.0),
        )
        well_cap_tph += well.max_injection_tph * max(0.0, float(injectivity))
    return min(nominal_injection_cap_tph, well_cap_tph)


def _forecast_vessel_speed_factor(env: CCSEnv, vessel_id: str, offset_h: int) -> float:
    state = env.simulator.state
    value = _forecast_series_value(
        env,
        env.scenario.vessel_speed_factor if env.scenario is not None else {},
        vessel_id,
        offset_h,
        state.vessel_speed_factor.get(vessel_id, 1.0),
    )
    return max(0.0, float(value))


def _forecast_series_value(
    env: CCSEnv,
    series_by_id: dict[str, list],
    entity_id: str,
    offset_h: int,
    fallback,
):
    series = series_by_id.get(entity_id)
    if not series or env.scenario is None:
        return fallback
    time_h = env.simulator.state.time_h + offset_h * env.network.time_step_hours
    return series[env.scenario.step_index(time_h)]


def _sailing_cost_expression(arcs: list[_ActionArc], arc_vars, params: EconomicParameters):
    import pulp

    return pulp.lpSum(
        arc.duration_h * params.vessel_fuel_eur_per_h_sailing * arc_vars[index]
        for index, arc in enumerate(arcs)
        if arc.is_sailing
    )


def _loading_cost_expression(env: CCSEnv, load, params: EconomicParameters):
    import pulp

    terms = []
    for (vessel_id, emitter_id, _t), var in load.items():
        vessel = env.network.entities[vessel_id]
        emitter = env.network.entities[emitter_id]
        load_rate_tph = max(1e-9, min(vessel.loading_rate_tph, emitter.loading_rate_tph))
        terms.append(var * (params.conditioning_eur_per_t + params.hoteling_fuel_eur_per_h / load_rate_tph))
    return pulp.lpSum(terms)


def _unloading_cost_expression(env: CCSEnv, unload, params: EconomicParameters):
    import pulp

    terms = []
    for (vessel_id, _t), var in unload.items():
        vessel = env.network.entities[vessel_id]
        unload_rate_tph = max(1e-9, vessel.unloading_rate_tph)
        terms.append(var * (params.hoteling_fuel_eur_per_h / unload_rate_tph))
    return pulp.lpSum(terms)


def _extract_actions(env: CCSEnv, horizon_h: int, arcs: list[_ActionArc], arc_vars) -> dict[str, list[int]]:
    actions = {vessel_id: [VESSEL_WAIT] * horizon_h for vessel_id in env.vessel_ids}
    for index, arc in enumerate(arcs):
        if round(arc_vars[index].value() or 0.0) == 1 and arc.start_h < horizon_h:
            actions[arc.vessel_id][arc.start_h] = arc.action
    return actions


def _value(var_or_expr) -> float:
    value = var_or_expr.value()
    return float(value) if value is not None else 0.0
