"""Full-scenario MILP oracle solved with an external CPLEX executable.

This module is intentionally separate from ``milp.py``. The existing MILP
benchmark uses PuLP's bundled CBC interface; this file builds the same kind of
perfect-foresight, fixed-horizon oracle but calls IBM CPLEX through
``pulp.CPLEX_CMD``. CPLEX is expected to be installed outside this project and
available either on ``PATH`` or via the ``cplex_path`` argument.

The formulation is an offline benchmark, not an online RL policy. It sees the
whole sampled ``Scenario``: capture availability, weather/speed factors, well
availability, and injectivity over the complete horizon.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
import os
import subprocess
import time

try:
    import pulp
except ImportError:  # pragma: no cover - exercised only in minimal installs
    pulp = None

from ..economics import EconomicParameters
from ..entities.emitter import Emitter
from ..entities.manifold import SubseaManifold
from ..entities.pipeline import Pipeline
from ..entities.storage import InjectionWell, Reservoir
from ..entities.terminal import Terminal
from ..entities.vessel import Vessel
from ..environment import VESSEL_GO_EMITTER_BASE, VESSEL_GO_TERMINAL, VESSEL_WAIT, WELL_RATE_LEVELS_MTPA
from ..line_source import variable_rate_bottomhole_pressure_bar
from ..operations.pressure_limits import mtpa_to_tph, pressure_limited_rate_level_mask, tph_to_mtpa
from ..routes import route_distance_km, sea_route
from ..scenario_generation import Scenario
from .objective import control_objective_value, control_objective_weights
from .replay import ReplayExpectation, ReplayTolerances, replay_native_actions

KNOTS_TO_KMH = 1.852


if pulp is not None:

    class _CplexMipDirectSolutionCmd(pulp.CPLEX_CMD):
        """CPLEX_CMD variant that writes the MIP incumbent directly."""

        def actualSolve(self, lp):
            if not self.executable(self.path):
                raise pulp.PulpSolverError("PuLP: cannot execute " + self.path)
            tmpLp, tmpSol, tmpMst = self.create_tmp_files(lp.name, "lp", "sol", "mst")
            vs = lp.writeLP(tmpLp, writeSOS=1)
            try:
                os.remove(tmpSol)
            except OSError:
                pass
            if not self.msg:
                cplex = subprocess.Popen(
                    self.path,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
            else:
                cplex = subprocess.Popen(self.path, stdin=subprocess.PIPE)
            cplex_cmds = "read " + tmpLp + "\n"
            if self.optionsDict.get("warmStart", False):
                self.writesol(filename=tmpMst, vs=vs)
                cplex_cmds += "read " + tmpMst + "\n"
                cplex_cmds += "set advance 1\n"

            if self.timeLimit is not None:
                cplex_cmds += "set timelimit " + str(self.timeLimit) + "\n"
            for option in self.options + self.getOptions():
                cplex_cmds += option + "\n"
            if lp.isMIP():
                if self.mip:
                    cplex_cmds += "mipopt\n"
                else:
                    cplex_cmds += "change problem lp\n"
                    cplex_cmds += "optimize\n"
            else:
                cplex_cmds += "optimize\n"
            cplex_cmds += "write " + tmpSol + "\n"
            cplex_cmds += "quit\n"
            cplex.communicate(cplex_cmds.encode("UTF-8"))
            if cplex.returncode != 0:
                raise pulp.PulpSolverError("PuLP: Error while trying to execute " + self.path)
            if not os.path.exists(tmpSol):
                status = pulp.constants.LpStatusInfeasible
                values = reducedCosts = shadowPrices = slacks = solStatus = None
            else:
                status, values, reducedCosts, shadowPrices, slacks, solStatus = self.readsol(tmpSol)
            self.delete_tmp_files(tmpLp, tmpMst, tmpSol)
            self._delete_default_log()
            if status != pulp.constants.LpStatusInfeasible:
                lp.assignVarsVals(values)
                lp.assignVarsDj(reducedCosts)
                lp.assignConsPi(shadowPrices)
                lp.assignConsSlack(slacks)
            lp.assignStatus(status, solStatus)
            return status

        def _delete_default_log(self) -> None:
            if self.optionsDict.get("logPath") == "cplex.log":
                return
            try:
                self.delete_tmp_files("cplex.log")
            except PermissionError:
                pass


@dataclass(frozen=True)
class CplexVesselParams:
    vessel_id: str
    source_id: str
    capacity_t: float
    load_rate_tph: float
    unload_rate_tph: float
    speed_knots: float

    @property
    def load_dur_h(self) -> int:
        return max(1, math.ceil(self.capacity_t / self.load_rate_tph))

    @property
    def unload_dur_h(self) -> int:
        return max(1, math.ceil(self.capacity_t / self.unload_rate_tph))


@dataclass(frozen=True)
class CplexCostBreakdown:
    vessel_fuel: float = 0.0
    conditioning: float = 0.0
    reconditioning: float = 0.0
    loading: float = 0.0
    unloading: float = 0.0

    @property
    def operating_cost(self) -> float:
        return self.vessel_fuel + self.conditioning + self.reconditioning + self.loading + self.unloading


@dataclass(frozen=True)
class FullScenarioCplexMilpResult:
    status: str
    horizon_h: int
    stored_t: float
    vented_t: float
    in_transit_t: float
    in_transit_growth_t: float
    shortfall_t: float
    deliveries: int
    departures: dict[str, list[int]]
    arrivals: dict[str, list[int]]
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
    captured_from_operations_t: float = 0.0
    overflow_risk_t: float = 0.0
    is_valid: bool = True
    validation_error: str = ""
    max_binary_integrality_violation: float = 0.0


@dataclass(frozen=True)
class CplexMilpReplayResult:
    elapsed_hours: int
    stored_t: float
    vented_t: float
    operating_cost: float
    total_cost: float
    total_reward: float
    objective_value: float
    overflow_risk_t: float
    stored_gap_t: float
    violations: list[str]
    is_executable: bool
    is_exact: bool = False
    mismatches: tuple[str, ...] = ()
    compared_fields: frozenset[str] = frozenset()


@dataclass(frozen=True)
class _DepartureOption:
    vessel_id: str
    depart_h: int
    arrival_h: int
    next_depart_h: int
    outbound_sail_h: int
    return_sail_h: int
    unload_dur_h: int


@dataclass(frozen=True)
class _Validation:
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
    arrives_within_horizon: bool = True

    @property
    def duration_h(self) -> int:
        return self.end_h - self.start_h


def solve_full_scenario_with_cplex(
    env,
    *,
    scenario: Scenario | None = None,
    horizon_h: int | None = None,
    economics: EconomicParameters | None = None,
    storage_reward_eur_per_t: float | None = None,
    warm_start_native_actions_by_hour: list[dict[str, list[int]]] | None = None,
    cplex_path: str | None = None,
    time_limit_s: float | None = None,
    mip_gap_rel: float | None = None,
    mip_gap_abs: float | None = None,
    threads: int | None = None,
    msg: bool = False,
    lexicographic_vent_first: bool = False,
    environment_aligned_service: bool = False,
    safe_execution_h: int | None = None,
) -> FullScenarioCplexMilpResult:
    """Solve one full scenario with perfect foresight using external CPLEX.

    The decision variables mirror the RL action protocol: hourly vessel action
    ids and per-well discrete rate-level indices. CPLEX sees the whole scenario,
    so the result is an offline upper-bound benchmark rather than an online
    policy.
    """

    _require_pulp()
    if scenario is None:
        scenario = getattr(env, "scenario", None)
    if scenario is None:
        raise ValueError("Pass a Scenario or call env.reset(seed=...) before solving.")
    if getattr(env, "simulator", None) is None:
        raise ValueError("Call env.reset(seed=...) before solving the native-action CPLEX MILP.")
    if abs(float(env.network.time_step_hours) - 1.0) > 1e-9:
        raise ValueError("The CPLEX MILP currently expects 1-hour network time steps.")

    params = economics or EconomicParameters()
    objective_weights = control_objective_weights(
        env,
        params,
        storage_reward_eur_per_t=storage_reward_eur_per_t,
    )
    reward_per_t = objective_weights.storage_reward_eur_per_t
    start_h = _current_start_hour(env)
    start_step = scenario.step_index(start_h)
    if horizon_h is None:
        horizon_h = max(0, scenario.n_steps - start_step)
    H = int(horizon_h)
    if H <= 0:
        return _empty_result()

    state = env.simulator.state
    hours = range(H)
    terminal_capacity_t = _terminal_capacity_t(env)
    arcs, starts = _build_action_arcs(env, scenario, start_step, H)
    well_rate_options = _well_rate_options_by_hour(env, scenario, H)

    prob = pulp.LpProblem("full_scenario_native_action_cplex_milp", pulp.LpMinimize)
    arc_vars = {
        index: pulp.LpVariable(f"x_arc_{index}", cat="Binary")
        for index in range(len(arcs))
    }
    cargo = {
        (vessel_id, t): pulp.LpVariable(
            f"cargo_{vessel_id}_{t}",
            lowBound=0.0,
            upBound=env.network.entities[vessel_id].capacity_t,
        )
        for vessel_id in env.vessel_ids
        for t in range(H + 1)
    }
    cargo_positive = {
        (vessel_id, t): pulp.LpVariable(f"cargo_positive_{vessel_id}_{t}", cat="Binary")
        for vessel_id in env.vessel_ids
        for t in hours
    } if environment_aligned_service else {}
    cargo_space = {
        (vessel_id, t): pulp.LpVariable(f"cargo_space_{vessel_id}_{t}", cat="Binary")
        for vessel_id in env.vessel_ids
        for t in hours
    } if environment_aligned_service else {}
    load = {
        (vessel_id, emitter_id, t): pulp.LpVariable(f"load_{vessel_id}_{emitter_id}_{t}", lowBound=0.0)
        for vessel_id in env.vessel_ids
        for emitter_id in env.emitter_ids
        for t in hours
    }
    load_active = {
        (vessel_id, emitter_id, t): pulp.LpVariable(f"load_active_{vessel_id}_{emitter_id}_{t}", cat="Binary")
        for vessel_id in env.vessel_ids
        for emitter_id in env.emitter_ids
        for t in hours
    }
    load_limit_choice = {
        (vessel_id, emitter_id, t, limit): pulp.LpVariable(
            f"load_limit_{vessel_id}_{emitter_id}_{t}_{limit}", cat="Binary"
        )
        for vessel_id in env.vessel_ids
        for emitter_id in env.emitter_ids
        for t in hours
        for limit in range(3)
    } if environment_aligned_service else {}
    unload = {
        (vessel_id, t): pulp.LpVariable(f"unload_{vessel_id}_{t}", lowBound=0.0)
        for vessel_id in env.vessel_ids
        for t in hours
    }
    unload_active = {
        (vessel_id, t): pulp.LpVariable(f"unload_active_{vessel_id}_{t}", cat="Binary")
        for vessel_id in env.vessel_ids
        for t in hours
    }
    unload_limit_choice = {
        (vessel_id, t, limit): pulp.LpVariable(
            f"unload_limit_{vessel_id}_{t}_{limit}", cat="Binary"
        )
        for vessel_id in env.vessel_ids
        for t in hours
        for limit in range(3)
    } if environment_aligned_service else {}
    source_stock = {
        (emitter_id, t): pulp.LpVariable(
            f"source_stock_{emitter_id}_{t}",
            lowBound=0.0,
            upBound=env.network.entities[emitter_id].buffer_capacity_t,
        )
        for emitter_id in env.emitter_ids
        for t in range(H + 1)
    }
    source_ready = {
        (emitter_id, t): pulp.LpVariable(
            f"source_ready_{emitter_id}_{t}",
            lowBound=0.0,
            upBound=env.network.entities[emitter_id].buffer_capacity_t,
        )
        for emitter_id in env.emitter_ids
        for t in hours
    }
    source_overflow_active = {
        (emitter_id, t): pulp.LpVariable(f"source_overflow_{emitter_id}_{t}", cat="Binary")
        for emitter_id in env.emitter_ids
        for t in hours
    }
    terminal_stock = {
        t: pulp.LpVariable(f"terminal_stock_{t}", lowBound=0.0, upBound=terminal_capacity_t)
        for t in range(H + 1)
    }
    well_choice = {
        (well_id, t, rate_index): pulp.LpVariable(f"well_{well_id}_{t}_{rate_index}", cat="Binary")
        for well_id in env.well_ids
        for t in hours
        for rate_index in well_rate_options[(well_id, t)]
    }
    well_inj = {
        (well_id, t): pulp.LpVariable(f"well_actual_{well_id}_{t}", lowBound=0.0)
        for well_id in env.well_ids
        for t in hours
    }
    injection_limit_choice = {
        (t, limit): pulp.LpVariable(f"injection_limit_{t}_{limit}", cat="Binary")
        for t in hours
        for limit in range(2)
    }
    vent = {
        (emitter_id, t): pulp.LpVariable(f"vent_{emitter_id}_{t}", lowBound=0.0)
        for emitter_id in env.emitter_ids
        for t in hours
    }

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

    if safe_execution_h is not None:
        safe_h = max(0, min(H, int(safe_execution_h)))
        for index, arc in enumerate(arcs):
            if not arc.is_sailing or arc.start_h >= safe_h:
                continue
            terminal_id = str(env._routes[arc.vessel_id]["destination"])
            if arc.origin_id != terminal_id:
                continue
            vessel_state = env.simulator.vessel_states[arc.vessel_id]
            initially_empty_at_terminal = (
                vessel_state["mode"] == "berthed"
                and str(vessel_state["berth"]) == terminal_id
                and float(state.entity_inventory_t.get(arc.vessel_id, 0.0)) <= 1e-9
            )
            if not (initially_empty_at_terminal and arc.start_h == 0):
                prob += arc_vars[index] == 0

    if environment_aligned_service:
        for t in hours:
            for node_id in [*env.emitter_ids, *env.terminal_ids]:
                prob += pulp.lpSum(
                    arc_vars[index]
                    for vessel_id in env.vessel_ids
                    for index in outgoing.get((vessel_id, t, node_id), [])
                ) <= 1

    for vessel_id in env.vessel_ids:
        vessel = env.network.entities[vessel_id]
        initial_cargo_t = float(state.entity_inventory_t.get(vessel_id, 0.0))
        prob += cargo[(vessel_id, 0)] == initial_cargo_t
        terminal_id = str(env._routes[vessel_id]["destination"])
        for t in hours:
            if environment_aligned_service:
                capacity_t = float(vessel.capacity_t)
                epsilon_t = 1e-6
                prob += cargo[(vessel_id, t)] <= capacity_t * cargo_positive[(vessel_id, t)]
                prob += cargo[(vessel_id, t)] >= epsilon_t * cargo_positive[(vessel_id, t)]
                prob += cargo[(vessel_id, t)] >= capacity_t * (1 - cargo_space[(vessel_id, t)])
                prob += (
                    cargo[(vessel_id, t)]
                    <= capacity_t - epsilon_t + capacity_t * (1 - cargo_space[(vessel_id, t)])
                )
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
                prob += load[(vessel_id, emitter_id, t)] <= load_cap_tph * load_active[(vessel_id, emitter_id, t)]
                prob += load_active[(vessel_id, emitter_id, t)] <= _wait_expr(
                    arc_vars,
                    wait_arc,
                    vessel_id,
                    emitter_id,
                    t,
                )
                prob += load[(vessel_id, emitter_id, t)] <= source_ready[(emitter_id, t)]
                prob += load[(vessel_id, emitter_id, t)] <= vessel.capacity_t - cargo[(vessel_id, t)]
                if environment_aligned_service:
                    wait = _wait_expr(arc_vars, wait_arc, vessel_id, emitter_id, t)
                    active = load_active[(vessel_id, emitter_id, t)]
                    prob += active <= cargo_space[(vessel_id, t)]
                    prob += active >= wait + cargo_space[(vessel_id, t)] - 1
                    choices = [load_limit_choice[(vessel_id, emitter_id, t, i)] for i in range(3)]
                    prob += pulp.lpSum(choices) == active
                    big_m = max(
                        float(vessel.capacity_t),
                        float(emitter.buffer_capacity_t),
                        float(load_cap_tph),
                    )
                    limits = (
                        float(load_cap_tph),
                        source_ready[(emitter_id, t)],
                        float(vessel.capacity_t) - cargo[(vessel_id, t)],
                    )
                    for choice, limit in zip(choices, limits):
                        prob += load[(vessel_id, emitter_id, t)] >= limit - big_m * (1 - choice)
            prob += (
                unload[(vessel_id, t)]
                <= vessel.unloading_rate_tph * _wait_expr(arc_vars, wait_arc, vessel_id, terminal_id, t)
            )
            prob += unload[(vessel_id, t)] <= vessel.unloading_rate_tph * unload_active[(vessel_id, t)]
            prob += unload_active[(vessel_id, t)] <= _wait_expr(arc_vars, wait_arc, vessel_id, terminal_id, t)
            prob += unload[(vessel_id, t)] <= cargo[(vessel_id, t)]
            if environment_aligned_service:
                wait = _wait_expr(arc_vars, wait_arc, vessel_id, terminal_id, t)
                active = unload_active[(vessel_id, t)]
                prob += active <= cargo_positive[(vessel_id, t)]
                prob += active >= wait + cargo_positive[(vessel_id, t)] - 1

        if environment_aligned_service and env.config.require_empty_terminal_departure:
            for index, arc in enumerate(arcs):
                if (
                    arc.vessel_id == vessel_id
                    and arc.is_sailing
                    and arc.origin_id == terminal_id
                ):
                    prob += cargo[(vessel_id, arc.start_h)] <= vessel.capacity_t * (1 - arc_vars[index])

    for emitter_id in env.emitter_ids:
        initial_source_t = float(state.entity_inventory_t.get(emitter_id, 0.0))
        prob += source_stock[(emitter_id, 0)] == initial_source_t
        emitter = env.network.entities[emitter_id]
        for t in hours:
            capture_t = _capture_tonnes(env, scenario, emitter_id, start_step + t)
            source_pre_load = source_stock[(emitter_id, t)] + capture_t
            source_m = emitter.buffer_capacity_t + max(0.0, capture_t)
            prob += source_ready[(emitter_id, t)] <= source_pre_load
            prob += source_ready[(emitter_id, t)] <= emitter.buffer_capacity_t
            prob += source_ready[(emitter_id, t)] >= source_pre_load - source_m * source_overflow_active[(emitter_id, t)]
            prob += (
                source_ready[(emitter_id, t)]
                >= emitter.buffer_capacity_t - emitter.buffer_capacity_t * (1 - source_overflow_active[(emitter_id, t)])
            )
            prob += vent[(emitter_id, t)] == source_pre_load - source_ready[(emitter_id, t)]
            prob += (
                source_stock[(emitter_id, t + 1)]
                == source_ready[(emitter_id, t)]
                - pulp.lpSum(load[(vessel_id, emitter_id, t)] for vessel_id in env.vessel_ids)
            )
            prob += pulp.lpSum(load_active[(vessel_id, emitter_id, t)] for vessel_id in env.vessel_ids) <= 1
            prob += (
                pulp.lpSum(load[(vessel_id, emitter_id, t)] for vessel_id in env.vessel_ids)
                <= emitter.loading_rate_tph
            )

    overflow_risk = _add_overflow_risk_constraints(
        prob,
        env,
        scenario,
        start_step,
        H,
        source_stock,
        objective_weights.overflow_risk_lookahead_h,
    )

    well_request = {
        (well_id, t): pulp.lpSum(
            mtpa_to_tph(WELL_RATE_LEVELS_MTPA[rate_index]) * well_choice[(well_id, t, rate_index)]
            for rate_index in well_rate_options[(well_id, t)]
        )
        for well_id in env.well_ids
        for t in hours
    }
    for well_id in env.well_ids:
        for t in hours:
            prob += pulp.lpSum(
                well_choice[(well_id, t, rate_index)]
                for rate_index in well_rate_options[(well_id, t)]
            ) == 1
            prob += well_inj[(well_id, t)] <= well_request[(well_id, t)]

    max_hourly_injection_t = max(
        (
            sum(
                max(mtpa_to_tph(WELL_RATE_LEVELS_MTPA[rate_index]) for rate_index in well_rate_options[(well_id, t)])
                for well_id in env.well_ids
            )
            for t in hours
        ),
        default=0.0,
    )
    max_hourly_supply_t = terminal_capacity_t + sum(
        float(env.network.entities[vessel_id].unloading_rate_tph) for vessel_id in env.vessel_ids
    )
    for t in hours:
        total_inj = pulp.lpSum(well_inj[(well_id, t)] for well_id in env.well_ids)
        total_request = pulp.lpSum(well_request[(well_id, t)] for well_id in env.well_ids)
        total_unload = pulp.lpSum(unload[(vessel_id, t)] for vessel_id in env.vessel_ids)
        available_for_injection = terminal_stock[t] + total_unload
        prob += total_inj <= total_request
        prob += total_inj <= available_for_injection
        prob += pulp.lpSum(injection_limit_choice[(t, limit)] for limit in range(2)) == 1
        prob += total_inj >= total_request - max_hourly_injection_t * (1 - injection_limit_choice[(t, 0)])
        prob += total_inj >= available_for_injection - max_hourly_supply_t * (1 - injection_limit_choice[(t, 1)])
        prob += terminal_stock[t + 1] == terminal_stock[t] + pulp.lpSum(
            unload[(vessel_id, t)] for vessel_id in env.vessel_ids
        ) - total_inj
        for terminal_id, berth_count in _terminal_berth_counts(env, scenario, start_step + t).items():
            vessels_for_terminal = [
                vessel_id
                for vessel_id in env.vessel_ids
                if str(env._routes[vessel_id]["destination"]) == terminal_id
            ]
            prob += (
                pulp.lpSum(unload_active[(vessel_id, t)] for vessel_id in vessels_for_terminal)
                <= min(1, berth_count)
            )
            terminal_free = terminal_capacity_t - terminal_stock[t] + total_inj
            for vessel_id in vessels_for_terminal:
                vessel = env.network.entities[vessel_id]
                prob += unload[(vessel_id, t)] <= terminal_free
                if environment_aligned_service:
                    active = unload_active[(vessel_id, t)]
                    choices = [unload_limit_choice[(vessel_id, t, i)] for i in range(3)]
                    prob += pulp.lpSum(choices) == active
                    big_m = max(
                        float(terminal_capacity_t + max_hourly_injection_t),
                        float(vessel.capacity_t),
                        float(vessel.unloading_rate_tph),
                    )
                    limits = (
                        float(vessel.unloading_rate_tph),
                        cargo[(vessel_id, t)],
                        terminal_free,
                    )
                    for choice, limit in zip(choices, limits):
                        prob += unload[(vessel_id, t)] >= limit - big_m * (1 - choice)
        for pipeline_id, pipeline in env.network._entities_of_type(Pipeline).items():
            prob += (
                pulp.lpSum(well_inj[(well_id, t)] for well_id in _pipeline_wells(env, pipeline_id))
                <= pipeline.max_flow_tph
            )
        for manifold_id, manifold in env.network._entities_of_type(SubseaManifold).items():
            prob += (
                pulp.lpSum(
                    well_inj[(well_id, t)]
                    for well_id in env.network._downstream_of_type(manifold_id, InjectionWell)
                )
                <= manifold.max_flow_tph
            )
    _add_single_well_dynamic_bhp_constraints(prob, env, well_inj, H)

    initial_terminal_t = sum(float(state.entity_inventory_t.get(tid, 0.0)) for tid in env.terminal_ids)
    prob += terminal_stock[0] == initial_terminal_t

    captured_from_operations_t = sum(
        _capture_tonnes(env, scenario, emitter_id, start_step + t)
        for emitter_id in env.emitter_ids
        for t in hours
    )
    stored_expr = pulp.lpSum(well_inj[(well_id, t)] for well_id in env.well_ids for t in hours)
    operating_cost_expr = (
        _sailing_cost_expression(arcs, arc_vars, params)
        + _loading_cost_expression(env, load, params)
        + _unloading_cost_expression(env, unload, params)
        + stored_expr * params.reconditioning_eur_per_t
    )
    vent_expr = pulp.lpSum(
        vent[(emitter_id, t)] for emitter_id in env.emitter_ids for t in hours
    )
    end_unstored_inventory_expr = (
        pulp.lpSum(source_stock[(emitter_id, H)] for emitter_id in env.emitter_ids)
        + terminal_stock[H]
        + pulp.lpSum(cargo[(vessel_id, H)] for vessel_id in env.vessel_ids)
    )
    weighted_objective = (
        objective_weights.operating_cost_weight * operating_cost_expr
        + objective_weights.vent_eur_per_t * vent_expr
        + objective_weights.overflow_risk_eur_per_t * pulp.lpSum(overflow_risk.values())
        - stored_expr * reward_per_t
    )

    use_warm_start = warm_start_native_actions_by_hour is not None
    if warm_start_native_actions_by_hour is not None:
        _apply_native_action_mip_start(
            env,
            arcs,
            starts,
            arc_vars,
            well_rate_options,
            well_choice,
            warm_start_native_actions_by_hour,
            horizon_h=H,
            scenario=scenario,
            start_step=start_step,
            cargo=cargo,
            cargo_positive=cargo_positive,
            cargo_space=cargo_space,
            load=load,
            load_active=load_active,
            load_limit_choice=load_limit_choice,
            unload=unload,
            unload_active=unload_active,
            unload_limit_choice=unload_limit_choice,
            source_stock=source_stock,
            source_ready=source_ready,
            source_overflow_active=source_overflow_active,
            terminal_stock=terminal_stock,
            well_inj=well_inj,
            injection_limit_choice=injection_limit_choice,
            vent=vent,
        )

    def solve_stage(limit_s: float | None, warm_start: bool) -> str:
        solver = _make_cplex_cmd(
            cplex_path=cplex_path,
            time_limit_s=limit_s,
            mip_gap_rel=mip_gap_rel,
            mip_gap_abs=mip_gap_abs,
            threads=threads,
            warm_start=warm_start,
            msg=msg,
        )
        try:
            prob.solve(solver)
        except pulp.PulpSolverError as exc:
            raise RuntimeError(
                "CPLEX_CMD failed. Install IBM ILOG CPLEX, add the cplex executable "
                "to PATH, or pass cplex_path=... to solve_full_scenario_with_cplex()."
            ) from exc
        return _solution_status_label(prob.status, getattr(prob, "sol_status", None))

    if lexicographic_vent_first:
        stage_start = time.perf_counter()
        prob.setObjective(vent_expr)
        status = solve_stage(time_limit_s, use_warm_start)
        if status in {"Optimal", "Integer Feasible"}:
            optimal_vent_t = max(0.0, _value(vent_expr))
            prob += vent_expr <= optimal_vent_t + 1e-3
            remaining_s = None if time_limit_s is None else max(
                1.0,
                float(time_limit_s) - (time.perf_counter() - stage_start),
            )
            stage_two_s = None if remaining_s is None else max(1.0, remaining_s / 2.0)
            prob.setObjective(end_unstored_inventory_expr)
            status = solve_stage(stage_two_s, True)
            if status in {"Optimal", "Integer Feasible"}:
                optimal_end_unstored_t = max(0.0, _value(end_unstored_inventory_expr))
                prob += end_unstored_inventory_expr <= optimal_end_unstored_t + 1e-3
                remaining_s = None if time_limit_s is None else max(
                    1.0,
                    float(time_limit_s) - (time.perf_counter() - stage_start),
                )
                prob.setObjective(operating_cost_expr)
                status = solve_stage(remaining_s, True)
    else:
        prob.setObjective(weighted_objective)
        status = solve_stage(time_limit_s, use_warm_start)
    vessel_actions_by_hour = _extract_vessel_actions(env, H, arcs, arc_vars)
    well_rate_indices_by_hour = _extract_well_rate_indices(env, H, well_rate_options, well_choice)
    native_actions_by_hour = [
        {
            "vessels": [vessel_actions_by_hour[vessel_id][t] for vessel_id in env.vessel_ids],
            "wells": [well_rate_indices_by_hour[well_id][t] for well_id in env.well_ids],
        }
        for t in hours
    ]
    injection_tph = [
        sum(_value(well_inj[(well_id, t)]) for well_id in env.well_ids)
        for t in hours
    ]
    stored_t = sum(injection_tph)
    vented_t = sum(_value(vent[(emitter_id, t)]) for emitter_id in env.emitter_ids for t in hours)
    shortfall_t = _storage_shortfall_t(env, captured_from_operations_t, stored_t)
    initial_source_total_t = sum(float(state.entity_inventory_t.get(eid, 0.0)) for eid in env.emitter_ids)
    initial_cargo_t = sum(float(state.entity_inventory_t.get(vessel_id, 0.0)) for vessel_id in env.vessel_ids)
    initial_in_transit_t = initial_source_total_t + initial_terminal_t + initial_cargo_t
    final_source_t = sum(_value(source_stock[(emitter_id, H)]) for emitter_id in env.emitter_ids)
    final_terminal_t = _value(terminal_stock[H])
    final_cargo_t = sum(_value(cargo[(vessel_id, H)]) for vessel_id in env.vessel_ids)
    in_transit_t = final_source_t + final_terminal_t + final_cargo_t
    in_transit_growth_t = in_transit_t - initial_in_transit_t
    unloaded_t = sum(_value(unload[(vessel_id, t)]) for vessel_id in env.vessel_ids for t in hours)
    cost = _native_cost_breakdown(env, arcs, arc_vars, load, unload, stored_t, params)
    total_cost = cost.operating_cost + vented_t * params.carbon_price_eur_per_t
    overflow_risk_t = sum(_value(var) for var in overflow_risk.values())
    objective_value = control_objective_value(
        objective_weights,
        operating_cost=cost.operating_cost,
        vented_t=vented_t,
        stored_t=stored_t,
        overflow_risk_t=overflow_risk_t,
    )
    net_reward = -objective_value
    departures, arrivals = _extract_departures_and_arrivals(env, arcs, arc_vars, H)
    validation = _validate_solution(
        status=status,
        binary_values=[
            *[arc_vars[index].value() for index in arc_vars],
            *[var.value() for var in cargo_positive.values()],
            *[var.value() for var in cargo_space.values()],
            *[var.value() for var in load_active.values()],
            *[var.value() for var in load_limit_choice.values()],
            *[var.value() for var in unload_active.values()],
            *[var.value() for var in unload_limit_choice.values()],
            *[var.value() for var in source_overflow_active.values()],
            *[var.value() for var in injection_limit_choice.values()],
            *[var.value() for var in well_choice.values()],
        ],
        stored_t=stored_t,
        vented_t=vented_t,
        in_transit_t=in_transit_t,
        captured_from_operations_t=captured_from_operations_t,
        initial_in_transit_t=initial_in_transit_t,
        max_storable_from_deliveries_t=initial_terminal_t + unloaded_t,
        integrality_tol=1e-5 if environment_aligned_service else 1e-6,
    )

    return FullScenarioCplexMilpResult(
        status=status,
        horizon_h=H,
        stored_t=stored_t,
        vented_t=vented_t,
        in_transit_t=in_transit_t,
        in_transit_growth_t=in_transit_growth_t,
        shortfall_t=shortfall_t,
        deliveries=sum(len(arrivals[vessel_id]) for vessel_id in arrivals),
        departures=departures,
        arrivals=arrivals,
        injection_tph=injection_tph,
        vessel_actions_by_hour=vessel_actions_by_hour,
        well_rate_indices_by_hour=well_rate_indices_by_hour,
        native_actions_by_hour=native_actions_by_hour,
        operating_cost=cost.operating_cost,
        total_cost=total_cost,
        cost_per_stored_t=cost.operating_cost / stored_t if stored_t > 0.0 else float("nan"),
        total_cost_per_stored_t=total_cost / stored_t if stored_t > 0.0 else float("nan"),
        storage_reward_eur_per_t=reward_per_t,
        net_reward=net_reward,
        objective_value=objective_value,
        vessel_fuel=cost.vessel_fuel,
        conditioning=cost.conditioning,
        reconditioning=cost.reconditioning,
        loading=cost.loading,
        unloading=cost.unloading,
        captured_from_operations_t=captured_from_operations_t,
        overflow_risk_t=overflow_risk_t,
        is_valid=validation.is_valid,
        validation_error=validation.validation_error,
        max_binary_integrality_violation=validation.max_binary_integrality_violation,
    )


def replay_full_scenario_cplex_plan(
    env,
    result: FullScenarioCplexMilpResult,
    *,
    stored_tol_t: float = 1e-6,
) -> CplexMilpReplayResult:
    """Replay a CPLEX native-action plan through the RL environment.

    This consumes the current ``env`` state by calling ``env.step``. Use it on
    the same initial state used for the MILP solve, or on an equivalent fresh
    reset, when checking whether the MILP plan is executable by the RL wrapper.
    """

    required_fields = frozenset(
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
            "objective_value",
            "overflow_risk_t",
            "injection_tph",
        }
    )
    expectation = ReplayExpectation(
        required_fields=required_fields,
        elapsed_hours=result.horizon_h,
        stored_t=result.stored_t,
        vented_t=result.vented_t,
        captured_t=result.captured_from_operations_t,
        in_transit_t=result.in_transit_t,
        vessel_fuel=result.vessel_fuel,
        conditioning=result.conditioning,
        reconditioning=result.reconditioning,
        loading=result.loading,
        unloading=result.unloading,
        operating_cost=result.operating_cost,
        total_cost=result.total_cost,
        objective_value=result.objective_value,
        overflow_risk_t=result.overflow_risk_t,
        injection_tph=tuple(result.injection_tph),
    )
    report = replay_native_actions(
        env,
        result.native_actions_by_hour,
        horizon_h=result.horizon_h,
        expected=expectation,
        tolerances=ReplayTolerances(mass_t=stored_tol_t),
        copy_env=False,
    )
    actual = report.actual
    stored_gap_t = actual.stored_t - result.stored_t
    return CplexMilpReplayResult(
        elapsed_hours=actual.elapsed_hours,
        stored_t=actual.stored_t,
        vented_t=actual.vented_t,
        operating_cost=actual.operating_cost,
        total_cost=actual.total_cost,
        total_reward=actual.total_reward,
        objective_value=actual.objective_value,
        overflow_risk_t=actual.overflow_risk_t,
        stored_gap_t=stored_gap_t,
        violations=list(report.violations),
        is_executable=report.is_executable,
        is_exact=report.is_exact,
        mismatches=report.mismatches,
        compared_fields=report.compared_fields,
    )


def _make_cplex_cmd(
    *,
    cplex_path: str | None = None,
    time_limit_s: float | None = None,
    mip_gap_rel: float | None = None,
    mip_gap_abs: float | None = None,
    threads: int | None = None,
    warm_start: bool = False,
    msg: bool = False,
):
    _require_pulp()
    return _CplexMipDirectSolutionCmd(
        mip=True,
        msg=1 if msg else 0,
        timeLimit=time_limit_s,
        gapRel=mip_gap_rel,
        gapAbs=mip_gap_abs,
        path=cplex_path,
        threads=threads,
        warmStart=warm_start,
    )


def _require_pulp() -> None:
    if pulp is None:
        raise ImportError("The CPLEX MILP requires PuLP. Install it with `pip install pulp`.")


def _solution_status_label(status: int, solution_status: int | None) -> str:
    _require_pulp()
    if solution_status == pulp.constants.LpSolutionIntegerFeasible:
        return "Integer Feasible"
    return pulp.LpStatus[status]


def _apply_native_action_mip_start(
    env,
    arcs: list[_ActionArc],
    starts: dict[str, _PathStart],
    arc_vars,
    well_rate_options,
    well_choice,
    native_actions_by_hour: list[dict[str, list[int]]],
    *,
    horizon_h: int,
    scenario: Scenario | None = None,
    start_step: int = 0,
    cargo=None,
    cargo_positive=None,
    cargo_space=None,
    load=None,
    load_active=None,
    load_limit_choice=None,
    unload=None,
    unload_active=None,
    unload_limit_choice=None,
    source_stock=None,
    source_ready=None,
    source_overflow_active=None,
    terminal_stock=None,
    well_inj=None,
    injection_limit_choice=None,
    vent=None,
    shortfall=None,
    required_storage_t: float | None = None,
) -> None:
    for var in arc_vars.values():
        var.setInitialValue(0)

    arc_by_step: dict[tuple[str, int, str, str], int] = {}
    for index, arc in enumerate(arcs):
        arc_by_step[(arc.vessel_id, arc.start_h, arc.origin_id, arc.destination_id)] = index

    selected_wait_node: dict[tuple[str, int], str] = {}
    for vessel_index, vessel_id in enumerate(env.vessel_ids):
        start = starts[vessel_id]
        if start.node_id is None or start.start_h >= horizon_h:
            continue
        node_id = start.node_id
        t = start.start_h
        while t < horizon_h:
            action = _warm_start_vessel_action(native_actions_by_hour, vessel_index, t)
            destination_id = _native_action_destination(env, vessel_id, action)
            if destination_id is None or destination_id == node_id:
                destination_id = node_id
            arc_index = arc_by_step.get((vessel_id, t, node_id, destination_id))
            if arc_index is None and destination_id != node_id:
                destination_id = node_id
                arc_index = arc_by_step.get((vessel_id, t, node_id, node_id))
            if arc_index is None:
                break
            arc_vars[arc_index].setInitialValue(1)
            selected_arc = arcs[arc_index]
            if not selected_arc.is_sailing:
                selected_wait_node[(vessel_id, t)] = selected_arc.origin_id
            node_id = selected_arc.destination_id
            t = max(t + 1, selected_arc.end_h)

    for well_index, well_id in enumerate(env.well_ids):
        for t in range(min(horizon_h, len(native_actions_by_hour))):
            rate_index = _warm_start_well_rate_index(native_actions_by_hour, well_index, t)
            if rate_index not in well_rate_options[(well_id, t)]:
                continue
            for candidate in well_rate_options[(well_id, t)]:
                well_choice[(well_id, t, candidate)].setInitialValue(1 if candidate == rate_index else 0)

    if scenario is None:
        return

    _seed_native_action_state_values(
        env,
        scenario,
        start_step,
        selected_wait_node,
        well_rate_options,
        native_actions_by_hour,
        horizon_h,
        cargo=cargo,
        cargo_positive=cargo_positive,
        cargo_space=cargo_space,
        load=load,
        load_active=load_active,
        load_limit_choice=load_limit_choice,
        unload=unload,
        unload_active=unload_active,
        unload_limit_choice=unload_limit_choice,
        source_stock=source_stock,
        source_ready=source_ready,
        source_overflow_active=source_overflow_active,
        terminal_stock=terminal_stock,
        well_inj=well_inj,
        injection_limit_choice=injection_limit_choice,
        vent=vent,
        shortfall=shortfall,
        required_storage_t=required_storage_t,
    )


def _warm_start_vessel_action(native_actions_by_hour: list[dict[str, list[int]]], vessel_index: int, t: int) -> int:
    try:
        return int(native_actions_by_hour[t]["vessels"][vessel_index])
    except (IndexError, KeyError, TypeError, ValueError):
        return VESSEL_WAIT


def _warm_start_well_rate_index(native_actions_by_hour: list[dict[str, list[int]]], well_index: int, t: int) -> int:
    try:
        return int(native_actions_by_hour[t]["wells"][well_index])
    except (IndexError, KeyError, TypeError, ValueError):
        return 0


def _native_action_destination(env, vessel_id: str, action: int) -> str | None:
    if action == VESSEL_WAIT:
        return None
    if action == VESSEL_GO_TERMINAL:
        return str(env._routes[vessel_id]["destination"])
    emitter_index = action - VESSEL_GO_EMITTER_BASE
    if 0 <= emitter_index < len(env.emitter_ids):
        return env.emitter_ids[emitter_index]
    return None


def _seed_native_action_state_values(
    env,
    scenario: Scenario,
    start_step: int,
    selected_wait_node: dict[tuple[str, int], str],
    well_rate_options,
    native_actions_by_hour: list[dict[str, list[int]]],
    horizon_h: int,
    *,
    cargo=None,
    cargo_positive=None,
    cargo_space=None,
    load=None,
    load_active=None,
    load_limit_choice=None,
    unload=None,
    unload_active=None,
    unload_limit_choice=None,
    source_stock=None,
    source_ready=None,
    source_overflow_active=None,
    terminal_stock=None,
    well_inj=None,
    injection_limit_choice=None,
    vent=None,
    shortfall=None,
    required_storage_t: float | None = None,
) -> None:
    state = env.simulator.state
    cargo_values = {
        vessel_id: float(state.entity_inventory_t.get(vessel_id, 0.0))
        for vessel_id in env.vessel_ids
    }
    source_values = {
        emitter_id: float(state.entity_inventory_t.get(emitter_id, 0.0))
        for emitter_id in env.emitter_ids
    }
    terminal_value = sum(float(state.entity_inventory_t.get(terminal_id, 0.0)) for terminal_id in env.terminal_ids)
    terminal_capacity_t = _terminal_capacity_t(env)
    total_stored_t = 0.0

    for vessel_id, value in cargo_values.items():
        _set_start_value(cargo, (vessel_id, 0), value)
    for emitter_id, value in source_values.items():
        _set_start_value(source_stock, (emitter_id, 0), value)
    _set_start_value(terminal_stock, 0, terminal_value)

    for t in range(horizon_h):
        load_values: dict[tuple[str, str], float] = {}
        unload_values: dict[str, float] = {}
        request_by_well = _warm_start_well_requests(env, well_rate_options, native_actions_by_hour, t)
        total_request = sum(request_by_well.values())

        for vessel_id in env.vessel_ids:
            vessel = env.network.entities[vessel_id]
            cargo_t = cargo_values[vessel_id]
            _set_start_value(cargo, (vessel_id, t), cargo_t)
            _set_start_value(cargo_positive, (vessel_id, t), 1 if cargo_t > 1e-9 else 0)
            _set_start_value(cargo_space, (vessel_id, t), 1 if cargo_t < vessel.capacity_t - 1e-9 else 0)

        for emitter_id in env.emitter_ids:
            emitter = env.network.entities[emitter_id]
            capture_t = _capture_tonnes(env, scenario, emitter_id, start_step + t)
            pre_load = source_values[emitter_id] + capture_t
            ready = min(pre_load, emitter.buffer_capacity_t)
            overflow = 1 if pre_load > emitter.buffer_capacity_t + 1e-9 else 0
            _set_start_value(source_ready, (emitter_id, t), ready)
            _set_start_value(source_overflow_active, (emitter_id, t), overflow)
            _set_start_value(vent, (emitter_id, t), pre_load - ready)

            loader_id = _warm_start_loader_for_emitter(env, selected_wait_node, cargo_values, emitter_id, t)
            loaded_total = 0.0
            for vessel_id in env.vessel_ids:
                amount = 0.0
                active = 0
                if vessel_id == loader_id:
                    vessel = env.network.entities[vessel_id]
                    rate = min(vessel.loading_rate_tph, emitter.loading_rate_tph)
                    amount = min(rate, ready, vessel.capacity_t - cargo_values[vessel_id])
                    active = 1 if amount > 1e-9 else 0
                load_values[(vessel_id, emitter_id)] = amount
                loaded_total += amount
                _set_start_value(load, (vessel_id, emitter_id, t), amount)
                _set_start_value(load_active, (vessel_id, emitter_id, t), active)
                _seed_limit_choice(
                    load_limit_choice,
                    (vessel_id, emitter_id, t),
                    active,
                    amount,
                    [
                        min(env.network.entities[vessel_id].loading_rate_tph, emitter.loading_rate_tph),
                        ready,
                        env.network.entities[vessel_id].capacity_t - cargo_values[vessel_id],
                    ],
                )
            source_values[emitter_id] = ready - loaded_total
            _set_start_value(source_stock, (emitter_id, t + 1), source_values[emitter_id])

        unload_vessel_id = _warm_start_unload_vessel(env, selected_wait_node, cargo_values, t)
        if unload_vessel_id is not None:
            vessel = env.network.entities[unload_vessel_id]
            unload_amount = _fixed_point_unload_amount(
                terminal_capacity_t,
                terminal_value,
                total_request,
                vessel.unloading_rate_tph,
                cargo_values[unload_vessel_id],
            )
            unload_values[unload_vessel_id] = unload_amount
        total_unload = sum(unload_values.values())
        total_injection = min(total_request, terminal_value + total_unload)
        terminal_free = terminal_capacity_t - terminal_value + total_injection

        for vessel_id in env.vessel_ids:
            amount = unload_values.get(vessel_id, 0.0)
            active = 1 if amount > 1e-9 else 0
            _set_start_value(unload, (vessel_id, t), amount)
            _set_start_value(unload_active, (vessel_id, t), active)
            vessel = env.network.entities[vessel_id]
            _seed_limit_choice(
                unload_limit_choice,
                (vessel_id, t),
                active,
                amount,
                [vessel.unloading_rate_tph, cargo_values[vessel_id], terminal_free],
            )

        remaining_injection = total_injection
        for well_id in env.well_ids:
            amount = min(request_by_well.get(well_id, 0.0), remaining_injection)
            remaining_injection -= amount
            _set_start_value(well_inj, (well_id, t), amount)
        _set_start_value(injection_limit_choice, (t, 0), 1 if total_request <= terminal_value + total_unload + 1e-9 else 0)
        _set_start_value(injection_limit_choice, (t, 1), 0 if total_request <= terminal_value + total_unload + 1e-9 else 1)

        for vessel_id in env.vessel_ids:
            cargo_values[vessel_id] = (
                cargo_values[vessel_id]
                + sum(load_values.get((vessel_id, emitter_id), 0.0) for emitter_id in env.emitter_ids)
                - unload_values.get(vessel_id, 0.0)
            )
            _set_start_value(cargo, (vessel_id, t + 1), cargo_values[vessel_id])

        terminal_value = terminal_value + total_unload - total_injection
        total_stored_t += total_injection
        _set_start_value(terminal_stock, t + 1, terminal_value)

    if required_storage_t is not None:
        shortfall_t = max(0.0, required_storage_t - (float(env.cumulative_stored_t) + total_stored_t))
        _set_start_value(shortfall, None, shortfall_t)


def _warm_start_well_requests(env, well_rate_options, native_actions_by_hour, t: int) -> dict[str, float]:
    requests: dict[str, float] = {}
    for well_index, well_id in enumerate(env.well_ids):
        rate_index = _warm_start_well_rate_index(native_actions_by_hour, well_index, t)
        if rate_index not in well_rate_options[(well_id, t)]:
            rate_index = 0
        requests[well_id] = mtpa_to_tph(WELL_RATE_LEVELS_MTPA[rate_index])
    return requests


def _warm_start_loader_for_emitter(
    env,
    selected_wait_node: dict[tuple[str, int], str],
    cargo_values: dict[str, float],
    emitter_id: str,
    t: int,
) -> str | None:
    for vessel_id in env.vessel_ids:
        if selected_wait_node.get((vessel_id, t)) != emitter_id:
            continue
        vessel = env.network.entities[vessel_id]
        if cargo_values[vessel_id] < vessel.capacity_t - 1e-9:
            return vessel_id
    return None


def _warm_start_unload_vessel(
    env,
    selected_wait_node: dict[tuple[str, int], str],
    cargo_values: dict[str, float],
    t: int,
) -> str | None:
    for terminal_id in env.terminal_ids:
        vessels_for_terminal = [
            vessel_id
            for vessel_id in env.vessel_ids
            if str(env._routes[vessel_id]["destination"]) == terminal_id
        ]
        for vessel_id in vessels_for_terminal:
            if selected_wait_node.get((vessel_id, t)) == terminal_id and cargo_values[vessel_id] > 1e-9:
                return vessel_id
    return None


def _fixed_point_unload_amount(
    terminal_capacity_t: float,
    terminal_stock_t: float,
    total_request_t: float,
    unload_rate_t: float,
    cargo_t: float,
) -> float:
    amount = 0.0
    for _ in range(8):
        injection_t = min(total_request_t, terminal_stock_t + amount)
        terminal_free_t = terminal_capacity_t - terminal_stock_t + injection_t
        next_amount = min(unload_rate_t, cargo_t, max(0.0, terminal_free_t))
        if abs(next_amount - amount) <= 1e-9:
            return next_amount
        amount = next_amount
    return amount


def _seed_limit_choice(var_map, key_prefix, active: int, amount: float, limits: list[float]) -> None:
    if var_map is None:
        return
    if not active:
        for limit_index in range(3):
            _set_start_value(var_map, (*key_prefix, limit_index), 0)
        return
    differences = [abs(amount - max(0.0, limit)) for limit in limits]
    selected = min(range(3), key=differences.__getitem__)
    for limit_index in range(3):
        _set_start_value(var_map, (*key_prefix, limit_index), 1 if limit_index == selected else 0)


def _set_start_value(var_map, key, value: float) -> None:
    if var_map is None:
        return
    var = var_map if key is None else var_map.get(key)
    if var is not None:
        var.setInitialValue(max(0.0, float(value)))


def _empty_result() -> FullScenarioCplexMilpResult:
    return FullScenarioCplexMilpResult(
        status="Empty horizon",
        horizon_h=0,
        stored_t=0.0,
        vented_t=0.0,
        in_transit_t=0.0,
        in_transit_growth_t=0.0,
        shortfall_t=0.0,
        deliveries=0,
        departures={},
        arrivals={},
        injection_tph=[],
        vessel_actions_by_hour={},
        well_rate_indices_by_hour={},
        native_actions_by_hour=[],
        operating_cost=0.0,
        total_cost=0.0,
        cost_per_stored_t=float("nan"),
        total_cost_per_stored_t=float("nan"),
    )


def _build_action_arcs(
    env,
    scenario: Scenario,
    start_step: int,
    horizon_h: int,
) -> tuple[list[_ActionArc], dict[str, _PathStart]]:
    arcs: list[_ActionArc] = []
    starts = {vessel_id: _path_start(env, scenario, start_step, vessel_id, horizon_h) for vessel_id in env.vessel_ids}
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
                        scenario=scenario,
                        start_step=start_step + t,
                        max_horizon_h=horizon_h - t,
                    )
                    arrives_within_horizon = duration_h <= horizon_h - t
                    end_h = min(t + duration_h, horizon_h)
                    arcs.append(
                        _ActionArc(
                            vessel_id=vessel_id,
                            start_h=t,
                            end_h=end_h,
                            origin_id=origin_id,
                            destination_id=destination_id,
                            action=_action_to_destination(env, vessel_id, destination_id),
                            is_sailing=True,
                            arrives_within_horizon=arrives_within_horizon,
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


def _nodes_for_vessel(env, vessel_id: str) -> list[str]:
    terminal_id = str(env._routes[vessel_id]["destination"])
    return list(dict.fromkeys([*env.emitter_ids, terminal_id]))


def _path_start(env, scenario: Scenario, start_step: int, vessel_id: str, horizon_h: int) -> _PathStart:
    vstate = env.simulator.vessel_states[vessel_id]
    if vstate["mode"] == "berthed":
        return _PathStart(0, str(vstate["berth"]))
    remaining_h = _remaining_sailing_hours(env, scenario, start_step, vessel_id, max_horizon_h=horizon_h)
    if remaining_h >= horizon_h:
        return _PathStart(horizon_h, None)
    return _PathStart(remaining_h, str(vstate["destination"]))


def _remaining_sailing_hours(
    env,
    scenario: Scenario,
    start_step: int,
    vessel_id: str,
    *,
    max_horizon_h: int,
) -> int:
    route = env._routes[vessel_id]
    vstate = env.simulator.vessel_states[vessel_id]
    distance_km = float(vstate.get("distance_km") or route["distance_km"])
    remaining_km = max(0.0, distance_km * (1.0 - float(vstate["progress"])))
    return _sailing_duration_h_for_distance(
        env,
        scenario,
        vessel_id,
        origin_id=str(vstate.get("origin", route["origin"])),
        destination_id=str(vstate["destination"]),
        distance_km=remaining_km,
        start_step=start_step,
        max_horizon_h=max_horizon_h,
    )


def _sail_hours_between(
    env,
    origin_id: str,
    destination_id: str,
    vessel_id: str,
    *,
    scenario: Scenario,
    start_step: int,
    max_horizon_h: int,
) -> int:
    route = env._routes[vessel_id]
    if {origin_id, destination_id} == {str(route["origin"]), str(route["destination"])}:
        distance_km = float(route["distance_km"])
    else:
        distance_km = _dynamic_leg_distance_km(env, route, origin_id, destination_id)
    return _sailing_duration_h_for_distance(
        env,
        scenario,
        vessel_id,
        origin_id=origin_id,
        destination_id=destination_id,
        distance_km=distance_km,
        start_step=start_step,
        max_horizon_h=max_horizon_h,
    )


def _sailing_duration_h_for_distance(
    env,
    scenario: Scenario,
    vessel_id: str,
    *,
    origin_id: str,
    destination_id: str,
    distance_km: float,
    start_step: int,
    max_horizon_h: int,
) -> int:
    route = env._routes[vessel_id]
    speed_kmh = max(0.0, float(route["speed_knots"])) * KNOTS_TO_KMH
    if distance_km <= 1e-9:
        return 0
    if speed_kmh <= 1e-9:
        return max_horizon_h + 1
    covered_km = 0.0
    for elapsed_h in range(1, max(1, int(max_horizon_h)) + 1):
        step = start_step + elapsed_h - 1
        speed_factor = _scenario_leg_speed_factor(scenario, origin_id, destination_id, vessel_id, step)
        covered_km += speed_kmh * speed_factor
        if covered_km >= distance_km - 1e-9:
            return elapsed_h
    return max_horizon_h + 1


def _scenario_leg_speed_factor(
    scenario: Scenario,
    origin_id: str,
    destination_id: str,
    vessel_id: str,
    step: int,
) -> float:
    vessel_factor = _scenario_series_value(scenario.vessel_speed_factor, vessel_id, step, 1.0)
    leg_id = f"{origin_id}->{destination_id}"
    return max(0.0, float(_scenario_series_value(scenario.leg_speed_factor, leg_id, step, vessel_factor)))


def _dynamic_leg_distance_km(env, route: dict, origin_id: str, destination_id: str) -> float:
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


def _action_to_destination(env, vessel_id: str, destination_id: str) -> int:
    if destination_id == str(env._routes[vessel_id]["destination"]):
        return VESSEL_GO_TERMINAL
    return env.vessel_go_emitter_action(destination_id)


def _wait_expr(arc_vars, wait_arc: dict[tuple[str, str, int], int], vessel_id: str, node_id: str, t: int):
    index = wait_arc.get((vessel_id, node_id, t))
    return 0 if index is None else arc_vars[index]


def _well_rate_options_by_hour(env, scenario: Scenario, horizon_h: int) -> dict[tuple[str, int], list[int]]:
    start_step = scenario.step_index(_current_start_hour(env))
    options: dict[tuple[str, int], list[int]] = {}
    for t in range(horizon_h):
        future_state = _future_state_for_step(env, scenario, start_step + t)
        interval_start_h = future_state.time_h
        evaluation_time_h = interval_start_h + env.network.time_step_hours
        for well_id in env.well_ids:
            physical_max = _physical_well_max_tph(env, future_state, well_id)
            if _uses_single_well_dynamic_bhp(env, well_id):
                mask = tuple(mtpa_to_tph(rate_mtpa) <= physical_max + 1e-9 for rate_mtpa in WELL_RATE_LEVELS_MTPA)
            else:
                mask = pressure_limited_rate_level_mask(
                    env.network,
                    future_state,
                    well_id,
                    rate_levels_mtpa=WELL_RATE_LEVELS_MTPA,
                    physical_max_rate_tph=physical_max,
                    evaluation_time_h=evaluation_time_h,
                    interval_start_h=interval_start_h,
                )
            feasible = [index for index, allowed in enumerate(mask) if allowed]
            options[(well_id, t)] = feasible or [0]
    return options


def _uses_single_well_dynamic_bhp(env, well_id: str) -> bool:
    reservoir_id = env.network._single_downstream_of_type(well_id, Reservoir)
    if reservoir_id is None:
        return False
    reservoir = env.network.entities[reservoir_id]
    assert isinstance(reservoir, Reservoir)
    if reservoir.line_source_parameters is None or reservoir.well_bottomhole_pressure_limit_bar is None:
        return False
    upstream_wells = env.network._upstream_of_type(reservoir_id, InjectionWell)
    return upstream_wells == [well_id]


def _add_single_well_dynamic_bhp_constraints(prob, env, well_inj, horizon_h: int) -> None:
    """Add decision-dependent BHP constraints for true single-well reservoirs.

    Multi-well interference is intentionally left to the existing pressure mask
    path until cross-well response terms are modelled explicitly.
    """
    state = env.simulator.state
    start_h = _current_start_hour(env)
    dt = float(env.network.time_step_hours)
    for well_id in env.well_ids:
        if not _uses_single_well_dynamic_bhp(env, well_id):
            continue
        reservoir_id = env.network._single_downstream_of_type(well_id, Reservoir)
        reservoir = env.network.entities[reservoir_id]
        assert isinstance(reservoir, Reservoir)
        alpha = _reservoir_pressure_bar_per_tonne(reservoir)
        initial_inventory_t = float(state.entity_inventory_t.get(reservoir_id, 0.0))
        reservoir_pressure_const = reservoir.initial_pressure_bar + alpha * initial_inventory_t
        limit_bar = float(reservoir.well_bottomhole_pressure_limit_bar)
        for t in range(horizon_h):
            evaluation_h = start_h + (t + 1) * dt
            response_const, response_coeffs = _single_well_line_source_response_terms(
                env,
                well_id,
                horizon_index=t,
                evaluation_h=evaluation_h,
            )
            cumulative_injected_expr = pulp.lpSum(well_inj[(well_id, tau)] * dt for tau in range(t + 1))
            line_source_expr = response_const + pulp.lpSum(
                response_coeffs[tau] * well_inj[(well_id, tau)]
                for tau in range(t + 1)
            )
            prob += reservoir_pressure_const + alpha * cumulative_injected_expr + line_source_expr <= limit_bar


def _reservoir_pressure_bar_per_tonne(reservoir: Reservoir) -> float:
    if reservoir.storage_capacity_t <= 0.0:
        return 0.0
    return (reservoir.pressure_at_capacity_bar - reservoir.initial_pressure_bar) / reservoir.storage_capacity_t


def _single_well_line_source_response_terms(
    env,
    well_id: str,
    *,
    horizon_index: int,
    evaluation_h: float,
) -> tuple[float, list[float]]:
    reservoir_id = env.network._single_downstream_of_type(well_id, Reservoir)
    reservoir = env.network.entities[reservoir_id]
    assert isinstance(reservoir, Reservoir)
    zero_params = replace(reservoir.line_source_parameters, initial_pressure_bar=0.0)
    start_h = _current_start_hour(env)
    zero_future = [0.0] * (horizon_index + 1)
    zero_history = _history_with_future_rates(env, well_id, start_h, zero_future)
    elapsed_days = evaluation_h / 24.0
    response_const = variable_rate_bottomhole_pressure_bar(
        zero_params,
        zero_history,
        elapsed_days=elapsed_days,
    )
    coeffs: list[float] = []
    for tau in range(horizon_index + 1):
        basis_future = list(zero_future)
        basis_future[tau] = 1.0
        basis_history = _history_with_future_rates(env, well_id, start_h, basis_future)
        basis_response = variable_rate_bottomhole_pressure_bar(
            zero_params,
            basis_history,
            elapsed_days=elapsed_days,
        )
        coeffs.append(basis_response - response_const)
    return response_const, coeffs


def _history_with_future_rates(env, well_id: str, start_h: float, future_rates_tph: list[float]) -> list[tuple[float, float]]:
    history = list(env.simulator.state.injection_rate_history_tph.get(well_id, []))
    for offset_h, rate_tph in enumerate(future_rates_tph):
        interval_start_h = start_h + offset_h * env.network.time_step_hours
        history.append((interval_start_h, float(rate_tph)))
    return [(hour / 24.0, tph_to_mtpa(rate_tph)) for hour, rate_tph in history]


def _future_state_for_step(env, scenario: Scenario, scenario_step: int):
    state = env.simulator.state.copy()
    state.time_h = scenario_step * scenario.time_step_hours
    scenario.apply_to_state(state, state.time_h)
    return state


def _physical_well_max_tph(env, state, well_id: str) -> float:
    well = env.network.entities[well_id]
    assert isinstance(well, InjectionWell)
    if not bool(state.well_available.get(well_id, well.available)):
        return 0.0
    injectivity = max(0.0, float(state.injectivity_factor.get(well_id, 1.0)))
    return well.max_injection_tph * injectivity


def _capture_tonnes(env, scenario: Scenario, emitter_id: str, scenario_step: int) -> float:
    emitter = env.network.entities[emitter_id]
    assert isinstance(emitter, Emitter)
    availability = _scenario_series_value(scenario.emitter_availability, emitter_id, scenario_step, emitter.availability)
    time_h = scenario_step * scenario.time_step_hours
    return emitter.capture_rate_tph_at(time_h) * max(0.0, float(availability))


def _add_overflow_risk_constraints(
    prob,
    env,
    scenario: Scenario,
    start_step: int,
    horizon_h: int,
    source_stock,
    lookahead_h: float,
):
    if lookahead_h <= 0.0:
        return {}
    risk = {
        (emitter_id, t): pulp.LpVariable(f"overflow_risk_{emitter_id}_{t}", lowBound=0.0)
        for emitter_id in env.emitter_ids
        for t in range(horizon_h)
    }
    for emitter_id in env.emitter_ids:
        emitter = env.network.entities[emitter_id]
        for t in range(horizon_h):
            capture_tph = _capture_tonnes(env, scenario, emitter_id, start_step + t)
            prob += risk[(emitter_id, t)] >= (
                capture_tph * lookahead_h
                - emitter.buffer_capacity_t
                + source_stock[(emitter_id, t + 1)]
            )
    return risk


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


def _terminal_berth_counts(env, scenario: Scenario, scenario_step: int) -> dict[str, int]:
    state = _future_state_for_step(env, scenario, scenario_step)
    return {
        terminal_id: max(0, int(state.berth_count_override.get(terminal_id, terminal.berth_count)))
        for terminal_id, terminal in env.network._entities_of_type(Terminal).items()
    }


def _pipeline_wells(env, pipeline_id: str) -> list[str]:
    wells = list(env.network._downstream_of_type(pipeline_id, InjectionWell))
    for manifold_id in env.network._downstream_of_type(pipeline_id, SubseaManifold):
        wells.extend(env.network._downstream_of_type(manifold_id, InjectionWell))
    return wells


def _sailing_fuel_hours(arc: _ActionArc) -> int:
    if not arc.is_sailing:
        return 0
    if arc.arrives_within_horizon:
        return max(0, arc.duration_h - 1)
    return arc.duration_h


def _sailing_cost_expression(arcs: list[_ActionArc], arc_vars, params: EconomicParameters):
    return pulp.lpSum(
        _sailing_fuel_hours(arc) * params.vessel_fuel_eur_per_h_sailing * arc_vars[index]
        for index, arc in enumerate(arcs)
        if arc.is_sailing
    )


def _loading_cost_expression(env, load, params: EconomicParameters):
    terms = []
    for (vessel_id, emitter_id, _t), var in load.items():
        vessel = env.network.entities[vessel_id]
        emitter = env.network.entities[emitter_id]
        load_rate_tph = max(1e-9, min(vessel.loading_rate_tph, emitter.loading_rate_tph))
        terms.append(var * (params.conditioning_eur_per_t + params.hoteling_fuel_eur_per_h / load_rate_tph))
    return pulp.lpSum(terms)


def _unloading_cost_expression(env, unload, params: EconomicParameters):
    terms = []
    for (vessel_id, _t), var in unload.items():
        vessel = env.network.entities[vessel_id]
        unload_rate_tph = max(1e-9, vessel.unloading_rate_tph)
        terms.append(var * (params.hoteling_fuel_eur_per_h / unload_rate_tph))
    return pulp.lpSum(terms)


def _extract_vessel_actions(env, horizon_h: int, arcs: list[_ActionArc], arc_vars) -> dict[str, list[int]]:
    actions = {vessel_id: [VESSEL_WAIT] * horizon_h for vessel_id in env.vessel_ids}
    for index, arc in enumerate(arcs):
        if round(_value(arc_vars[index])) == 1 and arc.start_h < horizon_h:
            actions[arc.vessel_id][arc.start_h] = arc.action
    return actions


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


def _native_cost_breakdown(env, arcs, arc_vars, load, unload, stored_t: float, params: EconomicParameters) -> CplexCostBreakdown:
    vessel_fuel = sum(
        _sailing_fuel_hours(arc) * params.vessel_fuel_eur_per_h_sailing
        for index, arc in enumerate(arcs)
        if arc.is_sailing and round(_value(arc_vars[index])) == 1
    )
    loaded_t = sum(_value(var) for var in load.values())
    conditioning = loaded_t * params.conditioning_eur_per_t
    loading = 0.0
    for (vessel_id, emitter_id, _t), var in load.items():
        amount_t = _value(var)
        vessel = env.network.entities[vessel_id]
        emitter = env.network.entities[emitter_id]
        rate_tph = max(1e-9, min(vessel.loading_rate_tph, emitter.loading_rate_tph))
        loading += (amount_t / rate_tph) * params.hoteling_fuel_eur_per_h
    unloading = 0.0
    for (vessel_id, _t), var in unload.items():
        amount_t = _value(var)
        vessel = env.network.entities[vessel_id]
        unloading += (amount_t / max(1e-9, vessel.unloading_rate_tph)) * params.hoteling_fuel_eur_per_h
    return CplexCostBreakdown(
        vessel_fuel=vessel_fuel,
        conditioning=conditioning,
        reconditioning=stored_t * params.reconditioning_eur_per_t,
        loading=loading,
        unloading=unloading,
    )


def _extract_departures_and_arrivals(env, arcs: list[_ActionArc], arc_vars, horizon_h: int) -> tuple[dict[str, list[int]], dict[str, list[int]]]:
    departures = {vessel_id: [] for vessel_id in env.vessel_ids}
    arrivals = {vessel_id: [] for vessel_id in env.vessel_ids}
    terminal_ids = set(env.terminal_ids)
    for index, arc in enumerate(arcs):
        if not arc.is_sailing or round(_value(arc_vars[index])) != 1:
            continue
        departures[arc.vessel_id].append(arc.start_h)
        if arc.destination_id in terminal_ids and arc.end_h <= horizon_h:
            arrivals[arc.vessel_id].append(arc.end_h)
    return departures, arrivals


def _extract_vessel_params(env) -> list[CplexVesselParams]:
    vessels: list[CplexVesselParams] = []
    for vessel_id, route in sorted(env._routes.items()):
        vessel = env.network.entities[vessel_id]
        if not isinstance(vessel, Vessel):
            continue
        vessels.append(
            CplexVesselParams(
                vessel_id=vessel_id,
                source_id=str(route["origin"]),
                capacity_t=float(vessel.capacity_t),
                load_rate_tph=float(vessel.loading_rate_tph),
                unload_rate_tph=float(vessel.unloading_rate_tph),
                speed_knots=float(route["speed_knots"]),
            )
        )
    return vessels


def _nominal_injection_capacity_tph(env) -> float:
    pipelines = [e.max_flow_tph for e in env.network._entities_of_type(Pipeline).values()]
    manifolds = [e.max_flow_tph for e in env.network._entities_of_type(SubseaManifold).values()]
    well_sum = sum(e.max_injection_tph for e in env.network._entities_of_type(InjectionWell).values())
    return min([well_sum] + pipelines + manifolds)


def _terminal_capacity_t(env) -> float:
    return sum(t.storage_capacity_t for t in env.network._entities_of_type(Terminal).values())


def _source_buffer_capacity_t(env) -> float:
    return sum(e.buffer_capacity_t for e in env.network._entities_of_type(Emitter).values())


def _current_start_hour(env) -> int:
    simulator = getattr(env, "simulator", None)
    if simulator is None:
        return 0
    return int(round(simulator.state.time_h))


def _initial_terminal_inventory_t(env, scenario: Scenario) -> float:
    simulator = getattr(env, "simulator", None)
    if simulator is not None:
        return sum(float(simulator.state.entity_inventory_t.get(tid, 0.0)) for tid in env.terminal_ids)
    return sum(float(scenario.initial_inventory_t.get(tid, 0.0)) for tid in env.terminal_ids)


def _initial_source_inventory_t(env, scenario: Scenario) -> float:
    simulator = getattr(env, "simulator", None)
    if simulator is not None:
        return sum(float(simulator.state.entity_inventory_t.get(eid, 0.0)) for eid in env.emitter_ids)
    return sum(float(scenario.initial_inventory_t.get(eid, 0.0)) for eid in env.emitter_ids)


def _captured_until(env, scenario: Scenario, start_step: int, steps: int) -> float:
    total = 0.0
    for offset in range(steps):
        index = start_step + offset
        time_h = float(index) * scenario.time_step_hours
        for emitter_id in env.emitter_ids:
            emitter = env.network.entities[emitter_id]
            availability = _scenario_series_value(
                scenario.emitter_availability,
                emitter_id,
                index,
                emitter.availability,
            )
            total += emitter.capture_rate_tph_at(time_h) * max(0.0, float(availability))
    return total


def _scenario_injection_cap_tph(env, scenario: Scenario, scenario_step: int, nominal_cap_tph: float) -> float:
    well_cap = 0.0
    for well_id, well in env.network._entities_of_type(InjectionWell).items():
        available = bool(_scenario_series_value(scenario.well_available, well_id, scenario_step, well.available))
        if not available:
            continue
        factor = _scenario_series_value(scenario.injectivity_factor, well_id, scenario_step, 1.0)
        well_cap += well.max_injection_tph * max(0.0, float(factor))
    return min(nominal_cap_tph, well_cap)


def _terminal_berth_count(env, _scenario: Scenario, _scenario_step: int) -> int:
    return max(0, sum(int(t.berth_count) for t in env.network._entities_of_type(Terminal).values()))


def _departure_options(
    env,
    vessels: list[CplexVesselParams],
    scenario: Scenario,
    start_step: int,
    horizon_h: int,
) -> list[_DepartureOption]:
    options: list[_DepartureOption] = []
    for vessel in vessels:
        earliest_depart = vessel.load_dur_h
        for depart_h in range(earliest_depart, horizon_h):
            outbound_h = _sailing_duration_h(env, scenario, vessel.vessel_id, start_step + depart_h, horizon_h - depart_h)
            arrival_h = depart_h + outbound_h
            if arrival_h >= horizon_h:
                continue
            return_start_h = arrival_h + vessel.unload_dur_h
            return_h = _sailing_duration_h(
                env,
                scenario,
                vessel.vessel_id,
                start_step + return_start_h,
                max(1, horizon_h - return_start_h),
            )
            next_depart_h = return_start_h + return_h + vessel.load_dur_h
            options.append(
                _DepartureOption(
                    vessel_id=vessel.vessel_id,
                    depart_h=depart_h,
                    arrival_h=arrival_h,
                    next_depart_h=next_depart_h,
                    outbound_sail_h=outbound_h,
                    return_sail_h=return_h,
                    unload_dur_h=vessel.unload_dur_h,
                )
            )
    return options


def _sailing_duration_h(
    env,
    scenario: Scenario,
    vessel_id: str,
    scenario_start_step: int,
    max_horizon_h: int,
) -> int:
    route = env._routes[vessel_id]
    distance_km = float(route["distance_km"])
    speed_kmh = float(route["speed_knots"]) * KNOTS_TO_KMH
    if distance_km <= 1e-9:
        return 0
    if speed_kmh <= 1e-9:
        return max_horizon_h + 1
    covered_km = 0.0
    for elapsed_h in range(1, max(1, max_horizon_h) + 1):
        step = scenario_start_step + elapsed_h - 1
        factor = _scenario_series_value(scenario.vessel_speed_factor, vessel_id, step, 1.0)
        covered_km += speed_kmh * max(0.0, float(factor))
        if covered_km >= distance_km - 1e-9:
            return elapsed_h
    return max_horizon_h + 1


def _scenario_series_value(series_by_id, entity_id: str, step: int, default):
    series = series_by_id.get(entity_id)
    if not series:
        return default
    index = max(0, min(len(series) - 1, step))
    return series[index]


def _unload_profile(vessel: CplexVesselParams) -> list[float]:
    profile = []
    remaining = vessel.capacity_t
    for _ in range(vessel.unload_dur_h):
        amount = min(vessel.unload_rate_tph, remaining)
        profile.append(amount)
        remaining -= amount
    return profile


def _cumulative_unloaded_at(profile: list[float], offset_h: int) -> float:
    if offset_h < 0:
        return 0.0
    return sum(profile[: min(offset_h + 1, len(profile))])


def _departure_cost_expression(options, vessels, depart, stored_expr, params: EconomicParameters):
    vessel_by_id = {v.vessel_id: v for v in vessels}
    per_departure = pulp.lpSum(
        (
            (option.outbound_sail_h + option.return_sail_h) * params.vessel_fuel_eur_per_h_sailing
            + vessel_by_id[option.vessel_id].capacity_t * params.conditioning_eur_per_t
            + (
                vessel_by_id[option.vessel_id].load_dur_h
                + vessel_by_id[option.vessel_id].unload_dur_h
            )
            * params.hoteling_fuel_eur_per_h
        )
        * depart[(option.vessel_id, option.depart_h)]
        for option in options
    )
    return per_departure + stored_expr * params.reconditioning_eur_per_t


def _schedule_cost_breakdown(
    vessels: list[CplexVesselParams],
    departures: dict[str, list[int]],
    sail_hours: dict[str, list[int]],
    stored_t: float,
    params: EconomicParameters,
) -> CplexCostBreakdown:
    vessel_by_id = {v.vessel_id: v for v in vessels}
    total_sail_h = sum(sum(hours) for hours in sail_hours.values())
    loaded_t = sum(vessel_by_id[vessel_id].capacity_t * len(times) for vessel_id, times in departures.items())
    loading_h = sum(vessel_by_id[vessel_id].load_dur_h * len(times) for vessel_id, times in departures.items())
    unloading_h = sum(vessel_by_id[vessel_id].unload_dur_h * len(times) for vessel_id, times in departures.items())
    return CplexCostBreakdown(
        vessel_fuel=total_sail_h * params.vessel_fuel_eur_per_h_sailing,
        conditioning=loaded_t * params.conditioning_eur_per_t,
        reconditioning=stored_t * params.reconditioning_eur_per_t,
        loading=loading_h * params.hoteling_fuel_eur_per_h,
        unloading=unloading_h * params.hoteling_fuel_eur_per_h,
    )


def _validate_solution(
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


def _value(var_or_expr) -> float:
    value = var_or_expr.value()
    return float(value) if value is not None else 0.0
