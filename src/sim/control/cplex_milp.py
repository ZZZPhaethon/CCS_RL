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

import copy
from dataclasses import dataclass, field, replace
import itertools
import math
import os
from pathlib import Path
import re
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
from ..operations.pressure_limits import (
    maximum_feasible_well_rate_tph,
    mtpa_to_tph,
    pressure_limited_rate_level_mask,
    tph_to_mtpa,
)
from ..operations.unloading import terminal_unload_queue_snapshot
from ..routes import route_distance_km, sea_route
from ..scenario_generation import Scenario
from ..scenario_generation.disturbance_resolver import terminal_berth_count
from .objective import control_objective_value, control_objective_weights
from .replay import (
    ReplayExpectation,
    ReplayTolerances,
    action_for_well_control_mode,
    replay_native_actions,
)

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
            stdout, stderr = cplex.communicate(cplex_cmds.encode("UTF-8"))
            self.last_stdout = stdout.decode("UTF-8", errors="replace") if stdout else ""
            self.last_stderr = stderr.decode("UTF-8", errors="replace") if stderr else ""
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
class CplexStageDiagnostic:
    stage: str
    time_limit_s: float | None
    wall_time_s: float
    status: str
    objective_value: float | None
    best_bound: float | None = None
    relative_gap: float | None = None
    nodes: int | None = None
    iterations: int | None = None
    reduced_rows: int | None = None
    reduced_columns: int | None = None
    reduced_nonzeros: int | None = None
    warm_start_requested: bool = False
    warm_start_accepted: bool | None = None
    warm_start_message: str = ""
    termination_reason: str = ""
    raw_log: str = field(default="", repr=False, compare=False)


@dataclass(frozen=True)
class CplexMipStartViolation:
    constraint: str
    sense: str
    residual: float
    violation: float
    variable_names: tuple[str, ...] = ()


@dataclass(frozen=True)
class CplexMipStartAudit:
    total_variables: int
    initialized_variables: int
    missing_variable_count: int
    missing_variable_names: tuple[str, ...]
    bound_violation_count: int
    integrality_violation_count: int
    total_constraints: int
    evaluated_constraints: int
    partial_constraint_count: int
    violated_constraint_count: int
    max_constraint_violation: float
    top_violations: tuple[CplexMipStartViolation, ...] = ()


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
    well_request_tph_by_hour: dict[str, list[float]]
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
    augmented_objective_value: float = 0.0
    economic_objective: bool = False
    terminal_cleanup_value_enabled: bool = False
    terminal_cleanup_cost: float = 0.0
    terminal_cleanup_vessel_fuel: float = 0.0
    terminal_cleanup_conditioning: float = 0.0
    terminal_cleanup_reconditioning: float = 0.0
    terminal_cleanup_loading: float = 0.0
    terminal_cleanup_unloading: float = 0.0
    terminal_cleanup_headroom_risk: float = 0.0
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
    stage_diagnostics: tuple[CplexStageDiagnostic, ...] = ()
    mip_start_audit: CplexMipStartAudit | None = None
    solution_audit: CplexMipStartAudit | None = None
    cargo_t_by_hour: dict[str, tuple[float, ...]] = field(default_factory=dict)
    source_stock_t_by_hour: dict[str, tuple[float, ...]] = field(default_factory=dict)
    terminal_stock_t_by_hour: tuple[float, ...] = ()
    load_t_by_hour: dict[str, tuple[float, ...]] = field(default_factory=dict)
    unload_t_by_hour: dict[str, tuple[float, ...]] = field(default_factory=dict)
    diagnostic_variable_values: dict[str, object] = field(default_factory=dict)


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
    remaining_cleanup_fuel_h: int = 0

    @property
    def duration_h(self) -> int:
        return self.end_h - self.start_h


@dataclass
class _TerminalCleanupModel:
    cost_expr: object
    vessel_fuel_expr: object
    conditioning_expr: object
    reconditioning_expr: object
    loading_expr: object
    unloading_expr: object
    headroom_risk_expr: object
    binary_variables: tuple[object, ...]
    shipped: dict
    trips: dict
    topup: dict
    use: dict
    cargo_positive: dict
    cargo_at_node: dict
    first_from: dict
    needs_return: dict
    end_at: dict
    trip_slots: dict
    first_response_h: dict
    first_response_choice: dict
    headroom_vent: dict


def _add_route_cargo_flow_linking(
    prob,
    env,
    *,
    horizon_h: int,
    arcs,
    starts,
    incoming,
    outgoing,
    arc_vars,
    cargo,
    load,
    unload,
) -> dict[tuple[str, str], object]:
    """Carry cargo on the same time-expanded path as each vessel.

    The base formulation tracks vessel position as a network flow but tracks
    cargo only as one aggregate balance per vessel and hour.  This continuous
    commodity-flow layer is redundant for integer routes and strengthens the
    relaxation by preventing cargo on one fractional branch from being reused
    by another branch.
    """

    H = int(horizon_h)
    arc_cargo = {
        index: pulp.LpVariable(f"route_cargo_{index}", lowBound=0.0)
        for index in range(len(arcs))
    }

    def service_delta(index: int):
        arc = arcs[index]
        if arc.is_sailing:
            return 0.0
        if arc.origin_id in env.emitter_ids:
            return load[(arc.vessel_id, arc.origin_id, arc.start_h)]
        return -unload[(arc.vessel_id, arc.start_h)]

    def end_cargo(index: int):
        return arc_cargo[index] + service_delta(index)

    for index, arc in enumerate(arcs):
        capacity_t = float(env.network.entities[arc.vessel_id].capacity_t)
        prob += arc_cargo[index] <= capacity_t * arc_vars[index]
        if not arc.is_sailing:
            prob += end_cargo(index) >= 0.0
            prob += end_cargo(index) <= capacity_t * arc_vars[index]

    boundary_cargo: dict[tuple[str, str], object] = {}
    for vessel_id in env.vessel_ids:
        start = starts[vessel_id]
        if start.node_id is None or start.start_h >= H:
            continue
        nodes = _nodes_for_vessel(env, vessel_id)
        initial_cargo_t = float(
            env.simulator.state.entity_inventory_t.get(vessel_id, 0.0)
        )
        for t in range(start.start_h, H):
            for node_id in nodes:
                supply = (
                    initial_cargo_t
                    if t == start.start_h and node_id == start.node_id
                    else 0.0
                )
                prob += pulp.lpSum(
                    arc_cargo[index]
                    for index in outgoing.get((vessel_id, t, node_id), [])
                ) == pulp.lpSum(
                    end_cargo(index)
                    for index in incoming.get((vessel_id, t, node_id), [])
                ) + supply

        for t in range(start.start_h, H):
            prob += cargo[(vessel_id, t)] == pulp.lpSum(
                arc_cargo[index]
                for index, arc in enumerate(arcs)
                if arc.vessel_id == vessel_id
                and arc.start_h <= t < arc.end_h
            )

        for node_id in nodes:
            boundary_cargo[(vessel_id, node_id)] = pulp.lpSum(
                end_cargo(index)
                for index in incoming.get((vessel_id, H, node_id), [])
            )
        prob += cargo[(vessel_id, H)] == pulp.lpSum(
            boundary_cargo[(vessel_id, node_id)] for node_id in nodes
        )

    return boundary_cargo


def _add_terminal_cleanup_value_model(
    prob,
    env,
    *,
    horizon_h: int,
    arcs,
    incoming,
    arc_vars,
    cargo,
    source_stock,
    terminal_stock,
    params: EconomicParameters,
    relax_integrality: bool = False,
    boundary_cargo: dict[tuple[str, str], object] | None = None,
    unary_trip_slots: bool = False,
    aggregate_full_trip_dominance: bool = False,
    return_partition_cut: bool = False,
    source_mode_partition_cut: bool = False,
    weather_aware_sailing_lower_bound: bool = False,
    source_headroom_risk: bool = False,
) -> _TerminalCleanupModel:
    """Attach a compact, disturbance-free coordinated cleanup tail.

    The tail clears all source, vessel and terminal inventory present at the
    planning boundary and returns every participating vessel empty to its
    nearest emitter.  It uses aggregate integer trips rather than hourly tail
    indices, so it remains part of the same MILP without adding another 144--216
    hours of native-action variables.
    """

    H = int(horizon_h)
    sources = tuple(env.emitter_ids)
    vessels = tuple(env.vessel_ids)
    if len(env.terminal_ids) != 1:
        raise NotImplementedError(
            "The compact terminal cleanup value currently supports one terminal."
        )
    terminal_id = str(env.terminal_ids[0])
    epsilon_t = 1e-3

    def binary_var(name: str):
        if relax_integrality:
            return pulp.LpVariable(
                name, lowBound=0.0, upBound=1.0, cat="Continuous"
            )
        return pulp.LpVariable(name, cat="Binary")
    end_at = {
        (vessel_id, node_id): pulp.lpSum(
            arc_vars[index]
            for index in incoming.get((vessel_id, H, node_id), [])
        )
        for vessel_id in vessels
        for node_id in (*sources, terminal_id)
    }

    boundary_step = int(env.t) + H

    def leg_fuel_h(vessel_id: str, origin_id: str, destination_id: str) -> int:
        if origin_id == destination_id:
            return 0
        route = env._routes[vessel_id]
        distance_km = (
            float(route["distance_km"])
            if {origin_id, destination_id}
            == {str(route["origin"]), str(route["destination"])}
            else _dynamic_leg_distance_km(env, route, origin_id, destination_id)
        )
        speed_kmh = max(1e-9, float(route["speed_knots"]) * KNOTS_TO_KMH)
        if weather_aware_sailing_lower_bound:
            return _best_future_weather_leg_fuel_h(
                env,
                env.scenario,
                vessel_id,
                origin_id=origin_id,
                destination_id=destination_id,
                distance_km=distance_km,
                earliest_start_step=boundary_step,
            )
        return max(0, math.ceil(distance_km / speed_kmh) - 1)

    to_terminal_leg_h = {
        (vessel_id, source_id): leg_fuel_h(vessel_id, source_id, terminal_id)
        for vessel_id in vessels
        for source_id in sources
    }
    from_terminal_leg_h = {
        (vessel_id, source_id): leg_fuel_h(vessel_id, terminal_id, source_id)
        for vessel_id in vessels
        for source_id in sources
    }
    direct_leg_h = {
        (vessel_id, origin_id, destination_id): leg_fuel_h(
            vessel_id, origin_id, destination_id
        )
        for vessel_id in vessels
        for origin_id in sources
        for destination_id in sources
    }
    ready_leg_h = {
        vessel_id: min(
            from_terminal_leg_h[(vessel_id, source_id)]
            for source_id in sources
        )
        for vessel_id in vessels
    }

    shipped = {
        (vessel_id, source_id): pulp.LpVariable(
            f"tail_shipped_{vessel_id}_{source_id}", lowBound=0.0
        )
        for vessel_id in vessels
        for source_id in sources
    }
    trips = {
        (vessel_id, source_id): pulp.LpVariable(
            f"tail_trips_{vessel_id}_{source_id}",
            lowBound=0,
            upBound=max(
                1,
                math.ceil(
                    float(env.network.entities[source_id].buffer_capacity_t)
                    / float(env.network.entities[vessel_id].capacity_t)
                )
                + 1,
            ),
            cat=(
                "Continuous"
                if relax_integrality or unary_trip_slots
                else "Integer"
            ),
        )
        for vessel_id in vessels
        for source_id in sources
    }
    trip_slots = {
        (vessel_id, source_id, slot): binary_var(
            f"tail_trip_slot_{vessel_id}_{source_id}_{slot}"
        )
        for vessel_id in vessels
        for source_id in sources
        for slot in range(1, int(trips[(vessel_id, source_id)].upBound) + 1)
    } if unary_trip_slots else {}
    if unary_trip_slots:
        for vessel_id in vessels:
            for source_id in sources:
                slots = [
                    trip_slots[(vessel_id, source_id, slot)]
                    for slot in range(
                        1, int(trips[(vessel_id, source_id)].upBound) + 1
                    )
                ]
                prob += trips[(vessel_id, source_id)] == pulp.lpSum(slots)
                for earlier, later in zip(slots, slots[1:]):
                    prob += earlier >= later
        if aggregate_full_trip_dominance:
            for source_id in sources:
                prob += pulp.lpSum(
                    shipped[(vessel_id, source_id)]
                    for vessel_id in vessels
                ) >= pulp.lpSum(
                    float(env.network.entities[vessel_id].capacity_t)
                    * (
                        trips[(vessel_id, source_id)]
                        - trip_slots[(vessel_id, source_id, 1)]
                    )
                    for vessel_id in vessels
                )
    topup = {
        (vessel_id, source_id): pulp.LpVariable(
            f"tail_topup_{vessel_id}_{source_id}", lowBound=0.0
        )
        for vessel_id in vessels
        for source_id in sources
    }
    use = {
        vessel_id: binary_var(f"tail_use_{vessel_id}")
        for vessel_id in vessels
    }
    cargo_positive = {
        vessel_id: binary_var(f"tail_cargo_positive_{vessel_id}")
        for vessel_id in vessels
    }
    cargo_at_node = {
        (vessel_id, node_id): binary_var(
            f"tail_cargo_at_{vessel_id}_{node_id}"
        )
        for vessel_id in vessels
        for node_id in (*sources, terminal_id)
    }
    first_from = {
        (vessel_id, origin_id, destination_id): binary_var(
            f"tail_first_{vessel_id}_{origin_id}_{destination_id}"
        )
        for vessel_id in vessels
        for origin_id in sources
        for destination_id in sources
    }
    needs_return = {
        vessel_id: binary_var(f"tail_needs_return_{vessel_id}")
        for vessel_id in vessels
    }

    for source_id in sources:
        prob += pulp.lpSum(
            shipped[(vessel_id, source_id)] + topup[(vessel_id, source_id)]
            for vessel_id in vessels
        ) == source_stock[(source_id, H)]

    for vessel_id in vessels:
        vessel = env.network.entities[vessel_id]
        capacity_t = float(vessel.capacity_t)
        end_cargo = cargo[(vessel_id, H)]
        positive = cargo_positive[vessel_id]
        prob += end_cargo <= capacity_t * positive
        prob += end_cargo >= epsilon_t * positive
        for node_id in (*sources, terminal_id):
            cargo_here = cargo_at_node[(vessel_id, node_id)]
            prob += cargo_here <= positive
            prob += cargo_here <= end_at[(vessel_id, node_id)]
            prob += cargo_here >= positive + end_at[(vessel_id, node_id)] - 1
            if boundary_cargo is not None:
                cargo_on_path = boundary_cargo[(vessel_id, node_id)]
                prob += cargo_on_path <= capacity_t * cargo_here
                prob += cargo_on_path >= epsilon_t * cargo_here
        # Convex hull of cargo-positive AND the one-hot end location.
        prob += pulp.lpSum(
            cargo_at_node[(vessel_id, node_id)]
            for node_id in (*sources, terminal_id)
        ) == positive
        total_trips = pulp.lpSum(trips[(vessel_id, source_id)] for source_id in sources)
        max_vessel_trips = sum(
            int(trips[(vessel_id, source_id)].upBound) for source_id in sources
        )
        prob += total_trips <= max_vessel_trips * use[vessel_id]
        prob += total_trips >= use[vessel_id]
        if unary_trip_slots:
            first_slots = [
                trip_slots[(vessel_id, source_id, 1)]
                for source_id in sources
            ]
            for first_slot in first_slots:
                prob += use[vessel_id] >= first_slot
            prob += use[vessel_id] <= pulp.lpSum(first_slots)
        for source_id in sources:
            prob += shipped[(vessel_id, source_id)] <= (
                capacity_t * trips[(vessel_id, source_id)]
            )
            prob += topup[(vessel_id, source_id)] <= capacity_t - end_cargo
            prob += topup[(vessel_id, source_id)] <= (
                capacity_t * cargo_at_node[(vessel_id, source_id)]
            )
            if boundary_cargo is not None:
                prob += (
                    topup[(vessel_id, source_id)]
                    + boundary_cargo[(vessel_id, source_id)]
                    <= capacity_t * cargo_at_node[(vessel_id, source_id)]
                )
        # Fractional end locations must not reuse the same free cargo space.
        prob += end_cargo + pulp.lpSum(
            topup[(vessel_id, source_id)] for source_id in sources
        ) <= capacity_t * positive

        first_terms = []
        for origin_id in sources:
            for destination_id in sources:
                first = first_from[(vessel_id, origin_id, destination_id)]
                first_terms.append(first)
                prob += first <= end_at[(vessel_id, origin_id)]
                prob += first <= trips[(vessel_id, destination_id)]
        first_sum = pulp.lpSum(first_terms)
        prob += first_sum <= use[vessel_id]
        prob += first_sum <= 1 - positive
        prob += first_sum <= 1 - end_at[(vessel_id, terminal_id)]
        prob += (
            first_sum
            >= use[vessel_id] - positive - end_at[(vessel_id, terminal_id)]
        )
        if source_mode_partition_cut:
            for source_id in sources:
                prob += (
                    cargo_at_node[(vessel_id, source_id)]
                    + pulp.lpSum(
                        first_from[(vessel_id, source_id, destination_id)]
                        for destination_id in sources
                    )
                    <= end_at[(vessel_id, source_id)]
                )

        returning = needs_return[vessel_id]
        prob += returning >= positive
        prob += returning >= use[vessel_id]
        prob += returning >= end_at[(vessel_id, terminal_id)]
        prob += returning <= positive + use[vessel_id] + end_at[(vessel_id, terminal_id)]
        if return_partition_cut:
            prob += returning >= (
                end_at[(vessel_id, terminal_id)]
                + pulp.lpSum(
                    cargo_at_node[(vessel_id, source_id)]
                    for source_id in sources
                )
                + first_sum
            )

    source_total = pulp.lpSum(source_stock[(source_id, H)] for source_id in sources)
    cargo_total = pulp.lpSum(cargo[(vessel_id, H)] for vessel_id in vessels)
    end_unstored = source_total + cargo_total + terminal_stock[H]
    sailing_fuel_h = (
        pulp.lpSum(
            arc.remaining_cleanup_fuel_h * arc_vars[index]
            for index, arc in enumerate(arcs)
            if arc.remaining_cleanup_fuel_h > 0
        )
        + pulp.lpSum(
            to_terminal_leg_h[(vessel_id, source_id)]
            * cargo_at_node[(vessel_id, source_id)]
            for vessel_id in vessels
            for source_id in sources
        )
        + pulp.lpSum(
            (
                to_terminal_leg_h[(vessel_id, source_id)]
                + from_terminal_leg_h[(vessel_id, source_id)]
            )
            * trips[(vessel_id, source_id)]
            for vessel_id in vessels
            for source_id in sources
        )
        + pulp.lpSum(
            (
                direct_leg_h[(vessel_id, origin_id, destination_id)]
                - from_terminal_leg_h[(vessel_id, destination_id)]
            )
            * first_from[(vessel_id, origin_id, destination_id)]
            for vessel_id in vessels
            for origin_id in sources
            for destination_id in sources
        )
        + pulp.lpSum(
            ready_leg_h[vessel_id] * needs_return[vessel_id]
            for vessel_id in vessels
        )
    )
    vessel_fuel_expr = sailing_fuel_h * params.vessel_fuel_eur_per_h_sailing
    conditioning_expr = source_total * params.conditioning_eur_per_t
    reconditioning_expr = end_unstored * params.reconditioning_eur_per_t
    loading_expr = pulp.lpSum(
        source_stock[(source_id, H)]
        / max(1e-9, float(env.network.entities[source_id].loading_rate_tph))
        for source_id in sources
    ) * params.hoteling_fuel_eur_per_h
    unloading_expr = pulp.lpSum(
        (
            cargo[(vessel_id, H)]
            + pulp.lpSum(
                shipped[(vessel_id, source_id)] + topup[(vessel_id, source_id)]
                for source_id in sources
            )
        )
        / max(1e-9, float(env.network.entities[vessel_id].unloading_rate_tph))
        for vessel_id in vessels
    ) * params.hoteling_fuel_eur_per_h
    first_response_h: dict[str, object] = {}
    first_response_choice: dict[tuple[str, str, str, str], object] = {}
    headroom_vent: dict[str, object] = {}
    headroom_risk_expr = pulp.lpSum([])
    if source_headroom_risk:
        nodes = (*sources, terminal_id)

        def nominal_leg_duration_h(
            vessel_id: str,
            origin_id: str,
            destination_id: str,
        ) -> int:
            if origin_id == destination_id:
                return 0
            route = env._routes[vessel_id]
            distance_km = (
                float(route["distance_km"])
                if {origin_id, destination_id}
                == {str(route["origin"]), str(route["destination"])}
                else _dynamic_leg_distance_km(
                    env, route, origin_id, destination_id
                )
            )
            speed_kmh = max(
                1e-9,
                float(route["speed_knots"]) * KNOTS_TO_KMH,
            )
            return max(1, math.ceil(distance_km / speed_kmh - 1e-9))

        remaining_h = {
            vessel_id: pulp.lpSum(
                arcs[index].remaining_cleanup_fuel_h * arc_vars[index]
                for index, arc in enumerate(arcs)
                if arc.vessel_id == vessel_id
                and arc.end_h == H
                and arc.remaining_cleanup_fuel_h > 0
            )
            for vessel_id in vessels
        }
        if len(sources) <= len(vessels):
            assigned_vessels = min(
                itertools.permutations(vessels, len(sources)),
                key=lambda assignment: sum(
                    nominal_leg_duration_h(
                        vessel_id, terminal_id, source_id
                    )
                    for source_id, vessel_id in zip(sources, assignment)
                ),
            )
            response_vessel_by_source = dict(
                zip(sources, assigned_vessels)
            )
        else:
            response_vessel_by_source = {
                source_id: min(
                    vessels,
                    key=lambda vessel_id: nominal_leg_duration_h(
                        vessel_id, terminal_id, source_id
                    ),
                )
                for source_id in sources
            }
        for source_id in sources:
            emitter = env.network.entities[source_id]
            profile = emitter.hourly_capture_profile_tph
            normal_capture_tph = (
                sum(float(value) for value in profile) / len(profile)
                if profile
                else float(emitter.nominal_capture_tph)
            )
            normal_capture_tph *= max(
                0.0,
                float(emitter.availability)
                * float(emitter.default_utilization),
            )
            # Use a matched response vessel and calculate the physical time
            # until it can start loading at this source: finish an unfinished
            # boundary leg; if loaded, reach the terminal and unload its actual
            # cargo; then travel to the source.  This keeps the timing linear
            # in the existing boundary route/cargo state.
            response_vessel_id = response_vessel_by_source[source_id]
            vessel = env.network.entities[response_vessel_id]
            response_h = (
                remaining_h[response_vessel_id]
                + pulp.lpSum(
                    nominal_leg_duration_h(
                        response_vessel_id, node_id, source_id
                    )
                    * (
                        end_at[(response_vessel_id, node_id)]
                        - cargo_at_node[(response_vessel_id, node_id)]
                    )
                    + nominal_leg_duration_h(
                        response_vessel_id, node_id, terminal_id
                    )
                    * cargo_at_node[(response_vessel_id, node_id)]
                    for node_id in nodes
                )
                + cargo[(response_vessel_id, H)]
                / max(1e-9, float(vessel.unloading_rate_tph))
                + nominal_leg_duration_h(
                    response_vessel_id, terminal_id, source_id
                )
                * cargo_positive[response_vessel_id]
            )
            first_response_h[source_id] = response_h
            risk = pulp.LpVariable(
                f"tail_headroom_vent_{source_id}", lowBound=0.0
            )
            headroom_vent[source_id] = risk
            prob += risk >= (
                source_stock[(source_id, H)]
                + normal_capture_tph * response_h
                - float(emitter.buffer_capacity_t)
            )

        headroom_risk_expr = (
            pulp.lpSum(headroom_vent.values())
            * params.carbon_price_eur_per_t
        )
    cost_expr = (
        vessel_fuel_expr
        + conditioning_expr
        + reconditioning_expr
        + loading_expr
        + unloading_expr
        + headroom_risk_expr
    )
    return _TerminalCleanupModel(
        cost_expr=cost_expr,
        vessel_fuel_expr=vessel_fuel_expr,
        conditioning_expr=conditioning_expr,
        reconditioning_expr=reconditioning_expr,
        loading_expr=loading_expr,
        unloading_expr=unloading_expr,
        headroom_risk_expr=headroom_risk_expr,
        binary_variables=tuple(
            [*use.values(), *cargo_positive.values(), *cargo_at_node.values(),
             *first_from.values(), *needs_return.values(), *trip_slots.values(),
             *first_response_choice.values()]
        ),
        shipped=shipped,
        trips=trips,
        topup=topup,
        use=use,
        cargo_positive=cargo_positive,
        cargo_at_node=cargo_at_node,
        first_from=first_from,
        needs_return=needs_return,
        end_at=end_at,
        trip_slots=trip_slots,
        first_response_h=first_response_h,
        first_response_choice=first_response_choice,
        headroom_vent=headroom_vent,
    )


def _terminal_cleanup_solution_for_state(
    env,
    params: EconomicParameters,
    *,
    aggregate_full_trip_dominance: bool = False,
    return_partition_cut: bool = False,
    source_mode_partition_cut: bool = False,
    weather_aware_sailing_lower_bound: bool = False,
    source_headroom_risk: bool = False,
) -> tuple[float, dict[str, float]]:
    """Solve compact cleanup and return its cost and variable assignment."""

    if pulp is None:  # pragma: no cover - guarded by the normal PuLP dependency
        raise RuntimeError("PuLP is required to evaluate terminal cleanup cost")
    H = 0
    terminal_id = str(env.terminal_ids[0])
    nodes = (*env.emitter_ids, terminal_id)
    arcs: list[_ActionArc] = []
    incoming: dict[tuple[str, int, str], list[int]] = {}
    boundary_cargo: dict[tuple[str, str], float] = {}
    state = env.simulator.state

    for vessel_id in env.vessel_ids:
        vessel_state = env.simulator.vessel_states[vessel_id]
        mode = str(vessel_state["mode"])
        if mode == "sailing":
            destination_id = str(vessel_state["destination"])
            origin_id = str(vessel_state.get("origin", destination_id))
            route = env._routes[vessel_id]
            distance_km = float(
                vessel_state.get("distance_km") or route["distance_km"]
            )
            remaining_km = max(
                0.0,
                distance_km * (1.0 - float(vessel_state.get("progress", 0.0))),
            )
            speed_kmh = max(
                1e-9,
                float(route["speed_knots"]) * KNOTS_TO_KMH,
            )
            remaining_fuel_h = (
                _weather_remaining_fuel_hours_after_boundary(
                    env,
                    env.scenario,
                    vessel_id,
                    origin_id=origin_id,
                    destination_id=destination_id,
                    distance_km=remaining_km,
                    boundary_step=int(env.t),
                )
                if weather_aware_sailing_lower_bound
                else max(
                    0,
                    math.ceil(remaining_km / speed_kmh - 1e-9),
                )
            )
        else:
            destination_id = str(
                vessel_state.get("berth")
                or state.vessel_berths.get(vessel_id)
                or env._routes[vessel_id]["origin"]
            )
            origin_id = destination_id
            remaining_fuel_h = 0
        index = len(arcs)
        arcs.append(
            _ActionArc(
                vessel_id=vessel_id,
                start_h=H,
                end_h=H,
                origin_id=origin_id,
                destination_id=destination_id,
                action=VESSEL_WAIT,
                is_sailing=(mode == "sailing"),
                remaining_cleanup_fuel_h=remaining_fuel_h,
            )
        )
        incoming[(vessel_id, H, destination_id)] = [index]
        cargo_t = float(state.entity_inventory_t.get(vessel_id, 0.0))
        for node_id in nodes:
            boundary_cargo[(vessel_id, node_id)] = (
                cargo_t if node_id == destination_id else 0.0
            )

    problem = pulp.LpProblem("fixed_state_terminal_cleanup", pulp.LpMinimize)
    cleanup = _add_terminal_cleanup_value_model(
        problem,
        env,
        horizon_h=H,
        arcs=arcs,
        incoming=incoming,
        arc_vars=[1.0] * len(arcs),
        cargo={
            (vessel_id, H): float(state.entity_inventory_t.get(vessel_id, 0.0))
            for vessel_id in env.vessel_ids
        },
        source_stock={
            (source_id, H): float(state.entity_inventory_t.get(source_id, 0.0))
            for source_id in env.emitter_ids
        },
        terminal_stock={
            H: float(state.entity_inventory_t.get(terminal_id, 0.0))
        },
        params=params,
        boundary_cargo=boundary_cargo,
        unary_trip_slots=True,
        aggregate_full_trip_dominance=aggregate_full_trip_dominance,
        return_partition_cut=return_partition_cut,
        source_mode_partition_cut=source_mode_partition_cut,
        weather_aware_sailing_lower_bound=(
            weather_aware_sailing_lower_bound
        ),
        source_headroom_risk=source_headroom_risk,
    )
    problem += cleanup.cost_expr
    problem.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=10.0))
    status = pulp.LpStatus.get(problem.status, str(problem.status))
    if status != "Optimal":
        raise RuntimeError(f"Fixed-state terminal cleanup returned {status}")
    return (
        float(pulp.value(cleanup.cost_expr)),
        {
            variable.name: float(variable.varValue)
            for variable in problem.variables()
            if variable.varValue is not None
        },
    )


def _terminal_cleanup_cost_for_state(
    env,
    params: EconomicParameters,
    *,
    source_mode_partition_cut: bool = False,
    weather_aware_sailing_lower_bound: bool = False,
    source_headroom_risk: bool = False,
) -> float:
    """Solve the same compact cleanup model for one fixed environment state."""

    cost, _variable_values = _terminal_cleanup_solution_for_state(
        env,
        params,
        source_mode_partition_cut=source_mode_partition_cut,
        weather_aware_sailing_lower_bound=(
            weather_aware_sailing_lower_bound
        ),
        source_headroom_risk=source_headroom_risk,
    )
    return cost


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
    cplex_options: list[str] | None = None,
    msg: bool = False,
    lexicographic_vent_first: bool = False,
    economic_objective: bool = False,
    max_nonstored_t: float | None = None,
    max_vented_t: float | None = None,
    max_end_unstored_t: float | None = None,
    execution_boundary_h: int | None = None,
    max_execution_vented_t: float | None = None,
    max_execution_unstored_t: float | None = None,
    environment_aligned_service: bool = False,
    safe_execution_h: int | None = None,
    terminal_cleanup_value: bool = False,
    terminal_cleanup_mip_start_mode: str = "partial",
    load_min_formulation: str = "choice3",
    fifo_diagnostic_mode: str = "full",
    vessel_visit_load_cuts: bool = False,
    vessel_visit_load_cut_stride_h: int = 24,
    source_visit_vent_cuts: bool = False,
    source_visit_vent_cut_stride_h: int = 24,
    terminal_visit_cuts: bool = False,
    terminal_visit_cut_stride_h: int = 24,
    service_reachability_cuts: bool = False,
    service_reachability_cut_stride_h: int = 12,
    route_cargo_flow_linking: bool = False,
    cleanup_unary_trip_slots: bool = False,
    cleanup_aggregate_full_trip_dominance: bool = False,
    cleanup_return_partition_cut: bool = False,
    cleanup_source_mode_partition_cut: bool = False,
    weather_aware_cleanup_sailing_lower_bound: bool = False,
    cleanup_source_headroom_risk: bool = False,
    prune_unreachable_route_arcs: bool = False,
    min_total_cleanup_trips: int | None = None,
    fixed_cleanup_trips_by_source: dict[str, int] | None = None,
    fixed_cleanup_trips_by_vessel_source: (
        dict[tuple[str, str], int] | None
    ) = None,
    fixed_boundary_node_by_vessel: dict[str, str] | None = None,
    fix_warm_start_vessel_routes: bool = False,
    fixed_terminal_departures_by_vessel: dict[str, int] | None = None,
    fixed_terminal_departures_by_vessel_source: (
        dict[tuple[str, str], int] | None
    ) = None,
    fixed_terminal_to_source_departures_by_vessel_source: (
        dict[tuple[str, str], int] | None
    ) = None,
    fixed_source_reposition_departures_by_vessel: dict[str, int] | None = None,
    min_total_source_reposition_departures: int | None = None,
    integrality_relax_groups: tuple[str, ...] = (),
    constraint_redundancy_audit: bool = False,
    export_model_lp_path: str | Path | None = None,
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
    terminal_cleanup_mip_start_mode = str(
        terminal_cleanup_mip_start_mode
    ).lower()
    if terminal_cleanup_mip_start_mode not in {"partial", "complete"}:
        raise ValueError(
            "terminal_cleanup_mip_start_mode must be 'partial' or 'complete'"
        )
    fixed_cleanup_trips_by_source = dict(
        fixed_cleanup_trips_by_source or {}
    )
    fixed_cleanup_trips_by_vessel_source = dict(
        fixed_cleanup_trips_by_vessel_source or {}
    )
    fixed_boundary_node_by_vessel = dict(
        fixed_boundary_node_by_vessel or {}
    )
    unknown_cleanup_sources = (
        set(fixed_cleanup_trips_by_source) - set(env.emitter_ids)
    )
    if unknown_cleanup_sources:
        raise ValueError(
            "Unknown cleanup sources: "
            f"{sorted(unknown_cleanup_sources)}"
        )
    if any(count < 0 for count in fixed_cleanup_trips_by_source.values()):
        raise ValueError("fixed cleanup trip counts must be non-negative")
    valid_vessel_source_pairs = {
        (vessel_id, source_id)
        for vessel_id in env.vessel_ids
        for source_id in env.emitter_ids
    }
    unknown_cleanup_pairs = (
        set(fixed_cleanup_trips_by_vessel_source)
        - valid_vessel_source_pairs
    )
    if unknown_cleanup_pairs:
        raise ValueError(
            "Unknown cleanup vessel/source pairs: "
            f"{sorted(unknown_cleanup_pairs)}"
        )
    if any(
        count < 0
        for count in fixed_cleanup_trips_by_vessel_source.values()
    ):
        raise ValueError("fixed cleanup vessel/source trips must be non-negative")
    unknown_boundary_vessels = (
        set(fixed_boundary_node_by_vessel) - set(env.vessel_ids)
    )
    if unknown_boundary_vessels:
        raise ValueError(
            "Unknown boundary vessels: "
            f"{sorted(unknown_boundary_vessels)}"
        )
    for vessel_id, node_id in fixed_boundary_node_by_vessel.items():
        if node_id not in _nodes_for_vessel(env, vessel_id):
            raise ValueError(
                f"Unknown boundary node {node_id!r} for vessel {vessel_id!r}"
            )
    if min_total_cleanup_trips is not None:
        min_total_cleanup_trips = int(min_total_cleanup_trips)
        if min_total_cleanup_trips < 0:
            raise ValueError("minimum total cleanup trips must be non-negative")
    if abs(float(env.network.time_step_hours) - 1.0) > 1e-9:
        raise ValueError("The CPLEX MILP currently expects 1-hour network time steps.")
    load_min_formulation = str(load_min_formulation).lower()
    if load_min_formulation not in {"choice3", "factored"}:
        raise ValueError(
            "load_min_formulation must be either 'choice3' or 'factored'"
        )
    fifo_diagnostic_mode = str(fifo_diagnostic_mode).lower()
    if fifo_diagnostic_mode not in {"full", "relaxed_pairwise", "relaxed_fifo"}:
        raise ValueError(
            "fifo_diagnostic_mode must be 'full', 'relaxed_pairwise', or "
            "'relaxed_fifo'"
        )
    vessel_visit_load_cut_stride_h = int(vessel_visit_load_cut_stride_h)
    if vessel_visit_load_cuts and vessel_visit_load_cut_stride_h <= 0:
        raise ValueError("vessel_visit_load_cut_stride_h must be positive")
    source_visit_vent_cut_stride_h = int(source_visit_vent_cut_stride_h)
    if source_visit_vent_cuts and source_visit_vent_cut_stride_h <= 0:
        raise ValueError("source_visit_vent_cut_stride_h must be positive")
    terminal_visit_cut_stride_h = int(terminal_visit_cut_stride_h)
    if terminal_visit_cuts and terminal_visit_cut_stride_h <= 0:
        raise ValueError("terminal_visit_cut_stride_h must be positive")
    service_reachability_cut_stride_h = int(
        service_reachability_cut_stride_h
    )
    if service_reachability_cuts and service_reachability_cut_stride_h <= 0:
        raise ValueError("service_reachability_cut_stride_h must be positive")
    integrality_relax_groups = tuple(
        str(group).strip().lower() for group in integrality_relax_groups
    )
    allowed_integrality_groups = {
        "all",
        "route",
        "service",
        "fifo",
        "overflow",
        "injection",
        "cleanup",
    }
    unknown_integrality_groups = (
        set(integrality_relax_groups) - allowed_integrality_groups
    )
    if unknown_integrality_groups:
        raise ValueError(
            "Unknown integrality relaxation groups: "
            f"{sorted(unknown_integrality_groups)}"
        )

    def group_is_relaxed(group: str) -> bool:
        return "all" in integrality_relax_groups or group in integrality_relax_groups

    def binary_var(name: str, group: str):
        if group_is_relaxed(group):
            return pulp.LpVariable(
                name, lowBound=0.0, upBound=1.0, cat="Continuous"
            )
        return pulp.LpVariable(name, cat="Binary")

    params = economics or EconomicParameters()
    if lexicographic_vent_first and economic_objective:
        raise ValueError("lexicographic_vent_first and economic_objective are mutually exclusive")
    objective_weights = control_objective_weights(
        env,
        params,
        storage_reward_eur_per_t=storage_reward_eur_per_t,
    )
    reward_per_t = 0.0 if economic_objective else objective_weights.storage_reward_eur_per_t
    start_h = _current_start_hour(env)
    start_step = scenario.step_index(start_h)
    if horizon_h is None:
        horizon_h = max(0, scenario.n_steps - start_step)
    H = int(horizon_h)
    if H <= 0:
        return _empty_result()
    state = env.simulator.state
    fixed_terminal_departures_by_vessel = (
        fixed_terminal_departures_by_vessel or {}
    )
    unknown_fixed_vessels = set(fixed_terminal_departures_by_vessel) - set(
        env.vessel_ids
    )
    if unknown_fixed_vessels:
        raise ValueError(
            "Unknown vessels in fixed_terminal_departures_by_vessel: "
            f"{sorted(unknown_fixed_vessels)}"
        )
    fixed_terminal_departures_by_vessel = {
        vessel_id: int(count)
        for vessel_id, count in fixed_terminal_departures_by_vessel.items()
    }
    if any(count < 0 for count in fixed_terminal_departures_by_vessel.values()):
        raise ValueError("fixed terminal departure counts must be non-negative")
    fixed_terminal_departures_by_vessel_source = (
        fixed_terminal_departures_by_vessel_source or {}
    )
    known_vessel_sources = {
        (vessel_id, emitter_id)
        for vessel_id in env.vessel_ids
        for emitter_id in env.emitter_ids
    }
    unknown_fixed_vessel_sources = (
        set(fixed_terminal_departures_by_vessel_source) - known_vessel_sources
    )
    if unknown_fixed_vessel_sources:
        raise ValueError(
            "Unknown vessel/source pairs in "
            "fixed_terminal_departures_by_vessel_source: "
            f"{sorted(unknown_fixed_vessel_sources)}"
        )
    fixed_terminal_departures_by_vessel_source = {
        key: int(count)
        for key, count in fixed_terminal_departures_by_vessel_source.items()
    }
    if any(
        count < 0
        for count in fixed_terminal_departures_by_vessel_source.values()
    ):
        raise ValueError("fixed terminal departure counts must be non-negative")
    fixed_terminal_to_source_departures_by_vessel_source = (
        fixed_terminal_to_source_departures_by_vessel_source or {}
    )
    unknown_fixed_return_vessel_sources = (
        set(fixed_terminal_to_source_departures_by_vessel_source)
        - known_vessel_sources
    )
    if unknown_fixed_return_vessel_sources:
        raise ValueError(
            "Unknown vessel/source pairs in "
            "fixed_terminal_to_source_departures_by_vessel_source: "
            f"{sorted(unknown_fixed_return_vessel_sources)}"
        )
    fixed_terminal_to_source_departures_by_vessel_source = {
        key: int(count)
        for key, count in (
            fixed_terminal_to_source_departures_by_vessel_source.items()
        )
    }
    if any(
        count < 0
        for count in (
            fixed_terminal_to_source_departures_by_vessel_source.values()
        )
    ):
        raise ValueError("fixed terminal departure counts must be non-negative")
    fixed_source_reposition_departures_by_vessel = (
        fixed_source_reposition_departures_by_vessel or {}
    )
    unknown_reposition_vessels = set(
        fixed_source_reposition_departures_by_vessel
    ) - set(env.vessel_ids)
    if unknown_reposition_vessels:
        raise ValueError(
            "Unknown vessels in fixed_source_reposition_departures_by_vessel: "
            f"{sorted(unknown_reposition_vessels)}"
        )
    fixed_source_reposition_departures_by_vessel = {
        vessel_id: int(count)
        for vessel_id, count in (
            fixed_source_reposition_departures_by_vessel.items()
        )
    }
    if any(
        count < 0
        for count in fixed_source_reposition_departures_by_vessel.values()
    ):
        raise ValueError("fixed source reposition counts must be non-negative")
    if min_total_source_reposition_departures is not None:
        min_total_source_reposition_departures = int(
            min_total_source_reposition_departures
        )
        if min_total_source_reposition_departures < 0:
            raise ValueError(
                "minimum total source reposition count must be non-negative"
            )
    hours = range(H)
    terminal_capacity_t = _terminal_capacity_t(env)
    arcs, starts = _build_action_arcs(
        env,
        scenario,
        start_step,
        H,
        prune_unreachable=prune_unreachable_route_arcs,
        weather_aware_cleanup_sailing_lower_bound=(
            weather_aware_cleanup_sailing_lower_bound
        ),
    )
    well_rate_options = _well_rate_options_by_hour(env, scenario, H)
    load_rate_groups = {
        emitter_id: tuple(
            sorted(
                {
                    float(
                        min(
                            env.network.entities[emitter_id].loading_rate_tph,
                            env.network.entities[vessel_id].loading_rate_tph,
                        )
                    )
                    for vessel_id in env.vessel_ids
                }
            )
        )
        for emitter_id in env.emitter_ids
    }
    load_rate_group_index = {
        (vessel_id, emitter_id): load_rate_groups[emitter_id].index(
            float(
                min(
                    env.network.entities[emitter_id].loading_rate_tph,
                    env.network.entities[vessel_id].loading_rate_tph,
                )
            )
        )
        for vessel_id in env.vessel_ids
        for emitter_id in env.emitter_ids
    }

    prob = pulp.LpProblem("full_scenario_native_action_cplex_milp", pulp.LpMinimize)
    arc_vars = {
        index: binary_var(f"x_arc_{index}", "route")
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
        (vessel_id, t): binary_var(
            f"cargo_positive_{vessel_id}_{t}", "service"
        )
        for vessel_id in env.vessel_ids
        for t in hours
    } if environment_aligned_service else {}
    cargo_space = {
        (vessel_id, t): binary_var(
            f"cargo_space_{vessel_id}_{t}", "service"
        )
        for vessel_id in env.vessel_ids
        for t in hours
    } if environment_aligned_service else {}
    terminal_eligible = {
        (vessel_id, t): binary_var(
            f"terminal_eligible_{vessel_id}_{t}", "fifo"
        )
        for vessel_id in env.vessel_ids
        for t in hours
    } if environment_aligned_service else {}
    terminal_queue_age = {
        (vessel_id, t): pulp.LpVariable(
            f"terminal_queue_age_{vessel_id}_{t}",
            lowBound=0.0,
            upBound=H + len(env.vessel_ids),
        )
        for vessel_id in env.vessel_ids
        for t in hours
    } if environment_aligned_service and fifo_diagnostic_mode != "relaxed_fifo" else {}
    load = {
        (vessel_id, emitter_id, t): pulp.LpVariable(f"load_{vessel_id}_{emitter_id}_{t}", lowBound=0.0)
        for vessel_id in env.vessel_ids
        for emitter_id in env.emitter_ids
        for t in hours
    }
    load_active = {
        (vessel_id, emitter_id, t): binary_var(
            f"load_active_{vessel_id}_{emitter_id}_{t}", "service"
        )
        for vessel_id in env.vessel_ids
        for emitter_id in env.emitter_ids
        for t in hours
    }
    load_limit_choice = {
        (vessel_id, emitter_id, t, limit): binary_var(
            f"load_limit_{vessel_id}_{emitter_id}_{t}_{limit}", "service"
        )
        for vessel_id in env.vessel_ids
        for emitter_id in env.emitter_ids
        for t in hours
        for limit in range(3)
    } if environment_aligned_service and load_min_formulation == "choice3" else {}
    unload = {
        (vessel_id, t): pulp.LpVariable(f"unload_{vessel_id}_{t}", lowBound=0.0)
        for vessel_id in env.vessel_ids
        for t in hours
    }
    unload_active = {
        (vessel_id, t): binary_var(
            f"unload_active_{vessel_id}_{t}", "service"
        )
        for vessel_id in env.vessel_ids
        for t in hours
    }
    unload_limit_choice = {
        (vessel_id, t, limit): binary_var(
            f"unload_limit_{vessel_id}_{t}_{limit}", "service"
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
    load_source_cap = {
        (emitter_id, t, rate_index): pulp.LpVariable(
            f"load_source_cap_{emitter_id}_{t}_{rate_index}", lowBound=0.0
        )
        for emitter_id in env.emitter_ids
        for t in hours
        for rate_index in range(len(load_rate_groups[emitter_id]))
    } if environment_aligned_service and load_min_formulation == "factored" else {}
    load_source_limit_choice = {
        (emitter_id, t, rate_index): binary_var(
            f"load_source_limit_{emitter_id}_{t}_{rate_index}", "service"
        )
        for emitter_id in env.emitter_ids
        for t in hours
        for rate_index in range(len(load_rate_groups[emitter_id]))
    } if environment_aligned_service and load_min_formulation == "factored" else {}
    load_capacity_limit_choice = {
        (vessel_id, emitter_id, t): binary_var(
            f"load_capacity_limit_{vessel_id}_{emitter_id}_{t}", "service"
        )
        for vessel_id in env.vessel_ids
        for emitter_id in env.emitter_ids
        for t in hours
    } if environment_aligned_service and load_min_formulation == "factored" else {}
    source_overflow_active = {
        (emitter_id, t): binary_var(
            f"source_overflow_{emitter_id}_{t}", "overflow"
        )
        for emitter_id in env.emitter_ids
        for t in hours
    }
    terminal_stock = {
        t: pulp.LpVariable(f"terminal_stock_{t}", lowBound=0.0, upBound=terminal_capacity_t)
        for t in range(H + 1)
    }
    well_choice = {
        (well_id, t, rate_index): binary_var(
            f"well_{well_id}_{t}_{rate_index}", "injection"
        )
        for well_id in env.well_ids
        for t in hours
        for rate_index in well_rate_options[(well_id, t)]
    }
    automatic_well_physical_max = (
        _physical_well_max_by_hour(env, scenario, H)
        if env.automatic_well_control
        else {}
    )
    automatic_well_request = {
        (well_id, t): pulp.LpVariable(
            f"well_request_{well_id}_{t}",
            lowBound=0.0,
            upBound=automatic_well_physical_max[(well_id, t)],
        )
        for well_id in env.well_ids
        for t in hours
    } if env.automatic_well_control else {}
    automatic_well_regime = {
        (well_id, t, regime): binary_var(
            f"well_regime_{well_id}_{t}_{regime}",
            "injection",
        )
        for well_id in env.well_ids
        if _uses_single_well_dynamic_bhp(env, well_id)
        for t in hours
        for regime in ("off", "physical", "pressure")
    } if env.automatic_well_control else {}
    well_inj = {
        (well_id, t): pulp.LpVariable(f"well_actual_{well_id}_{t}", lowBound=0.0)
        for well_id in env.well_ids
        for t in hours
    }
    injection_limit_choice = {
        (t, limit): binary_var(f"injection_limit_{t}_{limit}", "injection")
        for t in hours
        for limit in range(2)
    }
    vent = {
        (emitter_id, t): pulp.LpVariable(f"vent_{emitter_id}_{t}", lowBound=0.0)
        for emitter_id in env.emitter_ids
        for t in hours
    }

    incoming, outgoing, wait_arc = _index_arcs(arcs)
    initial_terminal_queue_ages = _initial_terminal_queue_ages(env)
    if environment_aligned_service and load_min_formulation == "factored":
        for emitter_id in env.emitter_ids:
            emitter = env.network.entities[emitter_id]
            source_m = float(emitter.buffer_capacity_t)
            for t in hours:
                for rate_index, load_cap_tph in enumerate(
                    load_rate_groups[emitter_id]
                ):
                    service_cap = load_source_cap[(emitter_id, t, rate_index)]
                    source_selected = load_source_limit_choice[
                        (emitter_id, t, rate_index)
                    ]
                    prob += service_cap <= float(load_cap_tph)
                    prob += service_cap <= source_ready[(emitter_id, t)]
                    prob += service_cap >= (
                        source_ready[(emitter_id, t)]
                        - source_m * (1 - source_selected)
                    )
                    prob += service_cap >= float(load_cap_tph) * (
                        1 - source_selected
                    )
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

    for vessel_id, departure_count in fixed_terminal_departures_by_vessel.items():
        terminal_id = str(env._routes[vessel_id]["destination"])
        prob += pulp.lpSum(
            arc_vars[index]
            for index, arc in enumerate(arcs)
            if arc.vessel_id == vessel_id
            and arc.is_sailing
            and arc.origin_id in env.emitter_ids
            and arc.destination_id == terminal_id
        ) == departure_count

    for (
        vessel_id,
        emitter_id,
    ), departure_count in fixed_terminal_departures_by_vessel_source.items():
        terminal_id = str(env._routes[vessel_id]["destination"])
        prob += pulp.lpSum(
            arc_vars[index]
            for index, arc in enumerate(arcs)
            if arc.vessel_id == vessel_id
            and arc.is_sailing
            and arc.origin_id == emitter_id
            and arc.destination_id == terminal_id
        ) == departure_count

    for (
        vessel_id,
        emitter_id,
    ), departure_count in (
        fixed_terminal_to_source_departures_by_vessel_source.items()
    ):
        terminal_id = str(env._routes[vessel_id]["destination"])
        prob += pulp.lpSum(
            arc_vars[index]
            for index, arc in enumerate(arcs)
            if arc.vessel_id == vessel_id
            and arc.is_sailing
            and arc.origin_id == terminal_id
            and arc.destination_id == emitter_id
        ) == departure_count

    emitter_ids = set(env.emitter_ids)
    for vessel_id, departure_count in (
        fixed_source_reposition_departures_by_vessel.items()
    ):
        prob += pulp.lpSum(
            arc_vars[index]
            for index, arc in enumerate(arcs)
            if arc.vessel_id == vessel_id
            and arc.is_sailing
            and arc.origin_id in emitter_ids
            and arc.destination_id in emitter_ids
            and arc.origin_id != arc.destination_id
        ) == departure_count
    if min_total_source_reposition_departures is not None:
        prob += pulp.lpSum(
            arc_vars[index]
            for index, arc in enumerate(arcs)
            if arc.is_sailing
            and arc.origin_id in emitter_ids
            and arc.destination_id in emitter_ids
            and arc.origin_id != arc.destination_id
        ) >= min_total_source_reposition_departures

    if vessel_visit_load_cuts:
        emitter_ids = set(env.emitter_ids)
        for vessel_id in env.vessel_ids:
            vessel_capacity_t = float(env.network.entities[vessel_id].capacity_t)
            start = starts[vessel_id]
            source_arrivals = [
                (index, arc)
                for index, arc in enumerate(arcs)
                if arc.vessel_id == vessel_id
                and arc.is_sailing
                and arc.arrives_within_horizon
                and arc.destination_id in emitter_ids
            ]
            cumulative_load_terms = []
            for t in hours:
                cumulative_load_terms.extend(
                    load[(vessel_id, emitter_id, t)]
                    for emitter_id in env.emitter_ids
                )
                if (
                    (t + 1) % vessel_visit_load_cut_stride_h != 0
                    and t != H - 1
                ):
                    continue
                initial_source_visit = int(
                    start.node_id in emitter_ids and start.start_h <= t
                )
                completed_source_arrivals = pulp.lpSum(
                    arc_vars[index]
                    for index, arc in source_arrivals
                    if arc.end_h <= t
                )
                prob += pulp.lpSum(cumulative_load_terms) <= vessel_capacity_t * (
                    initial_source_visit + completed_source_arrivals
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

    for vessel_id in env.vessel_ids:
        vessel = env.network.entities[vessel_id]
        initial_cargo_t = float(state.entity_inventory_t.get(vessel_id, 0.0))
        prob += cargo[(vessel_id, 0)] == initial_cargo_t
        terminal_id = str(env._routes[vessel_id]["destination"])
        for t in hours:
            if environment_aligned_service:
                capacity_t = float(vessel.capacity_t)
                epsilon_t = 1e-3
                prob += cargo[(vessel_id, t)] <= capacity_t * cargo_positive[(vessel_id, t)]
                prob += cargo[(vessel_id, t)] >= epsilon_t * cargo_positive[(vessel_id, t)]
                prob += cargo[(vessel_id, t)] >= capacity_t * (1 - cargo_space[(vessel_id, t)])
                prob += (
                    cargo[(vessel_id, t)]
                    <= capacity_t - epsilon_t + capacity_t * (1 - cargo_space[(vessel_id, t)])
                )
                wait_terminal = _wait_expr(arc_vars, wait_arc, vessel_id, terminal_id, t)
                eligible = terminal_eligible[(vessel_id, t)]
                prob += eligible <= wait_terminal
                prob += eligible <= cargo_positive[(vessel_id, t)]
                prob += eligible >= wait_terminal + cargo_positive[(vessel_id, t)] - 1
                if fifo_diagnostic_mode != "relaxed_fifo":
                    queue_age = terminal_queue_age[(vessel_id, t)]
                    max_queue_age = H + len(env.vessel_ids)
                    prob += queue_age <= max_queue_age * eligible
                    if t == 0:
                        prob += queue_age == initial_terminal_queue_ages.get(vessel_id, 0) * eligible
                    else:
                        previous_age = terminal_queue_age[(vessel_id, t - 1)]
                        prob += queue_age >= previous_age + 1 - max_queue_age * (1 - eligible)
                        prob += queue_age <= previous_age + 1 + max_queue_age * (1 - eligible)
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
                    if load_min_formulation == "choice3":
                        choices = [
                            load_limit_choice[(vessel_id, emitter_id, t, i)]
                            for i in range(3)
                        ]
                        prob += pulp.lpSum(choices) == active
                        limits = (
                            float(load_cap_tph),
                            source_ready[(emitter_id, t)],
                            float(vessel.capacity_t) - cargo[(vessel_id, t)],
                        )
                        big_m = max(
                            float(vessel.capacity_t),
                            float(emitter.buffer_capacity_t),
                            float(load_cap_tph),
                        )
                        for choice, limit in zip(choices, limits):
                            prob += load[(vessel_id, emitter_id, t)] >= (
                                limit - big_m * (1 - choice)
                            )
                    else:
                        rate_index = load_rate_group_index[(vessel_id, emitter_id)]
                        service_cap = load_source_cap[
                            (emitter_id, t, rate_index)
                        ]
                        service_selected = load_capacity_limit_choice[
                            (vessel_id, emitter_id, t)
                        ]
                        remaining_capacity = (
                            float(vessel.capacity_t) - cargo[(vessel_id, t)]
                        )
                        prob += load[(vessel_id, emitter_id, t)] <= service_cap
                        prob += load[(vessel_id, emitter_id, t)] >= (
                            service_cap
                            - float(load_cap_tph) * (1 - service_selected)
                            - float(load_cap_tph) * (1 - active)
                        )
                        prob += load[(vessel_id, emitter_id, t)] >= (
                            remaining_capacity
                            - float(vessel.capacity_t) * service_selected
                            - float(vessel.capacity_t) * (1 - active)
                        )
            prob += (
                unload[(vessel_id, t)]
                <= vessel.unloading_rate_tph * _wait_expr(arc_vars, wait_arc, vessel_id, terminal_id, t)
            )
            prob += unload[(vessel_id, t)] <= vessel.unloading_rate_tph * unload_active[(vessel_id, t)]
            prob += unload_active[(vessel_id, t)] <= _wait_expr(arc_vars, wait_arc, vessel_id, terminal_id, t)
            prob += unload[(vessel_id, t)] <= cargo[(vessel_id, t)]
            if environment_aligned_service:
                active = unload_active[(vessel_id, t)]
                prob += active <= cargo_positive[(vessel_id, t)]
                prob += active <= terminal_eligible[(vessel_id, t)]

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
            # source_stock is already bounded by buffer capacity, so only the
            # current capture can exceed that capacity in source_pre_load.
            source_m = max(0.0, capture_t)
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
            if environment_aligned_service:
                active_loading = pulp.lpSum(
                    load_active[(vessel_id, emitter_id, t)] for vessel_id in env.vessel_ids
                )
                for vessel_index, vessel_id in enumerate(env.vessel_ids):
                    prob += active_loading >= (
                        _wait_expr(arc_vars, wait_arc, vessel_id, emitter_id, t)
                        + cargo_space[(vessel_id, t)]
                        - 1
                    )
                    for earlier_vessel_id in env.vessel_ids[:vessel_index]:
                        prob += load_active[(vessel_id, emitter_id, t)] <= (
                            2
                            - _wait_expr(arc_vars, wait_arc, earlier_vessel_id, emitter_id, t)
                            - cargo_space[(earlier_vessel_id, t)]
                        )
            prob += (
                pulp.lpSum(load[(vessel_id, emitter_id, t)] for vessel_id in env.vessel_ids)
                <= emitter.loading_rate_tph
            )

    if source_visit_vent_cuts:
        for emitter_id in env.emitter_ids:
            emitter = env.network.entities[emitter_id]
            cumulative_supply_t = float(state.entity_inventory_t.get(emitter_id, 0.0))
            cumulative_vent_terms = []
            source_arrivals = {
                vessel_id: [
                    (index, arc)
                    for index, arc in enumerate(arcs)
                    if arc.vessel_id == vessel_id
                    and arc.is_sailing
                    and arc.arrives_within_horizon
                    and arc.destination_id == emitter_id
                ]
                for vessel_id in env.vessel_ids
            }
            for t in hours:
                cumulative_supply_t += _capture_tonnes(
                    env, scenario, emitter_id, start_step + t
                )
                cumulative_vent_terms.append(vent[(emitter_id, t)])
                if (
                    (t + 1) % source_visit_vent_cut_stride_h != 0
                    and t != H - 1
                ):
                    continue
                visit_capacity_terms = []
                initial_visit_capacity_t = 0.0
                for vessel_id in env.vessel_ids:
                    vessel_capacity_t = float(
                        env.network.entities[vessel_id].capacity_t
                    )
                    start = starts[vessel_id]
                    if start.node_id == emitter_id and start.start_h <= t:
                        initial_visit_capacity_t += vessel_capacity_t
                    visit_capacity_terms.extend(
                        vessel_capacity_t * arc_vars[index]
                        for index, arc in source_arrivals[vessel_id]
                        if arc.end_h <= t
                    )
                prob += pulp.lpSum(cumulative_vent_terms) >= (
                    cumulative_supply_t
                    - float(emitter.buffer_capacity_t)
                    - initial_visit_capacity_t
                    - pulp.lpSum(visit_capacity_terms)
                )

    overflow_risk = _add_overflow_risk_constraints(
        prob,
        env,
        scenario,
        start_step,
        H,
        source_stock,
        0.0 if economic_objective else objective_weights.overflow_risk_lookahead_h,
    )

    if env.automatic_well_control:
        well_request = automatic_well_request
    else:
        well_request = {
            (well_id, t): pulp.lpSum(
                mtpa_to_tph(WELL_RATE_LEVELS_MTPA[rate_index])
                * well_choice[(well_id, t, rate_index)]
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
    _add_continuous_automatic_well_request_constraints(
        prob,
        env,
        scenario,
        automatic_well_request,
        automatic_well_regime,
        well_inj,
        automatic_well_physical_max,
        H,
    )

    if env.automatic_well_control:
        max_hourly_injection_t = max(
            (
                sum(
                    automatic_well_physical_max[(well_id, t)]
                    for well_id in env.well_ids
                )
                for t in hours
            ),
            default=0.0,
        )
    else:
        max_hourly_injection_t = max(
            (
                sum(
                    max(
                        mtpa_to_tph(
                            WELL_RATE_LEVELS_MTPA[rate_index]
                        )
                        for rate_index in well_rate_options[
                            (well_id, t)
                        ]
                    )
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
            if environment_aligned_service and berth_count > 0:
                active_unloading = pulp.lpSum(
                    unload_active[(vessel_id, t)] for vessel_id in vessels_for_terminal
                )
                for vessel_id in vessels_for_terminal:
                    prob += active_unloading >= terminal_eligible[(vessel_id, t)]
                if fifo_diagnostic_mode == "full":
                    ordered_vessels = sorted(vessels_for_terminal)
                    priority_scale = len(ordered_vessels) + 1
                    max_priority = (H + len(env.vessel_ids)) * priority_scale + len(ordered_vessels)
                    for vessel_index, vessel_id in enumerate(ordered_vessels):
                        active = unload_active[(vessel_id, t)]
                        vessel_priority = (
                            priority_scale * terminal_queue_age[(vessel_id, t)]
                            + len(ordered_vessels)
                            - vessel_index
                        )
                        for other_index, other_vessel_id in enumerate(ordered_vessels):
                            other_priority = (
                                priority_scale * terminal_queue_age[(other_vessel_id, t)]
                                + len(ordered_vessels)
                                - other_index
                            )
                            prob += vessel_priority + max_priority * (1 - active) >= (
                                other_priority
                                - max_priority * (1 - terminal_eligible[(other_vessel_id, t)])
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
    _add_single_well_dynamic_bhp_constraints(
        prob,
        env,
        well_inj,
        H,
        well_choice=well_choice,
        well_rate_options=well_rate_options,
        automatic_well_regime=automatic_well_regime,
        automatic_well_physical_max=automatic_well_physical_max,
    )

    initial_terminal_t = sum(float(state.entity_inventory_t.get(tid, 0.0)) for tid in env.terminal_ids)
    prob += terminal_stock[0] == initial_terminal_t

    if terminal_visit_cuts:
        cumulative_unload_terms = {
            vessel_id: [] for vessel_id in env.vessel_ids
        }
        cumulative_injection_terms = []
        terminal_arrivals = {
            vessel_id: [
                (index, arc)
                for index, arc in enumerate(arcs)
                if arc.vessel_id == vessel_id
                and arc.is_sailing
                and arc.arrives_within_horizon
                and arc.destination_id
                == str(env._routes[vessel_id]["destination"])
            ]
            for vessel_id in env.vessel_ids
        }
        for t in hours:
            cumulative_injection_terms.extend(
                well_inj[(well_id, t)] for well_id in env.well_ids
            )
            for vessel_id in env.vessel_ids:
                cumulative_unload_terms[vessel_id].append(
                    unload[(vessel_id, t)]
                )
            if (
                (t + 1) % terminal_visit_cut_stride_h != 0
                and t != H - 1
            ):
                continue
            visit_capacity_terms = []
            initial_visit_capacity_t = 0.0
            for vessel_id in env.vessel_ids:
                vessel_capacity_t = float(
                    env.network.entities[vessel_id].capacity_t
                )
                terminal_id = str(env._routes[vessel_id]["destination"])
                start = starts[vessel_id]
                initial_visit = int(
                    start.node_id == terminal_id and start.start_h <= t
                )
                completed_arrivals = pulp.lpSum(
                    arc_vars[index]
                    for index, arc in terminal_arrivals[vessel_id]
                    if arc.end_h <= t
                )
                visit_count = initial_visit + completed_arrivals
                prob += pulp.lpSum(cumulative_unload_terms[vessel_id]) <= (
                    vessel_capacity_t * visit_count
                )
                initial_visit_capacity_t += vessel_capacity_t * initial_visit
                visit_capacity_terms.extend(
                    vessel_capacity_t * arc_vars[index]
                    for index, arc in terminal_arrivals[vessel_id]
                    if arc.end_h <= t
                )
            prob += pulp.lpSum(cumulative_injection_terms) <= (
                initial_terminal_t
                + initial_visit_capacity_t
                + pulp.lpSum(visit_capacity_terms)
            )

    if service_reachability_cuts:
        source_arrivals = {
            (vessel_id, emitter_id): [
                (index, arc)
                for index, arc in enumerate(arcs)
                if arc.vessel_id == vessel_id
                and arc.is_sailing
                and arc.arrives_within_horizon
                and arc.destination_id == emitter_id
            ]
            for vessel_id in env.vessel_ids
            for emitter_id in env.emitter_ids
        }
        terminal_arrivals = {
            vessel_id: [
                (index, arc)
                for index, arc in enumerate(arcs)
                if arc.vessel_id == vessel_id
                and arc.is_sailing
                and arc.arrives_within_horizon
                and arc.destination_id
                == str(env._routes[vessel_id]["destination"])
            ]
            for vessel_id in env.vessel_ids
        }
        cumulative_load_by_vessel = {
            vessel_id: [] for vessel_id in env.vessel_ids
        }
        cumulative_load_by_source = {
            emitter_id: [] for emitter_id in env.emitter_ids
        }
        cumulative_unload_by_vessel = {
            vessel_id: [] for vessel_id in env.vessel_ids
        }
        cumulative_vent_by_source = {
            emitter_id: [] for emitter_id in env.emitter_ids
        }
        cumulative_supply_by_source = {
            emitter_id: float(state.entity_inventory_t.get(emitter_id, 0.0))
            for emitter_id in env.emitter_ids
        }
        cumulative_injection_terms = []

        for t in hours:
            cumulative_injection_terms.extend(
                well_inj[(well_id, t)] for well_id in env.well_ids
            )
            for vessel_id in env.vessel_ids:
                cumulative_unload_by_vessel[vessel_id].append(
                    unload[(vessel_id, t)]
                )
                for emitter_id in env.emitter_ids:
                    load_term = load[(vessel_id, emitter_id, t)]
                    cumulative_load_by_vessel[vessel_id].append(load_term)
                    cumulative_load_by_source[emitter_id].append(load_term)
            for emitter_id in env.emitter_ids:
                cumulative_vent_by_source[emitter_id].append(
                    vent[(emitter_id, t)]
                )
                cumulative_supply_by_source[emitter_id] += _capture_tonnes(
                    env, scenario, emitter_id, start_step + t
                )

            if (
                (t + 1) % service_reachability_cut_stride_h != 0
                and t != H - 1
            ):
                continue

            source_service_capacity = {
                emitter_id: [] for emitter_id in env.emitter_ids
            }
            terminal_service_capacity = []
            for vessel_id in env.vessel_ids:
                vessel = env.network.entities[vessel_id]
                vessel_capacity_t = float(vessel.capacity_t)
                initial_cargo_t = float(
                    state.entity_inventory_t.get(vessel_id, 0.0)
                )
                vessel_source_capacity = []
                start = starts[vessel_id]

                for emitter_id in env.emitter_ids:
                    emitter = env.network.entities[emitter_id]
                    if start.node_id == emitter_id and start.start_h <= t:
                        initial_hours = t - start.start_h + 1
                        initial_capacity_t = min(
                            max(0.0, vessel_capacity_t - initial_cargo_t),
                            float(emitter.loading_rate_tph) * initial_hours,
                        )
                        vessel_source_capacity.append(initial_capacity_t)
                        source_service_capacity[emitter_id].append(
                            initial_capacity_t
                        )
                    for index, arc in source_arrivals[(vessel_id, emitter_id)]:
                        if arc.end_h > t:
                            continue
                        service_hours = t - arc.end_h + 1
                        reachable_capacity_t = min(
                            vessel_capacity_t,
                            float(emitter.loading_rate_tph) * service_hours,
                        )
                        term = reachable_capacity_t * arc_vars[index]
                        vessel_source_capacity.append(term)
                        source_service_capacity[emitter_id].append(term)


                prob += pulp.lpSum(
                    cumulative_load_by_vessel[vessel_id]
                ) <= pulp.lpSum(vessel_source_capacity)

                terminal_id = str(env._routes[vessel_id]["destination"])
                if start.node_id == terminal_id and start.start_h <= t:
                    initial_hours = t - start.start_h + 1
                    terminal_service_capacity.append(
                        min(
                            initial_cargo_t,
                            float(vessel.unloading_rate_tph) * initial_hours,
                        )
                    )
                vessel_terminal_capacity = []
                for index, arc in terminal_arrivals[vessel_id]:
                    if arc.end_h > t:
                        continue
                    service_hours = t - arc.end_h + 1
                    reachable_capacity_t = min(
                        vessel_capacity_t,
                        float(vessel.unloading_rate_tph) * service_hours,
                    )
                    term = reachable_capacity_t * arc_vars[index]
                    vessel_terminal_capacity.append(term)
                    terminal_service_capacity.append(term)
                initial_terminal_service_t = (
                    min(
                        initial_cargo_t,
                        float(vessel.unloading_rate_tph)
                        * (t - start.start_h + 1),
                    )
                    if start.node_id == terminal_id and start.start_h <= t
                    else 0.0
                )
                prob += pulp.lpSum(
                    cumulative_unload_by_vessel[vessel_id]
                ) <= initial_terminal_service_t + pulp.lpSum(
                    vessel_terminal_capacity
                )

            for emitter_id in env.emitter_ids:
                reachable_load = pulp.lpSum(
                    source_service_capacity[emitter_id]
                )
                prob += pulp.lpSum(
                    cumulative_load_by_source[emitter_id]
                ) <= reachable_load
                emitter = env.network.entities[emitter_id]
                prob += pulp.lpSum(
                    cumulative_vent_by_source[emitter_id]
                ) >= (
                    cumulative_supply_by_source[emitter_id]
                    - float(emitter.buffer_capacity_t)
                    - reachable_load
                )

            prob += pulp.lpSum(cumulative_injection_terms) <= (
                initial_terminal_t + pulp.lpSum(terminal_service_capacity)
            )

    captured_from_operations_t = sum(
        _capture_tonnes(env, scenario, emitter_id, start_step + t)
        for emitter_id in env.emitter_ids
        for t in hours
    )
    stored_expr = pulp.lpSum(well_inj[(well_id, t)] for well_id in env.well_ids for t in hours)
    initial_sailing_fuel_hours = _initial_sailing_fuel_hours(starts, H)
    operating_cost_expr = (
        initial_sailing_fuel_hours * params.vessel_fuel_eur_per_h_sailing
        + _sailing_cost_expression(arcs, arc_vars, params)
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
    boundary_cargo = (
        _add_route_cargo_flow_linking(
            prob,
            env,
            horizon_h=H,
            arcs=arcs,
            starts=starts,
            incoming=incoming,
            outgoing=outgoing,
            arc_vars=arc_vars,
            cargo=cargo,
            load=load,
            unload=unload,
        )
        if route_cargo_flow_linking
        else None
    )
    terminal_cleanup_model = (
        _add_terminal_cleanup_value_model(
            prob,
            env,
            horizon_h=H,
            arcs=arcs,
            incoming=incoming,
            arc_vars=arc_vars,
            cargo=cargo,
            source_stock=source_stock,
            terminal_stock=terminal_stock,
            params=params,
            relax_integrality=group_is_relaxed("cleanup"),
            boundary_cargo=boundary_cargo,
            unary_trip_slots=cleanup_unary_trip_slots,
            aggregate_full_trip_dominance=(
                cleanup_aggregate_full_trip_dominance
            ),
            return_partition_cut=cleanup_return_partition_cut,
            source_mode_partition_cut=cleanup_source_mode_partition_cut,
            weather_aware_sailing_lower_bound=(
                weather_aware_cleanup_sailing_lower_bound
            ),
            source_headroom_risk=cleanup_source_headroom_risk,
        )
        if terminal_cleanup_value
        else None
    )
    if terminal_cleanup_model is not None:
        if min_total_cleanup_trips is not None:
            prob += pulp.lpSum(
                terminal_cleanup_model.trips.values()
            ) >= min_total_cleanup_trips
        for source_id, trip_count in fixed_cleanup_trips_by_source.items():
            prob += pulp.lpSum(
                terminal_cleanup_model.trips[(vessel_id, source_id)]
                for vessel_id in env.vessel_ids
            ) == int(trip_count)
        for key, trip_count in fixed_cleanup_trips_by_vessel_source.items():
            prob += terminal_cleanup_model.trips[key] == int(trip_count)
        for vessel_id, node_id in fixed_boundary_node_by_vessel.items():
            prob += terminal_cleanup_model.end_at[(vessel_id, node_id)] == 1
    terminal_cleanup_cost_expr = (
        terminal_cleanup_model.cost_expr if terminal_cleanup_model is not None else 0.0
    )
    augmented_operating_cost_expr = operating_cost_expr + terminal_cleanup_cost_expr
    if economic_objective:
        weighted_objective = augmented_operating_cost_expr + params.carbon_price_eur_per_t * vent_expr
    else:
        weighted_objective = (
            objective_weights.operating_cost_weight * augmented_operating_cost_expr
            + objective_weights.vent_eur_per_t * vent_expr
            + objective_weights.overflow_risk_eur_per_t * pulp.lpSum(overflow_risk.values())
            - stored_expr * reward_per_t
        )
    if max_nonstored_t is not None:
        prob += vent_expr + end_unstored_inventory_expr <= float(max_nonstored_t) + 1e-3
    if max_vented_t is not None:
        prob += vent_expr <= float(max_vented_t) + 1e-3
    if max_end_unstored_t is not None:
        prob += end_unstored_inventory_expr <= float(max_end_unstored_t) + 1e-3
    if max_execution_vented_t is not None or max_execution_unstored_t is not None:
        execution_h = max(0, min(H, int(execution_boundary_h or H)))
        if max_execution_vented_t is not None:
            prob += pulp.lpSum(
                vent[(emitter_id, t)]
                for emitter_id in env.emitter_ids
                for t in range(execution_h)
            ) <= float(max_execution_vented_t) + 1e-3
        if max_execution_unstored_t is not None:
            prob += (
                pulp.lpSum(
                    source_stock[(emitter_id, execution_h)] for emitter_id in env.emitter_ids
                )
                + terminal_stock[execution_h]
                + pulp.lpSum(cargo[(vessel_id, execution_h)] for vessel_id in env.vessel_ids)
                <= float(max_execution_unstored_t) + 1e-3
            )

    constraint_redundancy = (
        _audit_constraint_redundancy(prob)
        if constraint_redundancy_audit
        else None
    )
    use_warm_start = warm_start_native_actions_by_hour is not None
    if fix_warm_start_vessel_routes and warm_start_native_actions_by_hour is None:
        raise ValueError(
            "fix_warm_start_vessel_routes requires warm-start native actions"
        )
    mip_start_audit = None
    mip_start_terminal_cleanup_cost = None
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
            terminal_eligible=terminal_eligible,
            terminal_queue_age=terminal_queue_age,
            load=load,
            load_active=load_active,
            load_limit_choice=load_limit_choice,
            load_rate_groups=load_rate_groups,
            load_source_cap=load_source_cap,
            load_source_limit_choice=load_source_limit_choice,
            load_capacity_limit_choice=load_capacity_limit_choice,
            unload=unload,
            unload_active=unload_active,
            unload_limit_choice=unload_limit_choice,
            source_stock=source_stock,
            source_ready=source_ready,
            source_overflow_active=source_overflow_active,
            terminal_stock=terminal_stock,
            well_inj=well_inj,
            automatic_well_request=automatic_well_request,
            automatic_well_regime=automatic_well_regime,
            automatic_well_physical_max=automatic_well_physical_max,
            injection_limit_choice=injection_limit_choice,
            vent=vent,
        )
        if (
            terminal_cleanup_mip_start_mode == "complete"
            and
            terminal_cleanup_model is not None
            and min_total_cleanup_trips is None
            and not fixed_cleanup_trips_by_source
            and not fixed_cleanup_trips_by_vessel_source
            and not fixed_boundary_node_by_vessel
        ):
            mip_start_terminal_cleanup_cost = (
                _seed_terminal_cleanup_mip_start(
                    prob,
                    env,
                    warm_start_native_actions_by_hour,
                    horizon_h=H,
                    params=params,
                    aggregate_full_trip_dominance=(
                        cleanup_aggregate_full_trip_dominance
                    ),
                    return_partition_cut=cleanup_return_partition_cut,
                    source_mode_partition_cut=(
                        cleanup_source_mode_partition_cut
                    ),
                    weather_aware_sailing_lower_bound=(
                        weather_aware_cleanup_sailing_lower_bound
                    ),
                    source_headroom_risk=cleanup_source_headroom_risk,
                )
            )
        if fix_warm_start_vessel_routes:
            for variable in arc_vars.values():
                prob += variable == round(_value(variable))
        mip_start_audit = _audit_mip_start(prob)

    if export_model_lp_path is not None:
        export_path = Path(export_model_lp_path)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        prob.setObjective(weighted_objective)
        prob.writeLP(str(export_path), writeSOS=1)

    stage_diagnostics: list[CplexStageDiagnostic] = []

    def solve_stage(
        stage: str,
        objective,
        limit_s: float | None,
        warm_start: bool,
    ) -> str:
        solver = _make_cplex_cmd(
            cplex_path=cplex_path,
            time_limit_s=limit_s,
            mip_gap_rel=mip_gap_rel,
            mip_gap_abs=mip_gap_abs,
            threads=threads,
            options=cplex_options,
            warm_start=warm_start,
            msg=msg,
        )
        solve_start = time.perf_counter()
        try:
            prob.solve(solver)
        except pulp.PulpSolverError as exc:
            raise RuntimeError(
                "CPLEX_CMD failed. Install IBM ILOG CPLEX, add the cplex executable "
                "to PATH, or pass cplex_path=... to solve_full_scenario_with_cplex()."
            ) from exc
        wall_time_s = time.perf_counter() - solve_start
        status = _solution_status_label(prob.status, getattr(prob, "sol_status", None))
        parsed = _parse_cplex_log(getattr(solver, "last_stdout", ""), warm_start)
        if (
            status == "Infeasible"
            and "no integer solution" in str(parsed["termination_reason"]).lower()
        ):
            status = "No Integer Solution"
        objective_value = None
        if status in {"Optimal", "Integer Feasible"}:
            objective_value = _value(objective)
        best_bound = parsed["best_bound"]
        relative_gap = parsed["relative_gap"]
        objective_constant = float(getattr(objective, "constant", 0.0) or 0.0)
        if best_bound is not None:
            # LP format omits objective constants, so CPLEX's logged incumbent
            # and bound are both shifted.  _value(objective) includes the
            # constant; restore it to the bound before reporting a gap.
            best_bound += objective_constant
        if objective_value is not None and best_bound is not None:
            relative_gap = max(0.0, objective_value - best_bound) / max(
                1e-10,
                abs(objective_value),
            )
        if (
            status == "Optimal"
            and "tolerance" not in str(parsed["termination_reason"]).lower()
            and objective_value is not None
        ):
            if best_bound is None:
                best_bound = objective_value
            if relative_gap is None:
                relative_gap = 0.0
        stage_diagnostics.append(
            CplexStageDiagnostic(
                stage=stage,
                time_limit_s=limit_s,
                wall_time_s=wall_time_s,
                status=status,
                objective_value=objective_value,
                best_bound=best_bound,
                relative_gap=relative_gap,
                nodes=parsed["nodes"],
                iterations=parsed["iterations"],
                reduced_rows=parsed["reduced_rows"],
                reduced_columns=parsed["reduced_columns"],
                reduced_nonzeros=parsed["reduced_nonzeros"],
                warm_start_requested=warm_start,
                warm_start_accepted=parsed["warm_start_accepted"],
                warm_start_message=parsed["warm_start_message"],
                termination_reason=parsed["termination_reason"],
                raw_log=getattr(solver, "last_stdout", ""),
            )
        )
        return status

    def snapshot_solution() -> dict[pulp.LpVariable, float | None]:
        return {variable: variable.varValue for variable in prob.variables()}

    def restore_solution(values: dict[pulp.LpVariable, float | None]) -> None:
        for variable, value in values.items():
            variable.varValue = value

    if lexicographic_vent_first:
        stage_start = time.perf_counter()
        prob.setObjective(vent_expr)
        status = solve_stage("vent", vent_expr, time_limit_s, use_warm_start)
        if status in {"Optimal", "Integer Feasible"}:
            last_feasible_status = status
            last_feasible_values = snapshot_solution()
            optimal_vent_t = max(0.0, _value(vent_expr))
            prob += vent_expr <= optimal_vent_t + 1e-3
            remaining_s = None if time_limit_s is None else max(
                1.0,
                float(time_limit_s) - (time.perf_counter() - stage_start),
            )
            stage_two_s = None if remaining_s is None else max(1.0, remaining_s / 2.0)
            prob.setObjective(end_unstored_inventory_expr)
            status = solve_stage("end_unstored", end_unstored_inventory_expr, stage_two_s, True)
            if status in {"Optimal", "Integer Feasible"}:
                last_feasible_status = status
                last_feasible_values = snapshot_solution()
                optimal_end_unstored_t = max(0.0, _value(end_unstored_inventory_expr))
                prob += end_unstored_inventory_expr <= optimal_end_unstored_t + 1e-3
                remaining_s = None if time_limit_s is None else max(
                    1.0,
                    float(time_limit_s) - (time.perf_counter() - stage_start),
                )
                prob.setObjective(augmented_operating_cost_expr)
                status = solve_stage(
                    "operating_cost", augmented_operating_cost_expr, remaining_s, True
                )
                if status not in {"Optimal", "Integer Feasible"}:
                    restore_solution(last_feasible_values)
                    status = last_feasible_status
            else:
                restore_solution(last_feasible_values)
                status = last_feasible_status
    else:
        prob.setObjective(weighted_objective)
        status = solve_stage("weighted", weighted_objective, time_limit_s, use_warm_start)
    vessel_actions_by_hour = _extract_vessel_actions(env, H, arcs, arc_vars)
    well_request_tph_by_hour = {
        well_id: [
            _value(well_request[(well_id, t)])
            for t in hours
        ]
        for well_id in env.well_ids
    }
    if env.automatic_well_control:
        well_rate_indices_by_hour = {}
        native_actions_by_hour = [
            {
                "vessels": [
                    vessel_actions_by_hour[vessel_id][t]
                    for vessel_id in env.vessel_ids
                ],
            }
            for t in hours
        ]
    else:
        well_rate_indices_by_hour = _extract_well_rate_indices(
            env,
            H,
            well_rate_options,
            well_choice,
        )
        native_actions_by_hour = [
            {
                "vessels": [
                    vessel_actions_by_hour[vessel_id][t]
                    for vessel_id in env.vessel_ids
                ],
                "wells": [
                    well_rate_indices_by_hour[well_id][t]
                    for well_id in env.well_ids
                ],
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
    cost = _native_cost_breakdown(
        env,
        arcs,
        arc_vars,
        load,
        unload,
        stored_t,
        params,
        initial_sailing_fuel_hours=initial_sailing_fuel_hours,
    )
    total_cost = cost.operating_cost + vented_t * params.carbon_price_eur_per_t
    terminal_cleanup_cost = (
        _value(terminal_cleanup_model.cost_expr)
        if terminal_cleanup_model is not None
        else 0.0
    )
    terminal_cleanup_vessel_fuel = (
        _value(terminal_cleanup_model.vessel_fuel_expr)
        if terminal_cleanup_model is not None
        else 0.0
    )
    terminal_cleanup_conditioning = (
        _value(terminal_cleanup_model.conditioning_expr)
        if terminal_cleanup_model is not None
        else 0.0
    )
    terminal_cleanup_reconditioning = (
        _value(terminal_cleanup_model.reconditioning_expr)
        if terminal_cleanup_model is not None
        else 0.0
    )
    terminal_cleanup_loading = (
        _value(terminal_cleanup_model.loading_expr)
        if terminal_cleanup_model is not None
        else 0.0
    )
    terminal_cleanup_unloading = (
        _value(terminal_cleanup_model.unloading_expr)
        if terminal_cleanup_model is not None
        else 0.0
    )
    terminal_cleanup_headroom_risk = (
        _value(terminal_cleanup_model.headroom_risk_expr)
        if terminal_cleanup_model is not None
        else 0.0
    )
    overflow_risk_t = _overflow_risk_value(
        env,
        scenario,
        start_step,
        H,
        source_stock,
        float(getattr(env.config, "overflow_risk_lookahead_h", 0.0)),
    )
    if economic_objective:
        objective_value = cost.operating_cost + params.carbon_price_eur_per_t * vented_t
    else:
        objective_value = control_objective_value(
            objective_weights,
            operating_cost=cost.operating_cost,
            vented_t=vented_t,
            stored_t=stored_t,
            overflow_risk_t=overflow_risk_t,
        )
    net_reward = -objective_value
    augmented_objective_value = _value(weighted_objective)
    departures, arrivals = _extract_departures_and_arrivals(env, arcs, arc_vars, H)
    validation = _validate_solution(
        status=status,
        binary_values=[
            *[arc_vars[index].value() for index in arc_vars],
            *[var.value() for var in cargo_positive.values()],
            *[var.value() for var in cargo_space.values()],
            *[var.value() for var in terminal_eligible.values()],
            *[var.value() for var in load_active.values()],
            *[var.value() for var in load_limit_choice.values()],
            *[var.value() for var in load_source_limit_choice.values()],
            *[var.value() for var in load_capacity_limit_choice.values()],
            *[var.value() for var in unload_active.values()],
            *[var.value() for var in unload_limit_choice.values()],
            *[var.value() for var in source_overflow_active.values()],
            *[var.value() for var in injection_limit_choice.values()],
            *[var.value() for var in well_choice.values()],
            *[var.value() for var in automatic_well_regime.values()],
            *(
                [var.value() for var in terminal_cleanup_model.binary_variables]
                if terminal_cleanup_model is not None
                else []
            ),
        ],
        stored_t=stored_t,
        vented_t=vented_t,
        in_transit_t=in_transit_t,
        captured_from_operations_t=captured_from_operations_t,
        initial_in_transit_t=initial_in_transit_t,
        max_storable_from_deliveries_t=initial_terminal_t + unloaded_t,
        integrality_tol=1e-5 if environment_aligned_service else 1e-6,
    )
    solution_audit = _audit_mip_start(prob, tolerance=1e-5)
    cargo_t_by_hour = {
        vessel_id: tuple(_value(cargo[(vessel_id, t)]) for t in range(H + 1))
        for vessel_id in env.vessel_ids
    }
    source_stock_t_by_hour = {
        emitter_id: tuple(_value(source_stock[(emitter_id, t)]) for t in range(H + 1))
        for emitter_id in env.emitter_ids
    }
    terminal_stock_t_by_hour = tuple(_value(terminal_stock[t]) for t in range(H + 1))
    load_t_by_hour = {
        vessel_id: tuple(
            sum(_value(load[(vessel_id, emitter_id, t)]) for emitter_id in env.emitter_ids)
            for t in hours
        )
        for vessel_id in env.vessel_ids
    }
    unload_t_by_hour = {
        vessel_id: tuple(_value(unload[(vessel_id, t)]) for t in hours)
        for vessel_id in env.vessel_ids
    }
    diagnostic_variable_values: dict[str, object] = {}
    if mip_start_terminal_cleanup_cost is not None:
        diagnostic_variable_values["mip_start_terminal_cleanup_cost"] = (
            mip_start_terminal_cleanup_cost
        )
    if constraint_redundancy is not None:
        diagnostic_variable_values["constraint_redundancy"] = (
            constraint_redundancy
        )
    if integrality_relax_groups and status in {"Optimal", "Integer Feasible"}:
        tolerance = 1e-8

        def positive_rows(variables):
            return [
                {"key": tuple(str(part) for part in key) if isinstance(key, tuple) else str(key),
                 "value": _value(variable)}
                for key, variable in variables.items()
                if _value(variable) > tolerance
            ]

        grouped_variables = {
            "route": list(arc_vars.values()),
            "service": [
                *cargo_positive.values(),
                *cargo_space.values(),
                *load_active.values(),
                *load_limit_choice.values(),
                *load_source_limit_choice.values(),
                *load_capacity_limit_choice.values(),
                *unload_active.values(),
                *unload_limit_choice.values(),
            ],
            "fifo": list(terminal_eligible.values()),
            "overflow": list(source_overflow_active.values()),
            "injection": [
                *well_choice.values(),
                *automatic_well_regime.values(),
                *injection_limit_choice.values(),
            ],
            "cleanup": (
                [
                    *terminal_cleanup_model.binary_variables,
                    *terminal_cleanup_model.trips.values(),
                ]
                if terminal_cleanup_model is not None
                else []
            ),
        }
        diagnostic_variable_values["fractional_counts"] = {
            group: sum(
                tolerance < _value(variable) < 1.0 - tolerance
                for variable in variables
            )
            for group, variables in grouped_variables.items()
        }
        service_variable_groups = {
            "cargo_positive": list(cargo_positive.values()),
            "cargo_space": list(cargo_space.values()),
            "load_active": list(load_active.values()),
            "load_limit_choice": list(load_limit_choice.values()),
            "load_source_limit_choice": list(
                load_source_limit_choice.values()
            ),
            "load_capacity_limit_choice": list(
                load_capacity_limit_choice.values()
            ),
            "unload_active": list(unload_active.values()),
            "unload_limit_choice": list(unload_limit_choice.values()),
        }
        diagnostic_variable_values["fractional_service_breakdown"] = {
            group: sum(
                tolerance < _value(variable) < 1.0 - tolerance
                for variable in variables
            )
            for group, variables in service_variable_groups.items()
        }
        diagnostic_variable_values["positive_arcs"] = [
            {
                "index": index,
                "vessel_id": arc.vessel_id,
                "start_h": arc.start_h,
                "end_h": arc.end_h,
                "origin_id": arc.origin_id,
                "destination_id": arc.destination_id,
                "is_sailing": arc.is_sailing,
                "arrives_within_horizon": arc.arrives_within_horizon,
                "remaining_cleanup_fuel_h": arc.remaining_cleanup_fuel_h,
                "value": _value(arc_vars[index]),
            }
            for index, arc in enumerate(arcs)
            if _value(arc_vars[index]) > tolerance
        ]
        diagnostic_variable_values["positive_loads"] = [
            {
                "vessel_id": vessel_id,
                "source_id": emitter_id,
                "hour": t,
                "load_t": _value(load[(vessel_id, emitter_id, t)]),
                "wait_fraction": _value(
                    _wait_expr(
                        arc_vars, wait_arc, vessel_id, emitter_id, t
                    )
                ),
                "active": _value(load_active[(vessel_id, emitter_id, t)]),
                "cargo_before_t": _value(cargo[(vessel_id, t)]),
            }
            for vessel_id in env.vessel_ids
            for emitter_id in env.emitter_ids
            for t in hours
            if _value(load[(vessel_id, emitter_id, t)]) > tolerance
        ]
        diagnostic_variable_values["positive_unloads"] = [
            {
                "vessel_id": vessel_id,
                "hour": t,
                "unload_t": _value(unload[(vessel_id, t)]),
                "wait_fraction": _value(
                    _wait_expr(
                        arc_vars,
                        wait_arc,
                        vessel_id,
                        str(env._routes[vessel_id]["destination"]),
                        t,
                    )
                ),
                "active": _value(unload_active[(vessel_id, t)]),
                "eligible": _value(terminal_eligible[(vessel_id, t)]),
                "cargo_before_t": _value(cargo[(vessel_id, t)]),
            }
            for vessel_id in env.vessel_ids
            for t in hours
            if _value(unload[(vessel_id, t)]) > tolerance
        ]
        diagnostic_variable_values["boundary"] = {
            "vessel_cargo_t": {
                vessel_id: _value(cargo[(vessel_id, H)])
                for vessel_id in env.vessel_ids
            },
            "source_stock_t": {
                emitter_id: _value(source_stock[(emitter_id, H)])
                for emitter_id in env.emitter_ids
            },
            "terminal_stock_t": _value(terminal_stock[H]),
            "vessel_cargo_by_node_t": (
                positive_rows(boundary_cargo)
                if boundary_cargo is not None
                else []
            ),
        }
        if terminal_cleanup_model is not None:
            diagnostic_variable_values["cleanup"] = {
                "end_at": positive_rows(terminal_cleanup_model.end_at),
                "trips": positive_rows(terminal_cleanup_model.trips),
                "trip_slots": positive_rows(
                    terminal_cleanup_model.trip_slots
                ),
                "use": positive_rows(terminal_cleanup_model.use),
                "cargo_positive": positive_rows(
                    terminal_cleanup_model.cargo_positive
                ),
                "cargo_at_node": positive_rows(
                    terminal_cleanup_model.cargo_at_node
                ),
                "first_from": positive_rows(
                    terminal_cleanup_model.first_from
                ),
                "needs_return": positive_rows(
                    terminal_cleanup_model.needs_return
                ),
                "shipped": positive_rows(terminal_cleanup_model.shipped),
                "topup": positive_rows(terminal_cleanup_model.topup),
                "first_response_h": positive_rows(
                    terminal_cleanup_model.first_response_h
                ),
                "first_response_choice": positive_rows(
                    terminal_cleanup_model.first_response_choice
                ),
                "headroom_vent": positive_rows(
                    terminal_cleanup_model.headroom_vent
                ),
            }

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
        well_request_tph_by_hour=well_request_tph_by_hour,
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
        augmented_objective_value=augmented_objective_value,
        economic_objective=economic_objective,
        terminal_cleanup_value_enabled=terminal_cleanup_value,
        terminal_cleanup_cost=terminal_cleanup_cost,
        terminal_cleanup_vessel_fuel=terminal_cleanup_vessel_fuel,
        terminal_cleanup_conditioning=terminal_cleanup_conditioning,
        terminal_cleanup_reconditioning=terminal_cleanup_reconditioning,
        terminal_cleanup_loading=terminal_cleanup_loading,
        terminal_cleanup_unloading=terminal_cleanup_unloading,
        terminal_cleanup_headroom_risk=terminal_cleanup_headroom_risk,
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
        stage_diagnostics=tuple(stage_diagnostics),
        mip_start_audit=mip_start_audit,
        solution_audit=solution_audit,
        cargo_t_by_hour=cargo_t_by_hour,
        source_stock_t_by_hour=source_stock_t_by_hour,
        terminal_stock_t_by_hour=terminal_stock_t_by_hour,
        load_t_by_hour=load_t_by_hour,
        unload_t_by_hour=unload_t_by_hour,
        diagnostic_variable_values=diagnostic_variable_values,
    )


def replay_full_scenario_cplex_plan(
    env,
    result: FullScenarioCplexMilpResult,
    *,
    stored_tol_t: float = 1.1e-3,
) -> CplexMilpReplayResult:
    """Replay a CPLEX native-action plan through the RL environment.

    This consumes the current ``env`` state by calling ``env.step``. Use it on
    the same initial state used for the MILP solve, or on an equivalent fresh
    reset, when checking whether the MILP plan is executable by the RL wrapper.
    The default 1.1 kg mass tolerance covers the model's 1 kg activation
    epsilon plus solver roundoff while remaining negligible relative to
    operational flows.
    """

    if getattr(result, "economic_objective", False):
        env.config.reward_mode = "economic"
        env.config.vent_penalty_weight = 1.0
        env.config.operating_cost_weight = 1.0
        env.config.store_reward_eur_per_t = 0.0
        env.config.injection_reward_eur_per_t = 0.0

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
        [
            action_for_well_control_mode(env, action)
            for action in result.native_actions_by_hour
        ],
        horizon_h=result.horizon_h,
        expected=expectation,
        tolerances=ReplayTolerances(
            mass_t=stored_tol_t,
            cost_eur=max(1e-5, stored_tol_t),
            objective=max(1e-3, stored_tol_t),
        ),
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
    options: list[str] | None = None,
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
        options=options or [],
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


_CPLEX_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?"


def _parse_cplex_log(log: str, warm_start_requested: bool) -> dict[str, object]:
    best_bound = None
    relative_gap = None
    bound_matches = list(
        re.finditer(
            rf"Current MIP best bound\s*=\s*({_CPLEX_NUMBER})"
            r"(?:\s*\(gap\s*=\s*([^)]*)\))?",
            log,
        )
    )
    if bound_matches:
        match = bound_matches[-1]
        best_bound = float(match.group(1))
        gap_text = match.group(2) or ""
        gap_values = re.findall(_CPLEX_NUMBER, gap_text)
        if gap_values:
            relative_gap = float(gap_values[-1]) / 100.0

    iterations = None
    nodes = None
    solution_matches = list(
        re.finditer(
            r"Solution time\s*=.*?Iterations\s*=\s*([\d,]+)\s+Nodes\s*=\s*([\d,]+)",
            log,
        )
    )
    if solution_matches:
        iterations = int(solution_matches[-1].group(1).replace(",", ""))
        nodes = int(solution_matches[-1].group(2).replace(",", ""))

    reduced_rows = reduced_columns = reduced_nonzeros = None
    reduced_matches = list(
        re.finditer(
            r"Reduced MIP has\s+([\d,]+)\s+rows,\s+([\d,]+)\s+columns,"
            r"\s+and\s+([\d,]+)\s+nonzeros",
            log,
        )
    )
    if reduced_matches:
        reduced_rows, reduced_columns, reduced_nonzeros = (
            int(value.replace(",", "")) for value in reduced_matches[-1].groups()
        )

    termination_reason = ""
    termination_matches = re.findall(
        r"^MIP\s+-\s+([^:\r\n]+?)(?::|$)",
        log,
        flags=re.MULTILINE,
    )
    if termination_matches:
        termination_reason = termination_matches[-1].strip()

    warm_start_accepted = None
    warm_start_message = ""
    if warm_start_requested:
        mip_start_lines = [line for line in log.splitlines() if "MIP start" in line]
        if mip_start_lines:
            warm_start_message = " | ".join(line.strip() for line in mip_start_lines)
            rejected = any(
                token in line.lower()
                for line in mip_start_lines
                for token in ("no solution", "rejected", "infeasible", "failed")
            )
            warm_start_accepted = not rejected

    return {
        "best_bound": best_bound,
        "relative_gap": relative_gap,
        "nodes": nodes,
        "iterations": iterations,
        "reduced_rows": reduced_rows,
        "reduced_columns": reduced_columns,
        "reduced_nonzeros": reduced_nonzeros,
        "warm_start_accepted": warm_start_accepted,
        "warm_start_message": warm_start_message,
        "termination_reason": termination_reason,
    }


def _audit_constraint_redundancy(prob) -> dict[str, object]:
    """Find algebraically duplicate and same-LHS dominated constraint rows."""
    exact_groups: dict[tuple[object, ...], list[str]] = {}
    lhs_groups: dict[tuple[object, ...], list[tuple[str, float]]] = {}
    for name, constraint in prob.constraints.items():
        terms = sorted(
            (variable.name, float(coefficient))
            for variable, coefficient in constraint.items()
            if abs(float(coefficient)) > 1e-12
        )
        if not terms:
            continue
        scale = abs(terms[0][1])
        direction = 1.0 if terms[0][1] > 0.0 else -1.0
        normalized_terms = tuple(
            (variable_name, round(direction * coefficient / scale, 10))
            for variable_name, coefficient in terms
        )
        sense = int(constraint.sense)
        if direction < 0.0:
            sense = -sense
        rhs = round(direction * (-float(constraint.constant)) / scale, 10)
        lhs_key = (normalized_terms, sense)
        exact_groups.setdefault((*lhs_key, rhs), []).append(name)
        lhs_groups.setdefault(lhs_key, []).append((name, rhs))

    duplicate_groups = [
        (signature, names)
        for signature, names in exact_groups.items()
        if len(names) > 1
    ]
    dominated_groups: list[dict[str, object]] = []
    dominated_count = 0
    for (_terms, sense), rows in lhs_groups.items():
        rhs_values = {rhs for _name, rhs in rows}
        if sense == 0 or len(rhs_values) <= 1:
            continue
        strongest_rhs = min(rhs_values) if sense == -1 else max(rhs_values)
        redundant = [name for name, rhs in rows if rhs != strongest_rhs]
        if not redundant:
            continue
        dominated_count += len(redundant)
        dominated_groups.append(
            {
                "sense": "<=" if sense == -1 else ">=",
                "strongest_rhs": strongest_rhs,
                "strongest_names": [
                    name for name, rhs in rows if rhs == strongest_rhs
                ],
                "dominated_names": redundant,
            }
        )

    duplicate_groups.sort(key=lambda group: len(group[1]), reverse=True)
    dominated_groups.sort(
        key=lambda group: len(group["dominated_names"]), reverse=True
    )
    return {
        "constraint_count": len(prob.constraints),
        "exact_duplicate_group_count": len(duplicate_groups),
        "exact_duplicate_constraint_count": sum(
            len(names) - 1 for _signature, names in duplicate_groups
        ),
        "exact_duplicate_groups": [
            {
                "names": names,
                "sense": "<=" if signature[1] == -1 else (
                    ">=" if signature[1] == 1 else "=="
                ),
                "rhs": signature[2],
                "term_count": len(signature[0]),
                "terms": signature[0][:20],
            }
            for signature, names in duplicate_groups[:500]
        ],
        "same_lhs_dominated_constraint_count": dominated_count,
        "same_lhs_dominated_groups": dominated_groups[:100],
    }


def _audit_mip_start(prob, *, tolerance: float = 1e-6) -> CplexMipStartAudit:
    variables = list(prob.variables())
    missing = [variable.name for variable in variables if variable.varValue is None]
    bound_violations = 0
    integrality_violations = 0
    for variable in variables:
        value = variable.varValue
        if value is None:
            continue
        if variable.lowBound is not None and value < variable.lowBound - tolerance:
            bound_violations += 1
        if variable.upBound is not None and value > variable.upBound + tolerance:
            bound_violations += 1
        if variable.cat in {pulp.constants.LpBinary, pulp.constants.LpInteger}:
            if abs(value - round(value)) > tolerance:
                integrality_violations += 1

    evaluated_constraints = 0
    partial_constraints = 0
    violations: list[CplexMipStartViolation] = []
    sense_labels = {
        pulp.constants.LpConstraintLE: "<=",
        pulp.constants.LpConstraintEQ: "==",
        pulp.constants.LpConstraintGE: ">=",
    }
    for name, constraint in prob.constraints.items():
        constraint_variables = [variable for variable, _coefficient in constraint.items()]
        if any(variable.varValue is None for variable in constraint_variables):
            partial_constraints += 1
            continue
        evaluated_constraints += 1
        residual = float(constraint.value())
        if constraint.sense == pulp.constants.LpConstraintLE:
            violation = max(0.0, residual)
        elif constraint.sense == pulp.constants.LpConstraintGE:
            violation = max(0.0, -residual)
        else:
            violation = abs(residual)
        if violation > tolerance:
            violations.append(
                CplexMipStartViolation(
                    constraint=name,
                    sense=sense_labels.get(constraint.sense, str(constraint.sense)),
                    residual=residual,
                    violation=violation,
                    variable_names=tuple(variable.name for variable in constraint_variables),
                )
            )
    violations.sort(key=lambda item: item.violation, reverse=True)
    return CplexMipStartAudit(
        total_variables=len(variables),
        initialized_variables=len(variables) - len(missing),
        missing_variable_count=len(missing),
        missing_variable_names=tuple(missing[:100]),
        bound_violation_count=bound_violations,
        integrality_violation_count=integrality_violations,
        total_constraints=len(prob.constraints),
        evaluated_constraints=evaluated_constraints,
        partial_constraint_count=partial_constraints,
        violated_constraint_count=len(violations),
        max_constraint_violation=violations[0].violation if violations else 0.0,
        top_violations=tuple(violations[:100]),
    )


def _seed_terminal_cleanup_mip_start(
    prob,
    env,
    native_actions_by_hour: list[dict[str, list[int]]],
    *,
    horizon_h: int,
    params: EconomicParameters,
    aggregate_full_trip_dominance: bool,
    return_partition_cut: bool,
    source_mode_partition_cut: bool,
    weather_aware_sailing_lower_bound: bool,
    source_headroom_risk: bool,
) -> float | None:
    """Complete an action MIP start with its optimal compact cleanup tail."""

    replay_env = copy.deepcopy(env)
    replay = replay_native_actions(
        replay_env,
        [
            action_for_well_control_mode(replay_env, action)
            for action in native_actions_by_hour
        ],
        horizon_h=horizon_h,
        copy_env=False,
    )
    if not replay.is_executable:
        return None
    cleanup_cost, cleanup_values = _terminal_cleanup_solution_for_state(
        replay_env,
        params,
        aggregate_full_trip_dominance=aggregate_full_trip_dominance,
        return_partition_cut=return_partition_cut,
        source_mode_partition_cut=source_mode_partition_cut,
        weather_aware_sailing_lower_bound=(
            weather_aware_sailing_lower_bound
        ),
        source_headroom_risk=source_headroom_risk,
    )
    variables_by_name = {variable.name: variable for variable in prob.variables()}
    for name, value in cleanup_values.items():
        variable = variables_by_name.get(name)
        if variable is None:
            continue
        if variable.lowBound is not None:
            value = max(float(variable.lowBound), value)
        if variable.upBound is not None:
            value = min(float(variable.upBound), value)
        variable.setInitialValue(value)
    return cleanup_cost


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
    terminal_eligible=None,
    terminal_queue_age=None,
    load=None,
    load_active=None,
    load_limit_choice=None,
    load_rate_groups=None,
    load_source_cap=None,
    load_source_limit_choice=None,
    load_capacity_limit_choice=None,
    unload=None,
    unload_active=None,
    unload_limit_choice=None,
    source_stock=None,
    source_ready=None,
    source_overflow_active=None,
    terminal_stock=None,
    well_inj=None,
    automatic_well_request=None,
    automatic_well_regime=None,
    automatic_well_physical_max=None,
    injection_limit_choice=None,
    vent=None,
    shortfall=None,
    required_storage_t: float | None = None,
) -> None:
    if getattr(env, "automatic_well_control", False):
        native_actions_by_hour = _automatic_well_warm_start_actions(
            env,
            native_actions_by_hour,
            horizon_h,
        )
    for var in arc_vars.values():
        var.setInitialValue(0)
    selected_arc_indices, selected_wait_node = (
        _selected_native_action_route(
            env,
            arcs,
            starts,
            native_actions_by_hour,
            horizon_h,
        )
    )
    for arc_index in selected_arc_indices:
        arc_vars[arc_index].setInitialValue(1)

    for well_index, well_id in enumerate(env.well_ids):
        for t in range(min(horizon_h, len(native_actions_by_hour))):
            rate_index = _warm_start_well_rate_index(native_actions_by_hour, well_index, t)
            if rate_index not in well_rate_options[(well_id, t)]:
                continue
            for candidate in well_rate_options[(well_id, t)]:
                well_choice[(well_id, t, candidate)].setInitialValue(1 if candidate == rate_index else 0)
            if automatic_well_request is not None:
                rate_tph = float(
                    native_actions_by_hour[t]
                    .get("well_rates_tph", [0.0] * len(env.well_ids))[
                        well_index
                    ]
                )
                _set_start_value(
                    automatic_well_request,
                    (well_id, t),
                    rate_tph,
                )
                physical_max_tph = float(
                    (automatic_well_physical_max or {}).get(
                        (well_id, t),
                        0.0,
                    )
                )
                selected_regime = (
                    "off"
                    if rate_tph <= 1e-9
                    else (
                        "physical"
                        if abs(rate_tph - physical_max_tph) <= 1e-7
                        else "pressure"
                    )
                )
                for regime in ("off", "physical", "pressure"):
                    _set_start_value(
                        automatic_well_regime,
                        (well_id, t, regime),
                        1 if regime == selected_regime else 0,
                    )

    if scenario is None:
        return

    replayed_unload_by_hour = _replay_native_action_unloads(
        env,
        native_actions_by_hour,
        horizon_h,
    )
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
        load_rate_groups=load_rate_groups,
        load_source_cap=load_source_cap,
        load_source_limit_choice=load_source_limit_choice,
        load_capacity_limit_choice=load_capacity_limit_choice,
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
        replayed_unload_by_hour=replayed_unload_by_hour,
    )
    _seed_terminal_queue_values(
        env,
        selected_wait_node,
        cargo,
        terminal_eligible,
        terminal_queue_age,
        horizon_h,
    )


def _selected_native_action_route(
    env,
    arcs: list[_ActionArc],
    starts: dict[str, _PathStart],
    native_actions_by_hour: list[dict[str, list[int]]],
    horizon_h: int,
) -> tuple[list[int], dict[tuple[str, int], str]]:
    arc_by_step = {
        (arc.vessel_id, arc.start_h, arc.origin_id, arc.destination_id): index
        for index, arc in enumerate(arcs)
    }
    selected_arc_indices: list[int] = []
    selected_wait_node: dict[tuple[str, int], str] = {}
    for vessel_index, vessel_id in enumerate(env.vessel_ids):
        start = starts[vessel_id]
        if start.node_id is None or start.start_h >= horizon_h:
            continue
        node_id = start.node_id
        t = start.start_h
        while t < horizon_h:
            action = _warm_start_vessel_action(
                native_actions_by_hour,
                vessel_index,
                t,
            )
            destination_id = _native_action_destination(
                env,
                vessel_id,
                action,
            )
            if destination_id is None or destination_id == node_id:
                destination_id = node_id
            arc_index = arc_by_step.get(
                (vessel_id, t, node_id, destination_id)
            )
            if arc_index is None and destination_id != node_id:
                destination_id = node_id
                arc_index = arc_by_step.get(
                    (vessel_id, t, node_id, node_id)
                )
            if arc_index is None:
                break
            selected_arc_indices.append(arc_index)
            selected_arc = arcs[arc_index]
            if not selected_arc.is_sailing:
                selected_wait_node[(vessel_id, t)] = selected_arc.origin_id
            node_id = selected_arc.destination_id
            t = max(t + 1, selected_arc.end_h)
    return selected_arc_indices, selected_wait_node


def _warm_start_vessel_action(native_actions_by_hour: list[dict[str, list[int]]], vessel_index: int, t: int) -> int:
    try:
        return int(native_actions_by_hour[t]["vessels"][vessel_index])
    except (IndexError, KeyError, TypeError, ValueError):
        return VESSEL_WAIT


def _seed_terminal_queue_values(
    env,
    selected_wait_node: dict[tuple[str, int], str],
    cargo,
    terminal_eligible,
    terminal_queue_age,
    horizon_h: int,
) -> None:
    if cargo is None or terminal_eligible is None or terminal_queue_age is None:
        return
    initial_ages = _initial_terminal_queue_ages(env)
    previous_ages = {vessel_id: 0 for vessel_id in env.vessel_ids}
    for t in range(horizon_h):
        current_ages: dict[str, int] = {}
        for vessel_id in env.vessel_ids:
            terminal_id = str(env._routes[vessel_id]["destination"])
            cargo_value = float(cargo[(vessel_id, t)].varValue or 0.0)
            eligible = int(
                selected_wait_node.get((vessel_id, t)) == terminal_id
                and cargo_value >= 1e-3
            )
            age = 0
            if eligible:
                age = initial_ages.get(vessel_id, 0) if t == 0 else previous_ages[vessel_id] + 1
            _set_start_value(terminal_eligible, (vessel_id, t), eligible)
            _set_start_value(terminal_queue_age, (vessel_id, t), age)
            current_ages[vessel_id] = age
        previous_ages = current_ages


def _warm_start_well_rate_index(native_actions_by_hour: list[dict[str, list[int]]], well_index: int, t: int) -> int:
    try:
        return int(native_actions_by_hour[t]["wells"][well_index])
    except (IndexError, KeyError, TypeError, ValueError):
        return 0


def _automatic_well_warm_start_actions(
    env,
    native_actions_by_hour: list[dict[str, list[int]]],
    horizon_h: int,
) -> list[dict[str, list[int]]]:
    replay_env = copy.deepcopy(env)
    augmented: list[dict[str, list[int]]] = []
    for t, raw_action in enumerate(native_actions_by_hour):
        action = action_for_well_control_mode(replay_env, raw_action)
        with_wells = {
            "vessels": list(action["vessels"]),
            "wells": [0 for _well_id in replay_env.well_ids],
            "well_rates_tph": replay_env.automatic_well_rates_tph(),
        }
        augmented.append(with_wells)
        if t + 1 >= horizon_h:
            continue
        try:
            replay_env.step(action)
        except (RuntimeError, ValueError):
            break
    augmented.extend(
        {
            "vessels": [
                int(value)
                for value in raw_action.get("vessels", [])
            ],
            "wells": [0 for _well_id in env.well_ids],
            "well_rates_tph": [0.0 for _well_id in env.well_ids],
        }
        for raw_action in native_actions_by_hour[len(augmented):]
    )
    return augmented


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
    load_rate_groups=None,
    load_source_cap=None,
    load_source_limit_choice=None,
    load_capacity_limit_choice=None,
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
    replayed_unload_by_hour: dict[int, dict[str, float]] | None = None,
    limit_tie_audit: list[dict[str, object]] | None = None,
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
            _set_start_value(cargo_positive, (vessel_id, t), 1 if cargo_t >= 1e-3 else 0)
            _set_start_value(cargo_space, (vessel_id, t), 1 if cargo_t <= vessel.capacity_t - 1e-3 else 0)

        for emitter_id in env.emitter_ids:
            emitter = env.network.entities[emitter_id]
            capture_t = _capture_tonnes(env, scenario, emitter_id, start_step + t)
            pre_load = source_values[emitter_id] + capture_t
            ready = min(pre_load, emitter.buffer_capacity_t)
            overflow = 1 if pre_load > emitter.buffer_capacity_t + 1e-9 else 0
            _set_start_value(source_ready, (emitter_id, t), ready)
            _set_start_value(source_overflow_active, (emitter_id, t), overflow)
            _set_start_value(vent, (emitter_id, t), pre_load - ready)
            for rate_index, rate in enumerate((load_rate_groups or {}).get(emitter_id, ())):
                service_cap = min(float(rate), ready)
                _set_start_value(
                    load_source_cap,
                    (emitter_id, t, rate_index),
                    service_cap,
                )
                _set_start_value(
                    load_source_limit_choice,
                    (emitter_id, t, rate_index),
                    1 if ready <= float(rate) + 1e-9 else 0,
                )

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
                load_limits = [
                    min(
                        env.network.entities[vessel_id].loading_rate_tph,
                        emitter.loading_rate_tph,
                    ),
                    ready,
                    env.network.entities[vessel_id].capacity_t
                    - cargo_values[vessel_id],
                ]
                _seed_limit_choice(
                    load_limit_choice,
                    (vessel_id, emitter_id, t),
                    active,
                    amount,
                    load_limits,
                )
                if active and limit_tie_audit is not None:
                    limit_tie_audit.append(
                        _limit_tie_row(
                            "load",
                            t,
                            vessel_id,
                            emitter_id,
                            amount,
                            load_limits,
                        )
                    )
                if load_rate_groups:
                    rate = float(
                        min(
                            env.network.entities[vessel_id].loading_rate_tph,
                            emitter.loading_rate_tph,
                        )
                    )
                    service_cap = min(rate, ready)
                    remaining_capacity = (
                        env.network.entities[vessel_id].capacity_t
                        - cargo_values[vessel_id]
                    )
                    _set_start_value(
                        load_capacity_limit_choice,
                        (vessel_id, emitter_id, t),
                        1
                        if active and service_cap <= remaining_capacity + 1e-9
                        else 0,
                    )
            source_values[emitter_id] = ready - loaded_total
            _set_start_value(source_stock, (emitter_id, t + 1), source_values[emitter_id])

        if replayed_unload_by_hour is not None and t in replayed_unload_by_hour:
            unload_values.update(replayed_unload_by_hour[t])
        else:
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
            # A FIFO head remains the active unload service even when the
            # terminal has zero free space and the realized amount is zero.
            active = 1 if vessel_id in unload_values else 0
            _set_start_value(unload, (vessel_id, t), amount)
            _set_start_value(unload_active, (vessel_id, t), active)
            vessel = env.network.entities[vessel_id]
            unload_limits = [
                vessel.unloading_rate_tph,
                cargo_values[vessel_id],
                terminal_free,
            ]
            _seed_limit_choice(
                unload_limit_choice,
                (vessel_id, t),
                active,
                amount,
                unload_limits,
            )
            if active and limit_tie_audit is not None:
                limit_tie_audit.append(
                    _limit_tie_row(
                        "unload",
                        t,
                        vessel_id,
                        str(env._routes[vessel_id]["destination"]),
                        amount,
                        unload_limits,
                    )
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


def _limit_tie_row(
    service: str,
    hour: int,
    vessel_id: str,
    node_id: str,
    amount_t: float,
    limits_t: list[float],
) -> dict[str, object]:
    normalized = [max(0.0, float(limit)) for limit in limits_t]
    ordered = sorted(normalized)
    minimum = ordered[0]
    second_gap = ordered[1] - minimum
    return {
        "service": service,
        "hour": int(hour),
        "vessel_id": vessel_id,
        "node_id": node_id,
        "amount_t": float(amount_t),
        "limits_t": normalized,
        "minimum_t": minimum,
        "second_limit_gap_t": second_gap,
        "exact_tie": second_gap <= 1e-6,
        "near_tie_1t": second_gap <= 1.0,
        "near_tie_10t": second_gap <= 10.0,
    }


def _native_action_limit_tie_audit(
    env,
    native_actions_by_hour: list[dict[str, list[int]]],
    horizon_h: int,
) -> list[dict[str, object]]:
    scenario = env.scenario
    start_step = scenario.step_index(_current_start_hour(env))
    arcs, starts = _build_action_arcs(env, scenario, start_step, horizon_h)
    _, selected_wait_node = _selected_native_action_route(
        env,
        arcs,
        starts,
        native_actions_by_hour,
        horizon_h,
    )
    rows: list[dict[str, object]] = []
    _seed_native_action_state_values(
        env,
        scenario,
        start_step,
        selected_wait_node,
        _well_rate_options_by_hour(env, scenario, horizon_h),
        native_actions_by_hour,
        horizon_h,
        replayed_unload_by_hour=_replay_native_action_unloads(
            env,
            native_actions_by_hour,
            horizon_h,
        ),
        limit_tie_audit=rows,
    )
    return rows


def _warm_start_well_requests(env, well_rate_options, native_actions_by_hour, t: int) -> dict[str, float]:
    requests: dict[str, float] = {}
    if env.automatic_well_control:
        rates_tph = native_actions_by_hour[t].get(
            "well_rates_tph",
            [0.0 for _well_id in env.well_ids],
        )
        return {
            well_id: float(rate_tph)
            for well_id, rate_tph in zip(env.well_ids, rates_tph)
        }
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


def _replay_native_action_unloads(
    env,
    native_actions_by_hour: list[dict[str, list[int]]],
    horizon_h: int,
) -> dict[int, dict[str, float]]:
    replay_env = copy.deepcopy(env)
    unload_by_hour: dict[int, dict[str, float]] = {}
    for t in range(min(horizon_h, len(native_actions_by_hour))):
        raw = native_actions_by_hour[t]
        try:
            action = action_for_well_control_mode(replay_env, raw)
        except (KeyError, TypeError, ValueError):
            break
        if not _native_action_is_executable(replay_env, action):
            break
        active_heads: set[str] = set()
        state = replay_env.simulator.state
        for terminal in replay_env.network._entities_of_type(Terminal).values():
            if terminal_berth_count(state, terminal) <= 0:
                continue
            queue = terminal_unload_queue_snapshot(
                replay_env.network,
                terminal,
                state,
            )
            if queue:
                active_heads.add(queue[0])
        before = {
            vessel_id: float(replay_env.simulator.state.entity_inventory_t.get(vessel_id, 0.0))
            for vessel_id in replay_env.vessel_ids
        }
        replay_env.step(action)
        unload_by_hour[t] = {
            vessel_id: max(
                0.0,
                before[vessel_id]
                - float(
                    replay_env.simulator.state.entity_inventory_t.get(
                        vessel_id, 0.0
                    )
                ),
            )
            for vessel_id in active_heads
        }
    return unload_by_hour


def _native_action_is_executable(env, action: dict[str, list[int]]) -> bool:
    choices_and_masks = [
        (action["vessels"], env.vessel_action_mask()),
    ]
    if not getattr(env, "automatic_well_control", False):
        choices_and_masks.append(
            (action["wells"], env.well_rate_action_mask())
        )
    for choices, masks in choices_and_masks:
        if len(choices) != len(masks):
            return False
        if any(choice < 0 or choice >= len(mask) or not mask[choice] for choice, mask in zip(choices, masks)):
            return False
    return True


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
        well_request_tph_by_hour={},
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
    *,
    prune_unreachable: bool = False,
    weather_aware_cleanup_sailing_lower_bound: bool = False,
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
                    remaining_cleanup_fuel_h = (
                        0
                        if arrives_within_horizon
                        else (
                            _weather_remaining_fuel_hours_for_truncated_leg(
                                env,
                                scenario,
                                vessel_id,
                                origin_id=origin_id,
                                destination_id=destination_id,
                                start_step=start_step + t,
                                elapsed_h=horizon_h - t,
                            )
                            if weather_aware_cleanup_sailing_lower_bound
                            else _nominal_remaining_fuel_hours_after_boundary(
                                env,
                                scenario,
                                vessel_id,
                                origin_id=origin_id,
                                destination_id=destination_id,
                                start_step=start_step + t,
                                elapsed_h=horizon_h - t,
                            )
                        )
                    )
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
                            remaining_cleanup_fuel_h=remaining_cleanup_fuel_h,
                        )
                    )
    if prune_unreachable:
        arcs = _reachable_action_arcs(arcs, starts)
    return arcs, starts


def _reachable_action_arcs(
    arcs: list[_ActionArc],
    starts: dict[str, _PathStart],
) -> list[_ActionArc]:
    """Return exactly the arcs reachable from each vessel's boundary state."""

    outgoing: dict[tuple[str, int, str], list[int]] = {}
    for index, arc in enumerate(arcs):
        outgoing.setdefault(
            (arc.vessel_id, arc.start_h, arc.origin_id), []
        ).append(index)
    pending = [
        (vessel_id, start.start_h, start.node_id)
        for vessel_id, start in starts.items()
        if start.node_id is not None
    ]
    reached_states = set(pending)
    reached_arcs: set[int] = set()
    while pending:
        state = pending.pop()
        for index in outgoing.get(state, []):
            reached_arcs.add(index)
            arc = arcs[index]
            destination = (
                arc.vessel_id,
                arc.end_h,
                arc.destination_id,
            )
            if destination not in reached_states:
                reached_states.add(destination)
                pending.append(destination)
    return [
        arc for index, arc in enumerate(arcs) if index in reached_arcs
    ]


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


def _nominal_remaining_fuel_hours_after_boundary(
    env,
    scenario: Scenario,
    vessel_id: str,
    *,
    origin_id: str,
    destination_id: str,
    start_step: int,
    elapsed_h: int,
) -> int:
    """Nominal cleanup fuel needed to finish a voyage truncated at the boundary."""

    route = env._routes[vessel_id]
    distance_km = (
        float(route["distance_km"])
        if {origin_id, destination_id}
        == {str(route["origin"]), str(route["destination"])}
        else _dynamic_leg_distance_km(env, route, origin_id, destination_id)
    )
    speed_kmh = max(1e-9, float(route["speed_knots"]) * KNOTS_TO_KMH)
    covered_km = sum(
        speed_kmh
        * _scenario_leg_speed_factor(
            scenario,
            origin_id,
            destination_id,
            vessel_id,
            start_step + offset,
        )
        for offset in range(max(0, int(elapsed_h)))
    )
    remaining_km = max(0.0, distance_km - covered_km)
    return max(0, math.ceil(remaining_km / speed_kmh - 1e-9))


def _weather_remaining_fuel_hours_for_truncated_leg(
    env,
    scenario: Scenario,
    vessel_id: str,
    *,
    origin_id: str,
    destination_id: str,
    start_step: int,
    elapsed_h: int,
) -> int:
    route = env._routes[vessel_id]
    distance_km = (
        float(route["distance_km"])
        if {origin_id, destination_id}
        == {str(route["origin"]), str(route["destination"])}
        else _dynamic_leg_distance_km(env, route, origin_id, destination_id)
    )
    speed_kmh = max(1e-9, float(route["speed_knots"]) * KNOTS_TO_KMH)
    covered_km = sum(
        speed_kmh
        * _scenario_leg_speed_factor(
            scenario,
            origin_id,
            destination_id,
            vessel_id,
            start_step + offset,
        )
        for offset in range(max(0, int(elapsed_h)))
    )
    remaining_km = max(0.0, distance_km - covered_km)
    return _weather_remaining_fuel_hours_after_boundary(
        env,
        scenario,
        vessel_id,
        origin_id=origin_id,
        destination_id=destination_id,
        distance_km=remaining_km,
        boundary_step=start_step + max(0, int(elapsed_h)),
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


def _best_future_weather_leg_fuel_h(
    env,
    scenario: Scenario,
    vessel_id: str,
    *,
    origin_id: str,
    destination_id: str,
    distance_km: float,
    earliest_start_step: int,
) -> int:
    """Return a valid best-departure fuel-hour lower bound over full weather."""

    if distance_km <= 1e-9:
        return 0
    route = env._routes[vessel_id]
    speed_kmh = max(1e-9, float(route["speed_knots"]) * KNOTS_TO_KMH)
    nominal_duration_h = max(1, math.ceil(distance_km / speed_kmh))
    max_leg_h = max(500, 10 * nominal_duration_h + 10)
    first_step = max(0, int(earliest_start_step))
    last_step = max(first_step, int(getattr(scenario, "n_steps", 0)) - 1)
    best_duration_h = min(
        _sailing_duration_h_for_distance(
            env,
            scenario,
            vessel_id,
            origin_id=origin_id,
            destination_id=destination_id,
            distance_km=distance_km,
            start_step=start_step,
            max_horizon_h=max_leg_h,
        )
        for start_step in range(first_step, last_step + 1)
    )
    return max(0, best_duration_h - 1)


def _weather_remaining_fuel_hours_after_boundary(
    env,
    scenario: Scenario,
    vessel_id: str,
    *,
    origin_id: str,
    destination_id: str,
    distance_km: float,
    boundary_step: int,
) -> int:
    """Fuel hours to finish an already-sailing leg from its known boundary."""

    if distance_km <= 1e-9:
        return 0
    route = env._routes[vessel_id]
    speed_kmh = max(1e-9, float(route["speed_knots"]) * KNOTS_TO_KMH)
    nominal_duration_h = max(1, math.ceil(distance_km / speed_kmh))
    return _sailing_duration_h_for_distance(
        env,
        scenario,
        vessel_id,
        origin_id=origin_id,
        destination_id=destination_id,
        distance_km=distance_km,
        start_step=max(0, int(boundary_step)),
        max_horizon_h=max(500, 10 * nominal_duration_h + 10),
    )


def _dynamic_leg_distance_km(env, route: dict, origin_id: str, destination_id: str) -> float:
    leg_routes = route.setdefault("dynamic_leg_routes", {})
    leg_id = f"{origin_id}->{destination_id}"
    if leg_id not in leg_routes:
        origin = env.locations[origin_id]
        destination = env.locations[destination_id]
        maritime_route = sea_route(origin, destination)
        coordinates = list(maritime_route.coordinates)
        if not coordinates:
            coordinates = [origin, destination]
        else:
            if coordinates[0] != origin:
                coordinates.insert(0, origin)
            if coordinates[-1] != destination:
                coordinates.append(destination)
        leg_routes[leg_id] = {
            "id": leg_id,
            "origin": origin_id,
            "destination": destination_id,
            "provider": maritime_route.provider,
            "distance_km": round(route_distance_km(coordinates), 2),
            "coordinates": coordinates,
        }
    return float(leg_routes[leg_id]["distance_km"])


def _action_to_destination(env, vessel_id: str, destination_id: str) -> int:
    if destination_id == str(env._routes[vessel_id]["destination"]):
        return VESSEL_GO_TERMINAL
    return env.vessel_go_emitter_action(destination_id)


def _wait_expr(arc_vars, wait_arc: dict[tuple[str, str, int], int], vessel_id: str, node_id: str, t: int):
    index = wait_arc.get((vessel_id, node_id, t))
    return 0 if index is None else arc_vars[index]


def _well_rate_options_by_hour(env, scenario: Scenario, horizon_h: int) -> dict[tuple[str, int], list[int]]:
    if env.automatic_well_control:
        return {
            (well_id, t): [0]
            for well_id in env.well_ids
            for t in range(horizon_h)
        }
    start_step = scenario.step_index(_current_start_hour(env))
    options: dict[tuple[str, int], list[int]] = {}
    for t in range(horizon_h):
        future_state = _future_state_for_step(env, scenario, start_step + t)
        interval_start_h = future_state.time_h
        evaluation_time_h = interval_start_h + env.network.time_step_hours
        for well_id in env.well_ids:
            physical_max = _physical_well_max_tph(env, future_state, well_id)
            if _uses_single_well_dynamic_bhp(env, well_id):
                well = env.network.entities[well_id]
                assert isinstance(well, InjectionWell)
                mask = tuple(
                    mtpa_to_tph(rate_mtpa) <= physical_max + 1e-9
                    and (
                        rate_mtpa <= 1e-12
                        or mtpa_to_tph(rate_mtpa)
                        >= well.min_stable_injection_tph - 1e-9
                    )
                    for rate_mtpa in WELL_RATE_LEVELS_MTPA
                )
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


def _physical_well_max_by_hour(
    env,
    scenario: Scenario,
    horizon_h: int,
) -> dict[tuple[str, int], float]:
    start_step = scenario.step_index(_current_start_hour(env))
    return {
        (well_id, t): _physical_well_max_tph(
            env,
            _future_state_for_step(env, scenario, start_step + t),
            well_id,
        )
        for well_id in env.well_ids
        for t in range(horizon_h)
    }


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


def _add_continuous_automatic_well_request_constraints(
    prob,
    env,
    scenario,
    well_request,
    well_regime,
    well_inj,
    physical_max_by_hour,
    horizon_h: int,
) -> None:
    """Make the MILP reproduce continuous maximum-feasible well control."""
    if not getattr(env, "automatic_well_control", False):
        return

    state = env.simulator.state
    start_step = scenario.step_index(_current_start_hour(env))
    start_h = _current_start_hour(env)
    dt = float(env.network.time_step_hours)
    for well_id in env.well_ids:
        if not _uses_single_well_dynamic_bhp(env, well_id):
            for t in range(horizon_h):
                future_state = _future_state_for_step(
                    env,
                    scenario,
                    start_step + t,
                )
                interval_start_h = future_state.time_h
                maximum_rate_tph = maximum_feasible_well_rate_tph(
                    env.network,
                    future_state,
                    well_id,
                    physical_max_by_hour[(well_id, t)],
                    evaluation_time_h=(
                        interval_start_h
                        + env.network.time_step_hours
                    ),
                    interval_start_h=interval_start_h,
                )
                prob += well_request[(well_id, t)] == maximum_rate_tph
            continue

        reservoir_id = env.network._single_downstream_of_type(
            well_id,
            Reservoir,
        )
        reservoir = env.network.entities[reservoir_id]
        assert isinstance(reservoir, Reservoir)
        alpha = _reservoir_pressure_bar_per_tonne(reservoir)
        initial_inventory_t = float(
            state.entity_inventory_t.get(reservoir_id, 0.0)
        )
        reservoir_pressure_const = (
            reservoir.initial_pressure_bar + alpha * initial_inventory_t
        )
        limit_bar = float(reservoir.well_bottomhole_pressure_limit_bar)
        well = env.network.entities[well_id]
        assert isinstance(well, InjectionWell)

        for t in range(horizon_h):
            physical_max_tph = physical_max_by_hour[(well_id, t)]
            request = well_request[(well_id, t)]
            off = well_regime[(well_id, t, "off")]
            physical = well_regime[(well_id, t, "physical")]
            pressure = well_regime[(well_id, t, "pressure")]
            prob += off + physical + pressure == 1

            if (
                physical_max_tph <= 1e-12
                or physical_max_tph
                < well.min_stable_injection_tph - 1e-9
            ):
                prob += request == 0
                prob += off == 1
                continue

            evaluation_h = start_h + (t + 1) * dt
            response_const, response_coeffs = (
                _single_well_line_source_response_terms(
                    env,
                    well_id,
                    horizon_index=t,
                    evaluation_h=evaluation_h,
                )
            )
            pressure_coeffs = [
                alpha * dt + response_coeffs[tau]
                for tau in range(t + 1)
            ]
            pressure_before_current = (
                reservoir_pressure_const
                + response_const
                + pulp.lpSum(
                    pressure_coeffs[tau] * well_inj[(well_id, tau)]
                    for tau in range(t)
                )
            )
            max_pressure_span = abs(
                reservoir_pressure_const + response_const - limit_bar
            )
            for tau in range(t + 1):
                max_pressure_span += (
                    physical_max_by_hour[(well_id, tau)]
                    * abs(pressure_coeffs[tau])
                )
            pressure_big_m = max(1.0, max_pressure_span + 1.0)
            current_pressure_coefficient = pressure_coeffs[t]
            pressure_at_request = (
                pressure_before_current
                + current_pressure_coefficient * request
            )
            pressure_at_physical_max = (
                pressure_before_current
                + current_pressure_coefficient * physical_max_tph
            )

            prob += request <= physical_max_tph * (1 - off)
            if well.min_stable_injection_tph > 1e-12:
                prob += request >= (
                    well.min_stable_injection_tph * (1 - off)
                )
            off_threshold_rate = max(
                0.0,
                float(well.min_stable_injection_tph),
            )
            off_threshold_pressure = (
                pressure_before_current
                + current_pressure_coefficient * off_threshold_rate
            )
            off_margin = (
                1e-7
                if well.min_stable_injection_tph > 1e-12
                else 0.0
            )
            prob += off_threshold_pressure >= (
                limit_bar
                + off_margin
                - pressure_big_m * (1 - off)
            )

            prob += request >= (
                physical_max_tph
                - physical_max_tph * (1 - physical)
            )
            prob += pressure_at_physical_max <= (
                limit_bar + pressure_big_m * (1 - physical)
            )

            if current_pressure_coefficient <= 1e-12:
                prob += pressure == 0
            else:
                prob += pressure_at_request <= (
                    limit_bar + pressure_big_m * (1 - pressure)
                )
                prob += pressure_at_request >= (
                    limit_bar - pressure_big_m * (1 - pressure)
                )


def _add_single_well_dynamic_bhp_constraints(
    prob,
    env,
    well_inj,
    horizon_h: int,
    *,
    well_choice=None,
    well_rate_options=None,
    automatic_well_regime=None,
    automatic_well_physical_max=None,
) -> None:
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
            pressure_expr = (
                reservoir_pressure_const
                + alpha * cumulative_injected_expr
                + line_source_expr
            )
            automatic_off = (
                automatic_well_regime.get((well_id, t, "off"))
                if automatic_well_regime
                else None
            )
            if (
                automatic_off is None
                and (well_choice is None or well_rate_options is None)
            ):
                prob += pressure_expr <= limit_bar
                continue

            maximum_pressure_span = abs(
                reservoir_pressure_const + response_const - limit_bar
            )
            for tau in range(t + 1):
                if automatic_well_regime:
                    maximum_rate_tph = float(
                        automatic_well_physical_max[
                            (well_id, tau)
                        ]
                    )
                else:
                    maximum_rate_tph = max(
                        (
                            mtpa_to_tph(WELL_RATE_LEVELS_MTPA[index])
                            for index in well_rate_options[(well_id, tau)]
                        ),
                        default=0.0,
                    )
                maximum_pressure_span += maximum_rate_tph * abs(
                    alpha * dt + response_coeffs[tau]
                )
            pressure_big_m = max(1.0, maximum_pressure_span + 1.0)
            off_indicator = (
                automatic_off
                if automatic_off is not None
                else well_choice[(well_id, t, 0)]
            )
            prob += pressure_expr <= (
                limit_bar
                + pressure_big_m * off_indicator
            )


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
            availability = _scenario_series_value(
                scenario.emitter_availability,
                emitter_id,
                start_step + t,
                emitter.availability,
            )
            capture_tph = emitter.nominal_capture_tph * max(0.0, float(availability))
            prob += risk[(emitter_id, t)] >= (
                capture_tph * lookahead_h
                - emitter.buffer_capacity_t
                + source_stock[(emitter_id, t + 1)]
            )
    return risk


def _overflow_risk_value(
    env,
    scenario: Scenario,
    start_step: int,
    horizon_h: int,
    source_stock,
    lookahead_h: float,
) -> float:
    risk_t = 0.0
    for emitter_id in env.emitter_ids:
        emitter = env.network.entities[emitter_id]
        for t in range(horizon_h):
            availability = _scenario_series_value(
                scenario.emitter_availability,
                emitter_id,
                start_step + t,
                emitter.availability,
            )
            capture_tph = emitter.nominal_capture_tph * max(0.0, float(availability))
            risk_t += max(
                0.0,
                capture_tph * lookahead_h
                - emitter.buffer_capacity_t
                + _value(source_stock[(emitter_id, t + 1)]),
            )
    return risk_t


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


def _initial_terminal_queue_ages(env) -> dict[str, int]:
    ages: dict[str, int] = {}
    state = env.simulator.state
    for terminal_id, terminal in env.network._entities_of_type(Terminal).items():
        queue = terminal_unload_queue_snapshot(env.network, terminal, state)
        for position, vessel_id in enumerate(queue):
            ages[vessel_id] = len(queue) - position
    return ages


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


def _initial_sailing_fuel_hours(starts: dict[str, _PathStart], horizon_h: int) -> int:
    return sum(
        horizon_h if start.node_id is None else max(0, start.start_h - 1)
        for start in starts.values()
        if start.start_h > 0
    )


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


def _native_cost_breakdown(
    env,
    arcs,
    arc_vars,
    load,
    unload,
    stored_t: float,
    params: EconomicParameters,
    *,
    initial_sailing_fuel_hours: int = 0,
) -> CplexCostBreakdown:
    vessel_fuel = initial_sailing_fuel_hours * params.vessel_fuel_eur_per_h_sailing + sum(
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
