"""``CCSEnv`` - a reinforcement-learning environment over the CCS physics.

This is the layer that turns the deterministic physical twin into something an
RL agent can train against. Each episode :meth:`reset` samples a
:class:`~sim.scenario_generation.Scenario` (the exogenous disturbances), and each
:meth:`step` maps a hybrid control action into physical action proposals, runs
one hour of physics through the :class:`~sim.simulator.PhysicalSimulator`,
applies the scenario disturbances, and prices the outcome with the
:class:`~sim.economics.CostModel` to produce the reward.

The interface is gym-style (``reset`` / ``step`` returning
``(obs, reward, done, info)``) but intentionally has **no numpy or gymnasium
dependency** so it stays importable anywhere. Observations are flat ``list[float]``
and the native action is a dictionary with discrete vessel choices plus
continuous well rates in Mt/y.

Controls (section 7.2 of the research note):
- per vessel: ``WAIT`` / ``GO_TERMINAL`` / ``GO_EMITTER[id]``;
- per well: continuous injection rate, bounded to 0.5-2.0 Mt/y when available
  and forced to 0 while the well is under maintenance.

Loading at any emitter berth and unloading at the terminal are issued
automatically (they are never the interesting decision); the agent chooses which
emitter or terminal to send vessels to and how hard to inject. A vessel action
mask exposes which destination choices are physically legal, while
``well_rate_bounds()`` exposes the continuous injection bounds.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..actions import ActionFrame, ActionProposal
from ..economics import CostModel, EconomicLedger
from ..entities.emitter import Emitter
from ..entities.manifold import SubseaManifold
from ..entities.pipeline import Pipeline
from ..entities.state import PhysicalState
from ..entities.storage import InjectionWell, Reservoir
from ..entities.terminal import Terminal
from ..entities.vessel import Vessel
from ..routes import route_distance_km, sea_route
from ..scenario_generation import Scenario, ScenarioGenerator
from ..simulator import PhysicalSimulator

# Vessel action ids. Emitter actions are dynamic:
# VESSEL_GO_EMITTER_BASE + env.emitter_ids.index(emitter_id).
VESSEL_WAIT, VESSEL_GO_TERMINAL = 0, 1
VESSEL_GO_EMITTER_BASE = 2
VESSEL_ACTIONS = VESSEL_GO_EMITTER_BASE

MIN_WELL_RATE_MTPA = 0.5
MAX_WELL_RATE_MTPA = 2.5
WELL_RATE_BOUNDS_MTPA = (MIN_WELL_RATE_MTPA, MAX_WELL_RATE_MTPA)
_MTPA_TO_TPH = 1_000_000.0 / (365.25 * 24.0)

Coordinate = tuple[float, float]
CCSAction = dict[str, list[int] | list[float]]


@dataclass
class CCSEnvConfig:
    episode_hours: int = 168
    storage_target_rate: float = 0.9
    reward_scale: float = 1e-3
    default_speed_knots: float = 12.0


class CCSEnv:
    """Single-agent (centralized) RL environment over the CCS network."""

    def __init__(
        self,
        network,
        locations: dict[str, Coordinate],
        scenario_generator: ScenarioGenerator | None = None,
        cost_model: CostModel | None = None,
        config: CCSEnvConfig | None = None,
        *,
        routes: dict[str, dict] | None = None,
    ) -> None:
        self.network = network
        self.config = config or CCSEnvConfig()
        self.scenario_generator = scenario_generator or ScenarioGenerator()
        self.cost_model = cost_model or CostModel()
        self.locations = locations
        self._routes = routes or self._build_routes(locations)

        self.emitter_ids = sorted(network._entities_of_type(Emitter))
        self.vessel_ids = sorted(network._entities_of_type(Vessel))
        self.terminal_ids = sorted(network._entities_of_type(Terminal))
        self.well_ids = sorted(network._entities_of_type(InjectionWell))
        self.reservoir_ids = sorted(network._entities_of_type(Reservoir))

        # Total capacity that can hold captured-but-not-yet-stored CO2, used to
        # normalise the in-transit observation into a horizon-invariant [0, 1].
        self._in_transit_capacity_t = (
            sum(network.entities[e].buffer_capacity_t for e in self.emitter_ids)
            + sum(network.entities[v].capacity_t for v in self.vessel_ids)
            + sum(network.entities[t].storage_capacity_t for t in self.terminal_ids)
        )

        self.n_steps = max(1, int(round(self.config.episode_hours / network.time_step_hours)))

        # Episode state, populated by reset().
        self.scenario: Scenario | None = None
        self.simulator: PhysicalSimulator | None = None
        self.t = 0
        self.ledger = EconomicLedger()
        self.cumulative_captured_t = 0.0
        self.cumulative_stored_t = 0.0
        self.last_info: dict = {}
        self._prev_shortfall_penalty = 0.0

    # -- spaces -----------------------------------------------------------
    @property
    def action_dims(self) -> list[int]:
        """Discrete dimensions for vessel decisions.

        The well controls are continuous and exposed through
        :meth:`well_rate_bounds`, so this compatibility alias intentionally only
        covers vessel dimensions.
        """
        return self.vessel_action_dims

    @property
    def vessel_action_dims(self) -> list[int]:
        return [self.vessel_action_count] * len(self.vessel_ids)

    def well_rate_bounds(self) -> list[tuple[float, float]]:
        return [self._well_rate_bound(wid) for wid in self.well_ids]

    def action_spec(self) -> dict[str, object]:
        return {
            "vessel_action_dims": self.vessel_action_dims,
            "well_rate_bounds": self.well_rate_bounds(),
        }

    @property
    def vessel_action_count(self) -> int:
        return VESSEL_GO_EMITTER_BASE + len(self.emitter_ids)

    def vessel_go_emitter_action(self, emitter_id: str) -> int:
        if emitter_id not in self.emitter_ids:
            raise ValueError(f"Unknown emitter: {emitter_id}")
        return VESSEL_GO_EMITTER_BASE + self.emitter_ids.index(emitter_id)

    @property
    def observation_size(self) -> int:
        return len(self.feature_names)

    @property
    def feature_names(self) -> list[str]:
        # Horizon-invariant globals only: a weekly clock (weather/ops cycle) and
        # the instantaneous in-transit inventory fill. No episode-relative features, so a
        # policy trained on short episodes transfers to a long evaluation rollout.
        names = ["hour_of_week", "in_transit_fill"]
        for eid in self.emitter_ids:
            names += [f"{eid}.fill", f"{eid}.capture_norm", f"{eid}.availability"]
        for vid in self.vessel_ids:
            names += [f"{vid}.cargo", f"{vid}.berthed", f"{vid}.at_terminal", f"{vid}.progress"]
            names += [f"{vid}.at_{eid}" for eid in self.emitter_ids]
        for tid in self.terminal_ids:
            names += [f"{tid}.fill", f"{tid}.berth_frac"]
        for wid in self.well_ids:
            names += [f"{wid}.inject_norm", f"{wid}.injectivity", f"{wid}.available"]
        for rid in self.reservoir_ids:
            names += [f"{rid}.pressure_margin"]
        return names

    # -- episode lifecycle ------------------------------------------------
    def reset(self, seed: int | None = None) -> list[float]:
        self.scenario = self.scenario_generator.sample(self.network, seed=seed)
        state = PhysicalState()
        self.scenario.apply_initial(state)
        self.simulator = PhysicalSimulator(
            self.network, state, routes=self._routes, locations=self.locations
        )
        self.t = 0
        self.ledger = EconomicLedger()
        self.cumulative_captured_t = 0.0
        self.cumulative_stored_t = 0.0
        self.initial_in_transit_t = self._in_transit_inventory()
        self._prev_shortfall_penalty = 0.0
        self._apply_disturbances()
        self.last_info = self._action_info()
        return self._observation()

    def step(self, action: CCSAction) -> tuple[list[float], float, bool, bool, dict]:
        if self.simulator is None or self.scenario is None:
            raise RuntimeError("Call reset() before step().")
        normalized_action = self._normalize_action(action)

        hours = self.network.time_step_hours
        current_time = self.simulator.state.time_h
        proposals = self._build_proposals(normalized_action)
        record = self.simulator.step(
            ActionFrame(time_h=current_time, proposals=proposals), compute_observation=False
        )
        step_result = record.step_result

        economics = self.cost_model.evaluate_step(self.network, step_result)
        # Gross captured = CO2 that entered the buffer plus CO2 the plant captured
        # but had to vent for lack of buffer/logistics. Venting therefore lowers
        # the storage rate, matching the section 8 definition (stored / captured).
        captured_step = (
            sum(step_result.state.last_capture_tph.values())
            + sum(step_result.state.last_vent_tph.values())
        ) * hours
        self.cumulative_captured_t += captured_step
        self.cumulative_stored_t += economics.stored_t
        self.ledger.add(economics)
        shortfall_penalty = self.cost_model.storage_shortfall_penalty(
            self.cumulative_captured_t,
            self.cumulative_stored_t,
            self.config.storage_target_rate,
        )
        shortfall_delta_penalty = shortfall_penalty - self._prev_shortfall_penalty
        self._prev_shortfall_penalty = shortfall_penalty
        self.ledger.storage_shortfall_penalty += shortfall_delta_penalty

        in_transit_now = self._in_transit_inventory()
        in_transit_growth = in_transit_now - self.initial_in_transit_t
        reward = (economics.net - shortfall_delta_penalty) * self.config.reward_scale

        self.t += 1
        # The operational task is fixed-horizon: there is no early terminal
        # condition, so the episode only ends through the time-limit truncation.
        terminated = False
        truncated = self.t >= self.n_steps
        if not (terminated or truncated):
            self._apply_disturbances()

        info = {
            "time_h": step_result.state.time_h,
            "economics": economics.as_dict(),
            "in_transit_t": in_transit_now,
            "in_transit_growth_t": in_transit_growth,
            "shortfall_penalty": shortfall_penalty,
            "shortfall_delta_penalty": shortfall_delta_penalty,
            "storage_rate": self.storage_rate(),
            "loss_rate": self.loss_rate(),
            "violations": [v.violation_type for v in step_result.violations],
            **self._action_info(),
        }
        self.last_info = info
        return self._observation(), reward, terminated, truncated, info

    def storage_rate(self) -> float:
        """Stored / gross-captured. Only meaningful over a long horizon, where
        in-transit CO2 is negligible relative to total captured."""
        if self.cumulative_captured_t <= 0.0:
            return 1.0
        return self.cumulative_stored_t / self.cumulative_captured_t

    def loss_rate(self) -> float:
        """Vented / gross-captured: the share of captured CO2 truly lost. This is
        the short-horizon truth - it ignores recoverable in-transit inventory."""
        if self.cumulative_captured_t <= 0.0:
            return 0.0
        return self.ledger.vented_t / self.cumulative_captured_t

    def _in_transit_inventory(self) -> float:
        """In-transit CO2: captured but not yet stored (everything but reservoirs)."""
        reservoirs = set(self.reservoir_ids)
        return sum(
            inventory
            for entity_id, inventory in self.simulator.state.entity_inventory_t.items()
            if entity_id not in reservoirs
        )

    # -- action mask ------------------------------------------------------
    def _action_info(self) -> dict:
        vessel_mask = self.vessel_action_mask()
        bounds = self.well_rate_bounds()
        return {
            "action_mask": vessel_mask,
            "vessel_action_mask": vessel_mask,
            "well_rate_bounds": bounds,
        }

    def action_mask(self) -> list[list[bool]]:
        return self.vessel_action_mask()

    def vessel_action_mask(self) -> list[list[bool]]:
        mask: list[list[bool]] = []
        for vid in self.vessel_ids:
            mask.append(self._vessel_mask(vid))
        return mask

    def _vessel_mask(self, vessel_id: str) -> list[bool]:
        vstate = self.simulator.vessel_states[vessel_id]
        route = self._routes[vessel_id]
        if vstate["mode"] != "berthed":
            return [True] + [False] * (self.vessel_action_count - 1)  # mid-voyage: can only WAIT
        berth = vstate["berth"]
        at_terminal = berth == route["destination"]
        mask = [True, not at_terminal]
        mask.extend(berth != emitter_id for emitter_id in self.emitter_ids)
        return mask

    def _well_rate_bound(self, well_id: str) -> tuple[float, float]:
        if self.simulator is None:
            return WELL_RATE_BOUNDS_MTPA
        available = self.simulator.state.well_available.get(well_id, True)
        if not available:
            return (0.0, 0.0)
        return WELL_RATE_BOUNDS_MTPA

    # -- action translation ----------------------------------------------
    def _normalize_action(self, action: CCSAction) -> dict[str, list]:
        if not isinstance(action, dict):
            raise ValueError("Expected action dict with 'vessels' and 'wells' entries.")
        if "vessels" not in action or "wells" not in action:
            raise ValueError("Expected action dict with 'vessels' and 'wells' entries.")

        vessel_actions = list(action["vessels"])
        well_rates = list(action["wells"])
        if len(vessel_actions) != len(self.vessel_ids):
            raise ValueError(f"Expected {len(self.vessel_ids)} vessel actions, got {len(vessel_actions)}.")
        if len(well_rates) != len(self.well_ids):
            raise ValueError(f"Expected {len(self.well_ids)} well rates, got {len(well_rates)}.")

        return {
            "vessels": [int(choice) for choice in vessel_actions],
            "wells": [float(rate) for rate in well_rates],
        }

    def _build_proposals(self, action: dict[str, list]) -> list[ActionProposal]:
        vessel_actions = action["vessels"]
        well_rates = action["wells"]
        proposals: list[ActionProposal] = []

        # Always capture at full rate (capture is not an RL control here).
        for emitter_id in self.emitter_ids:
            proposals.append(self._proposal(emitter_id, "set_capture_utilization", {"utilization": 1.0}))

        departing = self._vessel_dispatch_proposals(vessel_actions, proposals)
        self._auto_loading_proposals(proposals, departing)
        self._auto_unloading_proposals(proposals, departing)
        self._injection_proposals(well_rates, proposals)
        return proposals

    def _vessel_dispatch_proposals(self, vessel_actions, proposals) -> set[str]:
        departing: set[str] = set()
        for vessel_id, choice in zip(self.vessel_ids, vessel_actions):
            vstate = self.simulator.vessel_states[vessel_id]
            if vstate["mode"] != "berthed":
                continue
            berth = vstate["berth"]
            destination = self._vessel_action_destination(vessel_id, choice)
            if destination == berth:
                destination = None
            if destination is not None:
                proposals.append(self._proposal(vessel_id, "sail_to", {"destination_id": destination}))
                departing.add(vessel_id)

        return departing

    def _vessel_action_destination(self, vessel_id: str, choice: int) -> str | None:
        if choice == VESSEL_WAIT:
            return None
        if choice == VESSEL_GO_TERMINAL:
            return str(self._routes[vessel_id]["destination"])
        emitter_index = choice - VESSEL_GO_EMITTER_BASE
        if 0 <= emitter_index < len(self.emitter_ids):
            return self.emitter_ids[emitter_index]
        return None

    def _auto_loading_proposals(self, proposals, departing) -> None:
        loaded_emitters: set[str] = set()
        for vessel_id in self.vessel_ids:
            if vessel_id in departing:
                continue
            vstate = self.simulator.vessel_states[vessel_id]
            emitter_id = vstate["berth"]
            if vstate["mode"] != "berthed" or emitter_id not in self.emitter_ids or emitter_id in loaded_emitters:
                continue
            vessel = self.network.entities[vessel_id]
            cargo = self.simulator.state.entity_inventory_t.get(vessel_id, 0.0)
            if cargo < vessel.capacity_t - 1e-9:
                proposals.append(self._proposal(emitter_id, "load_vessel", {"vessel_id": vessel_id}))
                loaded_emitters.add(emitter_id)

    def _auto_unloading_proposals(self, proposals, departing) -> None:
        for terminal_id in self.terminal_ids:
            head = self._terminal_unload_head(terminal_id, departing)
            if head is not None:
                proposals.append(self._proposal(terminal_id, "unload_vessel", {"vessel_id": head}))

    def _terminal_unload_head(self, terminal_id: str, departing) -> str | None:
        candidates = []
        for vessel_id in self.vessel_ids:
            if vessel_id in departing:
                continue
            if self.simulator.state.vessel_berths.get(vessel_id) != terminal_id:
                continue
            if self.simulator.state.entity_inventory_t.get(vessel_id, 0.0) > 1e-9:
                candidates.append(vessel_id)
        return sorted(candidates)[0] if candidates else None

    def _injection_proposals(self, well_rates, proposals) -> None:
        desired: dict[str, float] = {}
        for well_id, requested_rate_mtpa in zip(self.well_ids, well_rates):
            lower, upper = self._well_rate_bound(well_id)
            if upper <= 0.0:
                desired[well_id] = 0.0
                continue
            rate_mtpa = min(max(float(requested_rate_mtpa), lower), upper)
            desired[well_id] = rate_mtpa * _MTPA_TO_TPH

        for pipeline_id in self.network._entities_of_type(Pipeline):
            wells = self._pipeline_wells(pipeline_id)
            total = sum(desired.get(w, 0.0) for w in wells)
            proposals.append(self._proposal(pipeline_id, "set_flow", {"flow_tph": total}))
            self._manifold_split_proposals(pipeline_id, desired, proposals)

    def _manifold_split_proposals(self, pipeline_id, desired, proposals) -> None:
        for manifold_id in self.network._downstream_of_type(pipeline_id, SubseaManifold):
            wells = self.network._downstream_of_type(manifold_id, InjectionWell)
            total = sum(desired.get(w, 0.0) for w in wells)
            if total <= 1e-9:
                continue  # all OFF: no split needed, pipeline flow already excludes them
            splits = {w: desired.get(w, 0.0) / total for w in wells}
            proposals.append(self._proposal(manifold_id, "set_well_split", {"well_splits": splits}))

    def _pipeline_wells(self, pipeline_id: str) -> list[str]:
        wells = list(self.network._downstream_of_type(pipeline_id, InjectionWell))
        for manifold_id in self.network._downstream_of_type(pipeline_id, SubseaManifold):
            wells += self.network._downstream_of_type(manifold_id, InjectionWell)
        return wells

    # -- observation ------------------------------------------------------
    def _observation(self) -> list[float]:
        state = self.simulator.state
        hour_of_week = (state.time_h % 168.0) / 168.0
        in_transit_fill = _safe_div(self._in_transit_inventory(), self._in_transit_capacity_t)
        obs: list[float] = [hour_of_week, in_transit_fill]
        for eid in self.emitter_ids:
            emitter = self.network.entities[eid]
            inv = state.entity_inventory_t.get(eid, 0.0)
            obs += [
                _safe_div(inv, emitter.buffer_capacity_t),
                _safe_div(state.last_capture_tph.get(eid, 0.0), emitter.nominal_capture_tph),
                state.emitter_availability.get(eid, emitter.availability),
            ]
        for vid in self.vessel_ids:
            vessel = self.network.entities[vid]
            vstate = self.simulator.vessel_states[vid]
            route = self._routes[vid]
            berthed = vstate["mode"] == "berthed"
            berth = vstate["berth"] if berthed else None
            obs += [
                _safe_div(state.entity_inventory_t.get(vid, 0.0), vessel.capacity_t),
                1.0 if berthed else 0.0,
                1.0 if berthed and berth == route["destination"] else 0.0,
                float(vstate["progress"]),
            ]
            obs += [1.0 if berthed and berth == emitter_id else 0.0 for emitter_id in self.emitter_ids]
        for tid in self.terminal_ids:
            terminal = self.network.entities[tid]
            berth_override = state.berth_count_override.get(tid, terminal.berth_count)
            obs += [
                _safe_div(state.entity_inventory_t.get(tid, 0.0), terminal.storage_capacity_t),
                _safe_div(berth_override, max(1, terminal.berth_count)),
            ]
        for wid in self.well_ids:
            well = self.network.entities[wid]
            obs += [
                _safe_div(state.last_injection_flow_tph.get(wid, 0.0), well.max_injection_tph),
                state.injectivity_factor.get(wid, 1.0),
                1.0 if state.well_available.get(wid, True) else 0.0,
            ]
        for rid in self.reservoir_ids:
            reservoir = self.network.entities[rid]
            inv = state.entity_inventory_t.get(rid, 0.0)
            span = reservoir.max_pressure_bar - reservoir.initial_pressure_bar
            obs += [_safe_div(reservoir.pressure_margin_bar(inv), span) if span > 0 else 1.0]
        return obs

    # -- helpers ----------------------------------------------------------
    def _apply_disturbances(self) -> None:
        self.scenario.apply_to_state(self.simulator.state, self.simulator.state.time_h)

    def _proposal(self, entity_id: str, verb: str, params: dict) -> ActionProposal:
        return ActionProposal(agent_id="ccs_env", entity_id=entity_id, verb=verb, params=params)

    def _build_routes(self, locations: dict[str, Coordinate]) -> dict[str, dict]:
        routes: dict[str, dict] = {}
        for vessel_id in sorted(self.network._entities_of_type(Vessel)):
            origin_id = self._upstream_id(vessel_id, Emitter)
            destination_id = self._downstream_id(vessel_id, Terminal)
            if origin_id is None or destination_id is None:
                raise ValueError(
                    f"Vessel {vessel_id} needs an upstream emitter and downstream terminal to build a route."
                )
            for location_id in (origin_id, destination_id):
                if location_id not in locations:
                    raise ValueError(f"Missing location coordinate for {location_id}.")
            route = sea_route(locations[origin_id], locations[destination_id])
            vessel = self.network.entities[vessel_id]
            routes[vessel_id] = {
                "origin": origin_id,
                "destination": destination_id,
                "distance_km": route.distance_km,
                "speed_knots": vessel.speed_knots or self.config.default_speed_knots,
                "coordinates": route.coordinates,
                "return_coordinates": list(reversed(route.coordinates)),
            }
        return routes

    def _upstream_id(self, entity_id: str, entity_type: type) -> str | None:
        matches = self.network._upstream_of_type(entity_id, entity_type)
        return matches[0] if matches else None

    def _downstream_id(self, entity_id: str, entity_type: type) -> str | None:
        matches = self.network._downstream_of_type(entity_id, entity_type)
        return matches[0] if matches else None


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0
