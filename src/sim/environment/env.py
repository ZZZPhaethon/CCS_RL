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
discrete well rate-level indices.

Controls (section 7.2 of the research note):
- per vessel: ``WAIT`` / ``GO_TERMINAL`` / ``GO_EMITTER[id]``;
- per well: a discrete injection-rate level index over
  ``(0.0, 0.5, 1.0, 1.5, 2.0, 2.5)`` Mt/y.

Loading at any emitter berth and unloading at the terminal are issued
automatically (they are never the interesting decision); the agent chooses which
emitter or terminal to send vessels to and how hard to inject. A vessel action
mask exposes which destination choices are physically legal, while
``well_rate_action_mask()`` exposes the currently feasible well rate levels.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

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
from ..scenario_generation.disturbance_resolver import well_max_injection_tph
from ..simulator import PhysicalSimulator
from ..operations.pressure_limits import (
    WELL_RATE_LEVELS_MTPA,
    mtpa_to_tph,
    pressure_limited_rate_level_mask,
)
from ..operations.unloading import sync_terminal_unload_queue

# Vessel action ids. Emitter actions are dynamic:
# VESSEL_GO_EMITTER_BASE + env.emitter_ids.index(emitter_id).
VESSEL_WAIT, VESSEL_GO_TERMINAL = 0, 1
VESSEL_GO_EMITTER_BASE = 2
VESSEL_ACTIONS = VESSEL_GO_EMITTER_BASE

OFF_WELL_RATE_INDEX = 0
MIN_WELL_RATE_INDEX = 1
MAX_WELL_RATE_INDEX = len(WELL_RATE_LEVELS_MTPA) - 1
MIN_WELL_RATE_MTPA = WELL_RATE_LEVELS_MTPA[MIN_WELL_RATE_INDEX]
MAX_WELL_RATE_MTPA = WELL_RATE_LEVELS_MTPA[MAX_WELL_RATE_INDEX]
WELL_RATE_BOUNDS_MTPA = (MIN_WELL_RATE_MTPA, MAX_WELL_RATE_MTPA)

Coordinate = tuple[float, float]
CCSAction = dict[str, list[int]]


@dataclass
class CCSEnvConfig:
    episode_hours: int = 168
    storage_target_rate: float = 0.9
    reward_scale: float = 1e-3
    default_speed_knots: float = 12.0
    # Dense shaping: EUR-equivalent reward per tonne of CO2 injected this step.
    # 0.0 keeps the pure economic-net reward (backward compatible). A positive
    # value (e.g. the carbon price) gives the agent an immediate signal for
    # storing CO2 instead of waiting for the delayed venting penalty once buffers
    # overflow, which is what makes short-horizon training reward idling.
    injection_reward_eur_per_t: float = 0.0
    # Tunable economic reward weights. The step reward is
    #   reward_scale * ( store_reward * stored_t
    #                    - vent_penalty_weight * vent_penalty
    #                    - operating_cost_weight * operating_cost )
    # Defaults reproduce the legacy reward exactly: store_reward falls back to
    # injection_reward_eur_per_t, and both weights are 1.0 (so vent penalty and
    # operating cost enter at the cost model's own EUR values). Raise
    # operating_cost_weight to make inefficient routing hurt, i.e. to align the
    # objective with cost-per-stored-tonne instead of "any storage is worth 80".
    store_reward_eur_per_t: float | None = None
    vent_penalty_weight: float = 1.0
    operating_cost_weight: float = 1.0
    # Reward objective. "economic" preserves the existing weighted economic
    # reward. "vent_first" removes the stored-tonne credit and directly
    # prioritises avoided venting, with overflow risk as dense early warning and
    # operating cost as the secondary objective.
    reward_mode: str = "economic"
    vent_first_vent_eur_per_t: float = 10_000.0
    overflow_risk_eur_per_t: float = 100.0
    overflow_risk_lookahead_h: float = 24.0
    # Business dispatch constraints (full vessel before sailing to the terminal;
    # unload before leaving it). Off by default so RL can learn partial-load
    # dispatch when avoiding venting is worth an extra trip. Turn on only for the
    # old curriculum-style behaviour.
    enforce_full_load_dispatch: bool = False
    # Expose weather speed factors (current + 24 h/168 h forecast). Global
    # probability-window weather uses one shared forecast plus per-route travel
    # times; leg weather keeps per-route forecasts. Off by default so existing
    # no-weather models keep their observation size.
    include_weather_obs: bool = False
    weather_observation_layout: str = "leg"
    # Expose a per-vessel emitter assignment (the high-level "goal") in the
    # observation: for each vessel, a one-hot over emitters marking which it is
    # meant to serve. This is what makes the policy goal-conditioned, so it can
    # execute an arbitrary assignment (from a heuristic or an LLM) and transfer
    # zero-shot to new layouts. Off by default (keeps the observation size).
    include_goal_obs: bool = False


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
        if self.config.weather_observation_layout not in {"global", "leg"}:
            raise ValueError(
                "weather_observation_layout must be 'global' or 'leg', "
                f"got {self.config.weather_observation_layout!r}."
            )
        self.scenario_generator = scenario_generator or ScenarioGenerator()
        self.cost_model = cost_model or CostModel()
        self.locations = locations
        self._routes = routes or self._build_routes(locations)
        self._leg_distance_cache: dict[tuple[str, str], float] = {}

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

        # High-level goal: which vessel serves each emitter (set by a planner /
        # LLM / heuristic). Exposed in the observation when include_goal_obs is on.
        self.goal_assignment: dict[str, str] = {}

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

        The modern action spec also includes :attr:`well_rate_action_dims`.
        This compatibility alias intentionally only covers vessel dimensions.
        """
        return self.vessel_action_dims

    @property
    def vessel_action_dims(self) -> list[int]:
        return [self.vessel_action_count] * len(self.vessel_ids)

    def well_rate_bounds(self) -> list[tuple[float, float]]:
        return [self._well_rate_bound(wid) for wid in self.well_ids]

    @property
    def well_rate_action_dims(self) -> list[int]:
        return [len(WELL_RATE_LEVELS_MTPA)] * len(self.well_ids)

    def well_rate_levels_mtpa(self) -> list[float]:
        return list(WELL_RATE_LEVELS_MTPA)

    def action_spec(self) -> dict[str, object]:
        return {
            "vessel_action_dims": self.vessel_action_dims,
            "well_rate_action_dims": self.well_rate_action_dims,
            "well_rate_levels_mtpa": self.well_rate_levels_mtpa(),
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
        if self.config.include_weather_obs:
            if self.config.weather_observation_layout == "global":
                names += [
                    "weather.speed_now",
                    "weather.speed_24h_mean",
                    "weather.speed_24h_min",
                    "weather.speed_168h_mean",
                    "weather.speed_168h_min",
                ]
                for vid in self.vessel_ids:
                    names += [
                        f"{vid}.{label}.travel_hours_now"
                        for label, _destination_id in self._weather_destination_slots()
                    ]
            else:
                for vid in self.vessel_ids:
                    for label, _destination_id in self._weather_destination_slots():
                        names += [
                            f"{vid}.{label}.leg_speed_now",
                            f"{vid}.{label}.leg_speed_24h_mean",
                            f"{vid}.{label}.leg_speed_24h_min",
                            f"{vid}.{label}.leg_speed_168h_mean",
                            f"{vid}.{label}.leg_speed_168h_min",
                            f"{vid}.{label}.travel_hours_now",
                        ]
        if self.config.include_goal_obs:
            for vid in self.vessel_ids:
                names += [f"{vid}.goal_{eid}" for eid in self.emitter_ids]
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
        # Tunable reward. "economic" preserves the legacy weighted economic
        # reward. "vent_first" makes venting the primary signal and uses
        # overflow risk as a dense warning before real venting occurs.
        store_reward = self.config.store_reward_eur_per_t
        if store_reward is None:
            store_reward = self.config.injection_reward_eur_per_t
        storage_shaping_reward = store_reward * economics.stored_t
        overflow_risk_t = self._overflow_risk_t()
        if self.config.reward_mode == "economic":
            reward = (
                storage_shaping_reward
                - self.config.vent_penalty_weight * economics.vent_penalty
                - self.config.operating_cost_weight * economics.operating_cost
            ) * self.config.reward_scale
        elif self.config.reward_mode == "vent_first":
            reward = -(
                self.config.vent_first_vent_eur_per_t * economics.vented_t
                + self.config.overflow_risk_eur_per_t * overflow_risk_t
                + self.config.operating_cost_weight * economics.operating_cost
            ) * self.config.reward_scale
        else:
            raise ValueError(f"Unknown reward_mode: {self.config.reward_mode}")

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
            "storage_shaping_reward": storage_shaping_reward,
            "overflow_risk_t": overflow_risk_t,
            "reward_mode": self.config.reward_mode,
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

    def _overflow_risk_t(self) -> float:
        """Estimated emitter overflow if logistics do not clear buffers soon."""
        state = self.simulator.state
        lookahead_h = max(0.0, float(self.config.overflow_risk_lookahead_h))
        risk_t = 0.0
        for emitter_id in self.emitter_ids:
            emitter = self.network.entities[emitter_id]
            assert isinstance(emitter, Emitter)
            inventory_t = state.entity_inventory_t.get(emitter_id, 0.0)
            headroom_t = max(0.0, emitter.buffer_capacity_t - inventory_t)
            availability = state.emitter_availability.get(emitter_id, emitter.availability)
            capture_tph = emitter.nominal_capture_tph * max(0.0, availability)
            risk_t += max(0.0, capture_tph * lookahead_h - headroom_t)
        return risk_t

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
        well_mask = self.well_rate_action_mask()
        bounds = self.well_rate_bounds()
        return {
            "action_mask": vessel_mask,
            "vessel_action_mask": vessel_mask,
            "well_rate_action_mask": well_mask,
            "well_rate_levels_mtpa": self.well_rate_levels_mtpa(),
            "well_rate_bounds": bounds,
        }

    def action_mask(self) -> list[list[bool]]:
        return self.vessel_action_mask()

    def vessel_action_mask(self) -> list[list[bool]]:
        mask: list[list[bool]] = []
        for vid in self.vessel_ids:
            mask.append(self._vessel_mask(vid))
        return mask

    def well_rate_action_mask(self) -> list[list[bool]]:
        return [self._well_rate_action_mask(well_id) for well_id in self.well_ids]

    def highest_feasible_well_rate_index(self, well_id: str) -> int:
        mask = self._well_rate_action_mask(well_id)
        feasible = [index for index, allowed in enumerate(mask) if allowed]
        return feasible[-1] if feasible else OFF_WELL_RATE_INDEX

    def _vessel_mask(self, vessel_id: str) -> list[bool]:
        vstate = self.simulator.vessel_states[vessel_id]
        route = self._routes[vessel_id]
        if vstate["mode"] != "berthed":
            return [True] + [False] * (self.vessel_action_count - 1)  # mid-voyage: can only WAIT
        berth = vstate["berth"]
        at_terminal = berth == route["destination"]
        if not self.config.enforce_full_load_dispatch:
            mask = [True, not at_terminal]
            mask.extend(berth != emitter_id for emitter_id in self.emitter_ids)
            return mask
        cargo_t = self.simulator.state.entity_inventory_t.get(vessel_id, 0.0)
        vessel = self.network.entities[vessel_id]
        # Business constraint: a loaded vessel at the terminal must finish
        # unloading before it may leave (can only WAIT).
        if at_terminal and cargo_t > 1e-9:
            return [True] + [False] * (self.vessel_action_count - 1)
        # Business constraint: only a full vessel may sail to the terminal.
        terminal_allowed = (not at_terminal) and cargo_t >= vessel.capacity_t - 1e-9
        mask = [True, terminal_allowed]
        mask.extend(berth != emitter_id for emitter_id in self.emitter_ids)
        return mask

    def _well_rate_bound(self, well_id: str) -> tuple[float, float]:
        if self.simulator is None:
            return WELL_RATE_BOUNDS_MTPA
        available = self.simulator.state.well_available.get(well_id, True)
        if not available:
            return (0.0, 0.0)
        return WELL_RATE_BOUNDS_MTPA

    def _well_rate_action_mask(self, well_id: str) -> list[bool]:
        if self.simulator is None:
            return [True] * len(WELL_RATE_LEVELS_MTPA)
        state = self.simulator.state
        well = self.network.entities[well_id]
        assert isinstance(well, InjectionWell)
        if not state.well_available.get(well_id, well.available):
            return [True] + [False] * (len(WELL_RATE_LEVELS_MTPA) - 1)
        effective_max_tph = well_max_injection_tph(state, well)
        return list(
            pressure_limited_rate_level_mask(
                self.network,
                state,
                well_id,
                rate_levels_mtpa=WELL_RATE_LEVELS_MTPA,
                physical_max_rate_tph=effective_max_tph,
                evaluation_time_h=state.time_h + self.network.time_step_hours,
                interval_start_h=state.time_h,
            )
        )

    # -- action translation ----------------------------------------------
    def _normalize_action(self, action: CCSAction) -> dict[str, list]:
        if not isinstance(action, dict):
            raise ValueError("Expected action dict with 'vessels' and 'wells' entries.")
        if "vessels" not in action or "wells" not in action:
            raise ValueError("Expected action dict with 'vessels' and 'wells' entries.")

        vessel_actions = list(action["vessels"])
        well_rate_indices = list(action["wells"])
        if len(vessel_actions) != len(self.vessel_ids):
            raise ValueError(
                f"Expected {len(self.vessel_ids)} vessel actions, got {len(vessel_actions)}."
            )
        if len(well_rate_indices) != len(self.well_ids):
            raise ValueError(
                f"Expected {len(self.well_ids)} well rate actions, got {len(well_rate_indices)}."
            )

        return {
            "vessels": [int(choice) for choice in vessel_actions],
            "wells": [self._normalize_well_rate_index(index) for index in well_rate_indices],
        }

    def _normalize_well_rate_index(self, index) -> int:
        if not isinstance(index, Integral) or isinstance(index, bool):
            raise ValueError("Well action entries must be integer rate-level indices.")
        index = int(index)
        if index < 0 or index >= len(WELL_RATE_LEVELS_MTPA):
            raise ValueError(
                f"Well action index {index} is outside 0..{len(WELL_RATE_LEVELS_MTPA) - 1}."
            )
        return index

    def _build_proposals(self, action: dict[str, list]) -> list[ActionProposal]:
        vessel_actions = action["vessels"]
        well_rate_indices = action["wells"]
        proposals: list[ActionProposal] = []

        # Always capture at full rate (capture is not an RL control here).
        for emitter_id in self.emitter_ids:
            proposals.append(self._proposal(emitter_id, "set_capture_utilization", {"utilization": 1.0}))

        departing = self._vessel_dispatch_proposals(vessel_actions, proposals)
        self._auto_loading_proposals(proposals, departing)
        self._auto_unloading_proposals(proposals, departing)
        self._injection_proposals(well_rate_indices, proposals)
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
            if self.config.enforce_full_load_dispatch:
                cargo_t = self.simulator.state.entity_inventory_t.get(vessel_id, 0.0)
                route = self._routes[vessel_id]
                vessel = self.network.entities[vessel_id]
                # A loaded vessel at the terminal must unload before leaving.
                if berth == route["destination"] and cargo_t > 1e-9:
                    destination = None
                # Only a full vessel may sail to the terminal.
                if destination == route["destination"] and cargo_t < vessel.capacity_t - 1e-9:
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
        terminal = self.network.entities[terminal_id]
        assert isinstance(terminal, Terminal)
        queue = sync_terminal_unload_queue(
            self.network,
            terminal,
            self.simulator.state,
            excluded_vessel_ids=set(departing),
        )
        return queue[0] if queue else None

    def _injection_proposals(self, well_rate_indices, proposals) -> None:
        desired: dict[str, float] = {}
        for well_id, rate_index in zip(self.well_ids, well_rate_indices):
            rate_mtpa = WELL_RATE_LEVELS_MTPA[int(rate_index)]
            desired[well_id] = mtpa_to_tph(rate_mtpa)

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
        if self.config.include_weather_obs:
            if self.config.weather_observation_layout == "global":
                obs += self._global_weather_observation()
            else:
                for vid in self.vessel_ids:
                    obs += self._weather_observation_for_vessel(vid)
        if self.config.include_goal_obs:
            for vid in self.vessel_ids:
                obs += [1.0 if self.goal_assignment.get(eid) == vid else 0.0 for eid in self.emitter_ids]
        return obs

    def set_goal_assignment(self, assignment: dict[str, str]) -> None:
        """Set the high-level emitter->vessel goal exposed in the observation."""
        self.goal_assignment = dict(assignment)

    def _weather_destination_slots(self) -> list[tuple[str, str]]:
        slots = [(f"to_{terminal_id}", terminal_id) for terminal_id in self.terminal_ids]
        slots.extend((f"to_{emitter_id}", emitter_id) for emitter_id in self.emitter_ids)
        return slots

    def _global_current_weather_feature_names(self) -> list[str]:
        names = ["weather.speed_now"]
        for vessel_id in self.vessel_ids:
            names += [
                f"{vessel_id}.{label}.travel_hours_now"
                for label, _destination_id in self._weather_destination_slots()
            ]
        return names

    def _global_current_weather_observation(self) -> list[float]:
        vessel_id = self.vessel_ids[0]
        now = self._weather_speed_at("", vessel_id, 0)
        values = [now]
        for current_vessel_id in self.vessel_ids:
            route = self._routes[current_vessel_id]
            origin_id = self._weather_reference_origin(current_vessel_id)
            for _label, destination_id in self._weather_destination_slots():
                values.append(
                    self._normalized_travel_hours(origin_id, destination_id, route, now)
                )
        return values

    def _global_weather_observation(self) -> list[float]:
        vessel_id = self.vessel_ids[0]
        mean24, min24 = self._weather_speed_forecast("", vessel_id, 24)
        mean168, min168 = self._weather_speed_forecast("", vessel_id, 168)
        current = self._global_current_weather_observation()
        return [current[0], mean24, min24, mean168, min168, *current[1:]]

    def _weather_observation_for_vessel(self, vessel_id: str) -> list[float]:
        route = self._routes[vessel_id]
        origin_id = self._weather_reference_origin(vessel_id)
        values: list[float] = []
        for _label, destination_id in self._weather_destination_slots():
            if destination_id == origin_id:
                values += [1.0, 1.0, 1.0, 1.0, 1.0, 0.0]
                continue
            leg_id = f"{origin_id}->{destination_id}"
            now = self._weather_speed_at(leg_id, vessel_id, 0)
            mean24, min24 = self._weather_speed_forecast(leg_id, vessel_id, 24)
            mean168, min168 = self._weather_speed_forecast(leg_id, vessel_id, 168)
            travel_hours = self._normalized_travel_hours(origin_id, destination_id, route, now)
            values += [now, mean24, min24, mean168, min168, travel_hours]
        return values

    def _weather_reference_origin(self, vessel_id: str) -> str:
        vstate = self.simulator.vessel_states[vessel_id]
        if vstate["mode"] == "berthed":
            return str(vstate["berth"])
        return str(vstate.get("origin") or self._routes[vessel_id]["origin"])

    def _normalized_travel_hours(self, origin_id: str, destination_id: str, route: dict, speed_factor: float) -> float:
        speed_knots = float(route.get("speed_knots") or self.config.default_speed_knots)
        effective_speed_knots = speed_knots * max(0.0, speed_factor)
        if effective_speed_knots <= 1e-9:
            return 1.0
        distance_km = self._leg_distance_km(origin_id, destination_id, route)
        travel_hours = distance_km / (effective_speed_knots * 1.852) if distance_km > 0 else 0.0
        return max(0.0, min(1.0, travel_hours / max(1.0, float(self.config.episode_hours))))

    def _leg_distance_km(self, origin_id: str, destination_id: str, route: dict) -> float:
        if origin_id == destination_id:
            return 0.0
        key = (origin_id, destination_id)
        if key in self._leg_distance_cache:
            return self._leg_distance_cache[key]
        if {origin_id, destination_id} == {route["origin"], route["destination"]}:
            distance_km = float(route.get("distance_km") or 0.0)
        else:
            if origin_id not in self.locations or destination_id not in self.locations:
                distance_km = 0.0
            else:
                maritime_route = sea_route(self.locations[origin_id], self.locations[destination_id])
                distance_km = route_distance_km(maritime_route.coordinates)
        self._leg_distance_cache[key] = distance_km
        return distance_km

    def _weather_speed_series(self, leg_id: str, vessel_id: str) -> list[float] | None:
        """Full weather speed-factor series, preferring leg-level data when present."""
        if self.scenario is None:
            return None
        series = self.scenario.leg_speed_factor.get(leg_id)
        if series:
            return series
        series = self.scenario.vessel_speed_factor.get(vessel_id)
        return series if series else None

    def _weather_speed_at(self, leg_id: str, vessel_id: str, offset_h: int) -> float:
        series = self._weather_speed_series(leg_id, vessel_id)
        if not series:
            return 1.0
        idx = int(round(self.simulator.state.time_h / self.network.time_step_hours)) + offset_h
        idx = max(0, min(idx, len(series) - 1))
        return float(series[idx])

    def _weather_speed_forecast(self, leg_id: str, vessel_id: str, window_h: int) -> tuple[float, float]:
        """(mean, min) weather speed factor over the next ``window_h`` hours."""
        series = self._weather_speed_series(leg_id, vessel_id)
        if not series:
            return 1.0, 1.0
        start = int(round(self.simulator.state.time_h / self.network.time_step_hours))
        window = series[start : start + window_h]
        if not window:
            window = [series[min(start, len(series) - 1)]]
        return sum(window) / len(window), min(window)

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
