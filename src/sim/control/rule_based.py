from __future__ import annotations

from typing import Any

from ..actions import ActionFrame, ActionProposal
from ..scenario_generation.disturbance_resolver import terminal_berth_count
from ..entities.emitter import Emitter
from ..entities.manifold import SubseaManifold
from ..entities.pipeline import Pipeline
from ..entities.state import PhysicalState
from ..entities.storage import InjectionWell
from ..entities.terminal import Terminal
from ..entities.vessel import Vessel
from ..network import PhysicalNetwork


class RuleBasedActionGenerator:
    """Simple scenario-aware baseline dispatcher.

    Rules:
    - keep every emitter at full utilization;
    - load a berthed, not-full vessel at whichever emitter it is serving;
    - send full vessels to the terminal;
    - unload loaded vessels at the terminal;
    - send empty vessels to the emitter with the most available buffer;
    - use one injection well and never proactively shut wells.
    """

    def __init__(
        self,
        network: PhysicalNetwork,
        routes: dict[str, dict[str, Any]],
        *,
        selected_well_id: str | None = None,
        agent_id: str = "rule_based_dispatcher",
    ) -> None:
        self.network = network
        self.routes = routes
        self.agent_id = agent_id
        self.selected_well_id = selected_well_id or self._first_injection_well_id()
        self._terminal_unload_queues: dict[str, list[str]] = {}

    def next_action_frame(self, state: PhysicalState) -> ActionFrame:
        self._update_terminal_unload_queues(state)
        proposals: list[ActionProposal] = []
        proposals.extend(self._full_capture_actions())
        proposals.extend(self._vessel_cycle_actions(state))
        proposals.extend(self._single_well_actions())
        proposals.extend(self._pipeline_actions(state))
        return ActionFrame(time_h=state.time_h, proposals=proposals)

    def _full_capture_actions(self) -> list[ActionProposal]:
        return [
            self._proposal(
                emitter_id,
                "set_capture_utilization",
                {"utilization": 1.0},
            )
            for emitter_id, entity in self.network.entities.items()
            if isinstance(entity, Emitter)
        ]

    def _vessel_cycle_actions(self, state: PhysicalState) -> list[ActionProposal]:
        proposals: list[ActionProposal] = []
        for vessel_id, entity in self.network.entities.items():
            if not isinstance(entity, Vessel):
                continue
            route = self.routes.get(vessel_id)
            if not route:
                continue
            terminal_id = str(route["destination"])
            berth_id = state.vessel_berths.get(vessel_id)
            if berth_id is None:
                continue

            cargo_t = state.entity_inventory_t.get(vessel_id, 0.0)
            is_full = cargo_t >= entity.capacity_t - 1e-9
            is_empty = cargo_t <= 1e-9

            if berth_id == terminal_id:
                if is_empty:
                    emitter_id = self._best_emitter_id(state)
                    if emitter_id is not None:
                        proposals.append(self._proposal(vessel_id, "sail_to", {"destination_id": emitter_id}))
                elif self._is_fifo_unload_head(terminal_id, vessel_id):
                    proposals.append(self._proposal(terminal_id, "unload_vessel", {"vessel_id": vessel_id}))
                continue

            if is_full:
                proposals.append(self._proposal(vessel_id, "sail_to", {"destination_id": terminal_id}))
            elif isinstance(self.network.entities.get(berth_id), Emitter):
                if self._emitter_supply_score(state, berth_id) > 1e-9:
                    proposals.append(self._proposal(berth_id, "load_vessel", {"vessel_id": vessel_id}))
                else:
                    emitter_id = self._best_emitter_id(state, exclude=berth_id)
                    if emitter_id is not None:
                        proposals.append(self._proposal(vessel_id, "sail_to", {"destination_id": emitter_id}))
        return proposals

    def _best_emitter_id(self, state: PhysicalState, *, exclude: str | None = None) -> str | None:
        best: tuple[float, str] | None = None
        for emitter_id, entity in self.network.entities.items():
            if not isinstance(entity, Emitter) or emitter_id == exclude:
                continue
            score = self._emitter_supply_score(state, emitter_id)
            if best is None or score > best[0]:
                best = (score, emitter_id)
        return None if best is None else best[1]

    def _emitter_supply_score(self, state: PhysicalState, emitter_id: str) -> float:
        emitter = self.network.entities[emitter_id]
        assert isinstance(emitter, Emitter)
        availability = state.emitter_availability.get(emitter_id, emitter.availability)
        return state.entity_inventory_t.get(emitter_id, 0.0) + emitter.nominal_capture_tph * max(0.0, availability)

    def _update_terminal_unload_queues(self, state: PhysicalState) -> None:
        terminal_ids = [
            entity_id
            for entity_id, entity in self.network.entities.items()
            if isinstance(entity, Terminal)
        ]
        for terminal_id in terminal_ids:
            queue = self._terminal_unload_queues.setdefault(terminal_id, [])
            queue[:] = [
                vessel_id
                for vessel_id in queue
                if (
                    state.vessel_berths.get(vessel_id) == terminal_id
                    and state.entity_inventory_t.get(vessel_id, 0.0) > 1e-9
                )
            ]
            for vessel_id in self.network._upstream_of_type(terminal_id, Vessel):
                if (
                    state.vessel_berths.get(vessel_id) == terminal_id
                    and state.entity_inventory_t.get(vessel_id, 0.0) > 1e-9
                    and vessel_id not in queue
                ):
                    queue.append(vessel_id)

    def _is_fifo_unload_head(self, terminal_id: str, vessel_id: str) -> bool:
        queue = self._terminal_unload_queues.get(terminal_id, [])
        return bool(queue) and queue[0] == vessel_id

    def _single_well_actions(self) -> list[ActionProposal]:
        proposals: list[ActionProposal] = []
        if self.selected_well_id is None:
            return proposals
        for manifold_id, entity in self.network.entities.items():
            if not isinstance(entity, SubseaManifold):
                continue
            downstream_wells = self.network._downstream_of_type(manifold_id, InjectionWell)
            if self.selected_well_id not in downstream_wells:
                continue
            proposals.append(
                self._proposal(
                    manifold_id,
                    "set_well_split",
                    {
                        "well_splits": {
                            well_id: 1.0 if well_id == self.selected_well_id else 0.0
                            for well_id in downstream_wells
                        }
                    },
                )
            )
        return proposals

    def _pipeline_actions(self, state: PhysicalState) -> list[ActionProposal]:
        proposals: list[ActionProposal] = []
        if self.selected_well_id is None:
            return proposals
        selected_well = self.network.entities.get(self.selected_well_id)
        if not isinstance(selected_well, InjectionWell):
            return proposals

        for pipeline_id, pipeline in self.network.entities.items():
            if not isinstance(pipeline, Pipeline):
                continue
            terminal_id = self._upstream_terminal_id(pipeline_id)
            if terminal_id is None:
                continue
            supply_t = state.entity_inventory_t.get(terminal_id, 0.0) + self._requested_unload_supply_t(state, terminal_id)
            if supply_t <= 1e-9:
                continue
            flow_tph = min(
                pipeline.max_flow_tph,
                selected_well.max_injection_tph,
                supply_t / self.network.time_step_hours,
            )
            proposals.append(self._proposal(pipeline_id, "set_flow", {"flow_tph": flow_tph}))
        return proposals

    def _requested_unload_supply_t(self, state: PhysicalState, terminal_id: str) -> float:
        terminal = self.network.entities[terminal_id]
        if not isinstance(terminal, Terminal):
            return 0.0
        amount_t = 0.0
        berth_slots = terminal_berth_count(state, terminal)
        for vessel_id in self.network._upstream_of_type(terminal_id, Vessel):
            if berth_slots <= 0:
                break
            if state.vessel_berths.get(vessel_id) != terminal_id:
                continue
            vessel = self.network.entities[vessel_id]
            assert isinstance(vessel, Vessel)
            cargo_t = state.entity_inventory_t.get(vessel_id, 0.0)
            if cargo_t <= 1e-9:
                continue
            amount_t += min(cargo_t, vessel.unloading_rate_tph * self.network.time_step_hours)
            berth_slots -= 1
        return amount_t

    def _upstream_terminal_id(self, pipeline_id: str) -> str | None:
        for upstream_id in self.network.upstream_of(pipeline_id):
            if isinstance(self.network.entities.get(upstream_id), Terminal):
                return upstream_id
        return None

    def _first_injection_well_id(self) -> str | None:
        for entity_id, entity in self.network.entities.items():
            if isinstance(entity, InjectionWell):
                return entity_id
        return None

    def _proposal(self, entity_id: str, verb: str, params: dict[str, Any]) -> ActionProposal:
        return ActionProposal(
            agent_id=self.agent_id,
            entity_id=entity_id,
            verb=verb,
            params=params,
        )
