from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..entities.emitter import Emitter
from ..entities.manifold import SubseaManifold
from ..entities.pipeline import Pipeline
from ..entities.storage import InjectionWell, Reservoir
from ..entities.terminal import Terminal
from ..entities.vessel import Vessel
from ..network import PhysicalNetwork
from .action import ActionDecision, ActionFrame, ActionProposal, CommittedActionFrame


@dataclass(frozen=True)
class ActionSpec:
    verb: str
    required_params: tuple[str, ...] = ()


ACTION_SPECS_BY_ENTITY_TYPE: dict[type, tuple[ActionSpec, ...]] = {
    Emitter: (
        ActionSpec("set_capture_utilization", ("utilization",)),
        ActionSpec("load_vessel", ("vessel_id",)),
        ActionSpec("hold"),
    ),
    Vessel: (
        ActionSpec("sail_to", ("destination_id",)),
        ActionSpec("hold"),
    ),
    Terminal: (
        ActionSpec("unload_vessel", ("vessel_id",)),
        ActionSpec("hold"),
    ),
    Pipeline: (
        ActionSpec("set_flow", ("flow_tph",)),
        ActionSpec("hold"),
    ),
    SubseaManifold: (
        ActionSpec("set_well_split", ("well_splits",)),
        ActionSpec("hold"),
    ),
    InjectionWell: (
        ActionSpec("set_available", ("available",)),
        ActionSpec("set_injection_limit", ("max_injection_tph",)),
        ActionSpec("hold"),
    ),
    Reservoir: (ActionSpec("hold"),),
}


class ActionResolver:
    """Validate heterogeneous agent proposals and translate accepted actions."""

    def __init__(self, network: PhysicalNetwork) -> None:
        self.network = network

    def resolve(self, frame: ActionFrame) -> CommittedActionFrame:
        actions: dict[str, dict[str, Any]] = {}
        decisions: list[ActionDecision] = []

        for proposal in frame.proposals:
            translated = self._translate(proposal)
            reason = self._rejection_reason(proposal, translated, actions)
            if reason:
                decisions.append(ActionDecision(proposal, accepted=False, reason=reason))
                continue
            if translated:
                actions.setdefault(proposal.entity_id, {}).update(translated)
            decisions.append(ActionDecision(proposal, accepted=True))

        return CommittedActionFrame(time_h=frame.time_h, actions=actions, decisions=decisions)

    def supported_actions_by_entity(self) -> dict[str, list[str]]:
        return {
            entity_id: [spec.verb for spec in self._specs_for_entity(entity)]
            for entity_id, entity in self.network.entities.items()
        }

    def _rejection_reason(
        self,
        proposal: ActionProposal,
        translated: dict[str, Any],
        committed_actions: dict[str, dict[str, Any]],
    ) -> str:
        entity = self.network.entities.get(proposal.entity_id)
        if entity is None:
            return f"Unknown entity: {proposal.entity_id}"
        specs = {spec.verb: spec for spec in self._specs_for_entity(entity)}
        spec = specs.get(proposal.verb)
        if spec is None:
            return f"{type(entity).__name__} does not support action {proposal.verb}."
        missing = [param for param in spec.required_params if param not in proposal.params]
        if missing:
            return f"Action {proposal.verb} is missing required params: {', '.join(missing)}."
        invalid = self._invalid_param_reason(proposal)
        if invalid:
            return invalid
        existing = committed_actions.get(proposal.entity_id, {})
        if any(key in existing for key in translated):
            return f"Action {proposal.verb} conflicts with an existing action for {proposal.entity_id}."
        return ""

    def _invalid_param_reason(self, proposal: ActionProposal) -> str:
        if proposal.verb == "set_capture_utilization":
            utilization = proposal.params["utilization"]
            if not isinstance(utilization, int | float) or isinstance(utilization, bool):
                return "Action set_capture_utilization parameter utilization must be numeric."
            if utilization < 0.0 or utilization > 1.0:
                return "Action set_capture_utilization parameter utilization must be between 0 and 1."
        if proposal.verb == "set_flow":
            flow_tph = proposal.params["flow_tph"]
            if not isinstance(flow_tph, int | float) or isinstance(flow_tph, bool):
                return "Action set_flow parameter flow_tph must be numeric."
            if flow_tph < 0.0:
                return "Action set_flow parameter flow_tph must be non-negative."
        if proposal.verb == "set_available" and not isinstance(proposal.params["available"], bool):
            return "Action set_available parameter available must be boolean."
        if proposal.verb == "set_injection_limit":
            max_injection_tph = proposal.params["max_injection_tph"]
            if not isinstance(max_injection_tph, int | float) or isinstance(max_injection_tph, bool):
                return "Action set_injection_limit parameter max_injection_tph must be numeric."
            if max_injection_tph < 0.0:
                return "Action set_injection_limit parameter max_injection_tph must be non-negative."
        if proposal.verb == "set_well_split":
            well_splits = proposal.params["well_splits"]
            if not isinstance(well_splits, dict) or not well_splits:
                return "Action set_well_split parameter well_splits must be a non-empty mapping."
            total = 0.0
            downstream_wells = set(self.network.downstream_of(proposal.entity_id))
            for well_id, split in well_splits.items():
                if str(well_id) not in downstream_wells:
                    return "Action set_well_split can only target downstream wells."
                if not isinstance(split, int | float) or isinstance(split, bool):
                    return "Action set_well_split values must be numeric."
                if split < 0.0:
                    return "Action set_well_split values must be non-negative."
                total += float(split)
            if abs(total - 1.0) > 1e-9:
                return "Action set_well_split values must sum to 1."
        return ""

    def _specs_for_entity(self, entity: object) -> tuple[ActionSpec, ...]:
        for entity_type, specs in ACTION_SPECS_BY_ENTITY_TYPE.items():
            if isinstance(entity, entity_type):
                return specs
        return ()

    def _translate(self, proposal: ActionProposal) -> dict[str, Any]:
        if proposal.verb == "hold":
            return {}
        if proposal.verb == "set_capture_utilization":
            return {"utilization": float(proposal.params["utilization"])}
        if proposal.verb == "load_vessel":
            return {"load_vessel": str(proposal.params["vessel_id"])}
        if proposal.verb == "sail_to":
            return {"sail_to": str(proposal.params["destination_id"])}
        if proposal.verb == "unload_vessel":
            return {"unload_vessel": str(proposal.params["vessel_id"])}
        if proposal.verb == "set_flow":
            return {"flow_tph": float(proposal.params["flow_tph"])}
        if proposal.verb == "set_available":
            return {"available": bool(proposal.params["available"])}
        if proposal.verb == "set_injection_limit":
            return {"max_injection_tph": float(proposal.params["max_injection_tph"])}
        if proposal.verb == "set_well_split":
            return {
                "well_splits": {
                    str(well_id): float(split)
                    for well_id, split in proposal.params["well_splits"].items()
                }
            }
        return {}
