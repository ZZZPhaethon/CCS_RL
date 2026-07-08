from __future__ import annotations

from ..scenario_generation.disturbance_resolver import terminal_berth_count
from ..entities.state import PhysicalState, Violation
from ..entities.terminal import Terminal
from ..entities.vessel import Vessel


def terminal_unload_request_capacity(
    network,
    terminal: Terminal,
    state: PhysicalState,
    actions: dict[str, dict[str, object]],
) -> float:
    vessel_ids = _terminal_vessels_for_action(network, terminal, state, actions)
    requested_t = _terminal_unload_requested_t(network, terminal, state, actions)
    remaining_t = requested_t
    capacity_t = 0.0
    for vessel_id in vessel_ids:
        vessel = network.entities[vessel_id]
        assert isinstance(vessel, Vessel)
        amount_t = min(
            remaining_t,
            state.entity_inventory_t.get(vessel_id, 0.0),
            vessel.unloading_rate_tph * network.time_step_hours,
        )
        capacity_t += amount_t
        remaining_t -= amount_t
        if remaining_t <= 1e-12:
            break
    return capacity_t


def project_terminal_unload(
    network,
    terminal: Terminal,
    outflow_t: float,
    state: PhysicalState,
    actions: dict[str, dict[str, object]],
    violations: list[Violation],
) -> dict[str, float]:
    vessel_ids = _terminal_vessels_for_action(network, terminal, state, actions)
    requested_t = _terminal_unload_requested_t(network, terminal, state, actions)
    if not vessel_ids and requested_t > 0:
        upstream_vessels = list(network._entities_of_type(Vessel))
        requested_vessel_id = _requested_unload_vessel_id(terminal.entity_id, actions)
        violation_entity_id = (
            requested_vessel_id
            if requested_vessel_id in upstream_vessels
            else upstream_vessels[0] if upstream_vessels else terminal.entity_id
        )
        violations.append(
            Violation(
                "berth_required",
                violation_entity_id,
                requested_t,
                0.0,
                requested_t,
                "Unload request requires the vessel to be at the terminal berth.",
            )
        )
    if not vessel_ids:
        return {}
    terminal_inventory_t = state.entity_inventory_t.get(terminal.entity_id, 0.0)
    free_with_outflow_t = max(0.0, terminal.storage_capacity_t - terminal_inventory_t + outflow_t)
    if "unload_tph" not in actions.get(terminal.entity_id, {}):
        requested_t = min(requested_t, free_with_outflow_t)
    remaining_t = min(requested_t, free_with_outflow_t)
    unloaded: dict[str, float] = {}
    for vessel_id in vessel_ids:
        vessel = network.entities[vessel_id]
        assert isinstance(vessel, Vessel)
        vessel_inventory_t = state.entity_inventory_t.get(vessel_id, 0.0)
        amount_t = min(remaining_t, vessel_inventory_t, vessel.unloading_rate_tph * network.time_step_hours)
        if amount_t > 0:
            unloaded[vessel_id] = amount_t
            remaining_t -= amount_t
    actual_t = sum(unloaded.values())
    if actual_t < requested_t:
        violations.append(
            Violation(
                "flow_clipped",
                terminal.entity_id,
                requested_t,
                actual_t,
                requested_t - actual_t,
                "Unload request clipped by berth, vessel cargo, or terminal free capacity.",
            )
        )
    return unloaded


def _terminal_vessels_for_action(
    network,
    terminal: Terminal,
    state: PhysicalState,
    actions: dict[str, dict[str, object]],
) -> list[str]:
    vessel_ids = list(network._entities_of_type(Vessel))
    berthed_vessel_ids = _vessels_berthed_at(vessel_ids, state, terminal.entity_id)
    requested_vessel_id = _requested_unload_vessel_id(terminal.entity_id, actions)
    berth_count = terminal_berth_count(state, terminal)
    if requested_vessel_id in berthed_vessel_ids:
        return [requested_vessel_id] if berth_count > 0 else []
    return berthed_vessel_ids[:berth_count]


def _requested_unload_vessel_id(
    terminal_id: str,
    actions: dict[str, dict[str, object]],
) -> object:
    action = actions.get(terminal_id, {})
    return action.get("unload_vessel", action.get("vessel_id"))


def _terminal_unload_requested_t(
    network,
    terminal: Terminal,
    state: PhysicalState,
    actions: dict[str, dict[str, object]],
) -> float:
    action = actions.get(terminal.entity_id, {})
    if "unload_tph" in action:
        return float(action["unload_tph"]) * network.time_step_hours
    requested_vessel_id = action.get("unload_vessel")
    if requested_vessel_id not in network.entities:
        return 0.0
    vessel = network.entities[requested_vessel_id]
    if not isinstance(vessel, Vessel):
        return 0.0
    return min(
        state.entity_inventory_t.get(str(requested_vessel_id), 0.0),
        vessel.unloading_rate_tph * network.time_step_hours,
    )


def _vessels_berthed_at(
    vessel_ids: list[str],
    state: PhysicalState,
    location_id: str,
) -> list[str]:
    return [
        vessel_id
        for vessel_id in vessel_ids
        if state.vessel_berths.get(vessel_id) == location_id
    ]
