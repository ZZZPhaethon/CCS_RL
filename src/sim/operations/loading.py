from __future__ import annotations

from ..entities.emitter import Emitter
from ..entities.state import PhysicalState, Violation
from ..entities.vessel import Vessel


def apply_loading(
    network,
    state: PhysicalState,
    actions: dict[str, dict[str, object]],
    flows: dict[tuple[str, str], float],
    violations: list[Violation],
) -> None:
    for emitter_id, emitter in network._entities_of_type(Emitter).items():
        action = actions.get(emitter_id, {})
        if "load_tph" not in action and "load_vessel" not in action:
            continue
        vessel_ids = list(network._entities_of_type(Vessel))
        if not vessel_ids:
            continue
        requested_vessel_id = action.get("load_vessel", action.get("vessel_id"))
        berthed_vessel_ids = _vessels_berthed_at(vessel_ids, state, emitter_id)
        if requested_vessel_id in vessel_ids:
            vessel_id = requested_vessel_id
            requested_t = _emitter_load_requested_t(network, emitter, state, action, vessel_id)
            if vessel_id not in berthed_vessel_ids:
                violations.append(
                    Violation(
                        "berth_required",
                        vessel_id,
                        requested_t,
                        0.0,
                        requested_t,
                        "Loading request requires the vessel to be at the emitter berth.",
                    )
                )
                continue
        elif berthed_vessel_ids:
            vessel_id = berthed_vessel_ids[0]
            requested_t = _emitter_load_requested_t(network, emitter, state, action, vessel_id)
        else:
            requested_t = actions.get(emitter_id, {}).get("load_tph", 0.0) * network.time_step_hours
            violation_entity_id = requested_vessel_id if requested_vessel_id in vessel_ids else vessel_ids[0]
            violations.append(
                Violation(
                    "berth_required",
                    violation_entity_id,
                    requested_t,
                    0.0,
                    requested_t,
                    "Loading request requires a vessel to be at the emitter berth.",
                )
            )
            continue
        vessel = network.entities[vessel_id]
        assert isinstance(vessel, Vessel)
        emitter_inventory_t = state.entity_inventory_t.get(emitter_id, 0.0)
        vessel_inventory_t = state.entity_inventory_t.get(vessel_id, 0.0)
        amount_t = min(
            requested_t,
            emitter.loading_rate_tph * network.time_step_hours,
            vessel.loading_rate_tph * network.time_step_hours,
            emitter_inventory_t,
            max(0.0, vessel.capacity_t - vessel_inventory_t),
        )
        if amount_t > 0:
            network._move(state, flows, emitter_id, vessel_id, amount_t)
        if amount_t < requested_t:
            violations.append(
                Violation(
                    "flow_clipped",
                    emitter_id,
                    requested_t,
                    amount_t,
                    requested_t - amount_t,
                    "Loading request clipped by source inventory, loading rate, or vessel capacity.",
                )
            )


def _emitter_load_requested_t(
    network,
    emitter: Emitter,
    state: PhysicalState,
    action: dict[str, object],
    vessel_id: str,
) -> float:
    if "load_tph" in action:
        return float(action["load_tph"]) * network.time_step_hours
    vessel = network.entities[vessel_id]
    assert isinstance(vessel, Vessel)
    emitter_inventory_t = state.entity_inventory_t.get(emitter.entity_id, 0.0)
    vessel_inventory_t = state.entity_inventory_t.get(vessel_id, 0.0)
    return min(
        emitter.loading_rate_tph * network.time_step_hours,
        vessel.loading_rate_tph * network.time_step_hours,
        emitter_inventory_t,
        max(0.0, vessel.capacity_t - vessel_inventory_t),
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
