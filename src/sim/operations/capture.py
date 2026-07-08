from __future__ import annotations

from ..scenario_generation.disturbance_resolver import emitter_availability
from ..entities.emitter import Emitter
from ..entities.state import PhysicalState, Violation


def apply_capture(
    network,
    state: PhysicalState,
    actions: dict[str, dict[str, object]],
    violations: list[Violation],
) -> float:
    generated_t = 0.0
    interval_start_h = state.time_h - network.time_step_hours
    for emitter_id, emitter in network._entities_of_type(Emitter).items():
        utilization = actions.get(emitter_id, {}).get("utilization", emitter.default_utilization)
        utilization = max(emitter.min_utilization, min(1.0, utilization))
        requested_t = (
            emitter.capture_rate_tph_at(interval_start_h)
            * utilization
            * emitter_availability(state, emitter)
            * network.time_step_hours
        )
        current_t = state.entity_inventory_t.get(emitter_id, 0.0)
        free_t = max(0.0, emitter.buffer_capacity_t - current_t)
        actual_t = min(requested_t, free_t)
        vented_t = requested_t - actual_t
        state.entity_inventory_t[emitter_id] = current_t + actual_t
        state.last_capture_tph[emitter_id] = actual_t / network.time_step_hours
        state.last_vent_tph[emitter_id] = vented_t / network.time_step_hours
        state.cumulative_vent_t[emitter_id] = state.cumulative_vent_t.get(emitter_id, 0.0) + vented_t
        generated_t += actual_t
        if actual_t < requested_t:
            violations.append(
                Violation(
                    "vented_capture",
                    emitter_id,
                    requested_t,
                    actual_t,
                    vented_t,
                    "Emitter vented CO2 because its buffer had no remaining capacity.",
                )
            )
    return generated_t
