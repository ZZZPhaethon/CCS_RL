"""Read-only operational context features for vessel dispatch policies."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..entities.emitter import Emitter
from ..entities.terminal import Terminal
from ..entities.vessel import Vessel
from ..operations.unloading import terminal_unload_queue_snapshot
from ..scenario_generation.disturbance_resolver import terminal_berth_count

if TYPE_CHECKING:
    from .env import CCSEnv


VESSEL_OPERATION_MODES = ("sailing", "loading", "unloading", "queued", "idle")
_EPS = 1e-9


def vessel_operation_modes(env: "CCSEnv") -> tuple[str, ...]:
    """Classify each vessel's current pre-action operational mode."""

    if env.simulator is None:
        raise RuntimeError("Call env.reset() before requesting vessel operation modes.")

    state = env.simulator.state
    modes = {vessel_id: "idle" for vessel_id in env.vessel_ids}

    for vessel_id in env.vessel_ids:
        if env.simulator.vessel_states[vessel_id]["mode"] != "berthed":
            modes[vessel_id] = "sailing"

    for terminal_id in env.terminal_ids:
        terminal = env.network.entities[terminal_id]
        assert isinstance(terminal, Terminal)
        queue = terminal_unload_queue_snapshot(env.network, terminal, state)
        can_unload = terminal_berth_count(state, terminal) > 0
        for position, vessel_id in enumerate(queue):
            if modes[vessel_id] == "sailing":
                continue
            modes[vessel_id] = "unloading" if position == 0 and can_unload else "queued"

    for emitter_id in env.emitter_ids:
        emitter = env.network.entities[emitter_id]
        assert isinstance(emitter, Emitter)
        if state.entity_inventory_t.get(emitter_id, 0.0) <= _EPS:
            continue
        eligible = []
        for vessel_id in env.vessel_ids:
            if modes[vessel_id] == "sailing":
                continue
            vessel_state = env.simulator.vessel_states[vessel_id]
            if (
                vessel_state["berth"] != emitter_id
                or state.vessel_berths.get(vessel_id) != emitter_id
            ):
                continue
            vessel = env.network.entities[vessel_id]
            assert isinstance(vessel, Vessel)
            cargo_t = state.entity_inventory_t.get(vessel_id, 0.0)
            if (
                cargo_t < vessel.capacity_t - _EPS
                and emitter.loading_rate_tph > 0.0
                and vessel.loading_rate_tph > 0.0
            ):
                eligible.append(vessel_id)
        for position, vessel_id in enumerate(eligible):
            modes[vessel_id] = "loading" if position == 0 else "queued"

    return tuple(modes[vessel_id] for vessel_id in env.vessel_ids)


def vessel_operation_mode_feature_names(env: "CCSEnv") -> tuple[str, ...]:
    return tuple(
        f"{vessel_id}.mode_{mode}"
        for vessel_id in env.vessel_ids
        for mode in VESSEL_OPERATION_MODES
    )


def vessel_operation_mode_observation(env: "CCSEnv") -> list[float]:
    modes = vessel_operation_modes(env)
    return [
        1.0 if current_mode == candidate else 0.0
        for current_mode in modes
        for candidate in VESSEL_OPERATION_MODES
    ]
