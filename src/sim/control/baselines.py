"""Simple baseline controllers for CCS environments."""

from __future__ import annotations

from ..environment import (
    CCSEnv,
    MIN_WELL_RATE_INDEX,
    VESSEL_GO_TERMINAL,
    VESSEL_WAIT,
)

_EPS = 1e-9


def idle_policy(env: CCSEnv) -> dict[str, list]:
    """Do nothing with vessels; legacy environments hold wells at minimum."""
    action = {"vessels": [VESSEL_WAIT] * len(env.vessel_ids)}
    if not env.automatic_well_control:
        action["wells"] = [MIN_WELL_RATE_INDEX] * len(env.well_ids)
    return action


def greedy_shuttle_policy(env: CCSEnv) -> dict[str, list]:
    """Send full vessels to terminal, otherwise serve the best buffered emitter."""
    state = env.simulator.state
    action: list[int] = []
    for i, vessel_id in enumerate(env.vessel_ids):
        mask = env.vessel_action_mask()[i]
        cargo = state.entity_inventory_t.get(vessel_id, 0.0)
        vessel = env.network.entities[vessel_id]
        berth = state.vessel_berths.get(vessel_id)
        best_emitter_action = _best_emitter_action(env, mask)

        if berth in env.terminal_ids and cargo > _EPS:
            action.append(VESSEL_WAIT)
        elif mask[VESSEL_GO_TERMINAL] and cargo >= vessel.capacity_t - _EPS:
            action.append(VESSEL_GO_TERMINAL)
        elif (
            berth in env.emitter_ids
            and cargo < vessel.capacity_t - _EPS
            and _emitter_supply_score(env, str(berth)) > _EPS
        ):
            action.append(VESSEL_WAIT)
        elif best_emitter_action is not None:
            action.append(best_emitter_action)
        elif mask[VESSEL_GO_TERMINAL] and cargo > _EPS:
            action.append(VESSEL_GO_TERMINAL)
        else:
            action.append(VESSEL_WAIT)
    return _with_legacy_automatic_wells(env, action)


def _best_emitter_action(env: CCSEnv, mask: list[bool]) -> int | None:
    best: tuple[float, int] | None = None
    for emitter_id in env.emitter_ids:
        action = env.vessel_go_emitter_action(emitter_id)
        if not mask[action]:
            continue
        score = _emitter_supply_score(env, emitter_id)
        if best is None or score > best[0]:
            best = (score, action)
    return None if best is None else best[1]


def _emitter_supply_score(env: CCSEnv, emitter_id: str) -> float:
    emitter = env.network.entities[emitter_id]
    state = env.simulator.state
    availability = state.emitter_availability.get(emitter_id, emitter.availability)
    return state.entity_inventory_t.get(emitter_id, 0.0) + emitter.nominal_capture_tph * max(0.0, availability)


def balanced_capture_assignment(env: CCSEnv) -> dict[str, str]:
    """Assign each emitter to a vessel, balancing total capture rate per vessel.

    Heaviest emitters first, each to the least-loaded vessel - a load-balanced
    partition that avoids overloading one vessel with several high-rate sources.
    """
    rates = {e: env.network.entities[e].nominal_capture_tph for e in env.emitter_ids}
    load = {v: 0.0 for v in env.vessel_ids}
    assign: dict[str, str] = {}
    for emitter_id in sorted(env.emitter_ids, key=lambda e: -rates[e]):
        vessel_id = min(env.vessel_ids, key=lambda v: load[v])
        assign[emitter_id] = vessel_id
        load[vessel_id] += rates[emitter_id]
    return assign


def make_cluster_shuttle_policy(env: CCSEnv, assignment: dict[str, str] | None = None):
    """Shuttle policy where each vessel only serves its assigned emitters.

    Unlike ``greedy_shuttle_policy`` (which lets every vessel chase the globally
    fullest emitter and mis-routes under a tight fleet), this partitions emitters
    across vessels. ``assignment`` maps emitter->vessel; if ``None`` a
    :func:`balanced_capture_assignment` is used.
    """
    assign = assignment or balanced_capture_assignment(env)

    def policy(env: CCSEnv) -> dict[str, list]:
        state = env.simulator.state
        acts: list[int] = []
        for i, vessel_id in enumerate(env.vessel_ids):
            mask = env.vessel_action_mask()[i]
            cargo = state.entity_inventory_t.get(vessel_id, 0.0)
            vessel = env.network.entities[vessel_id]
            berth = state.vessel_berths.get(vessel_id)
            mine = [e for e in env.emitter_ids if assign.get(e) == vessel_id]
            if berth in env.terminal_ids and cargo > _EPS:
                acts.append(VESSEL_WAIT); continue
            if mask[VESSEL_GO_TERMINAL] and cargo >= vessel.capacity_t - _EPS:
                acts.append(VESSEL_GO_TERMINAL); continue
            if berth in mine and cargo < vessel.capacity_t - _EPS and _emitter_supply_score(env, str(berth)) > _EPS:
                acts.append(VESSEL_WAIT); continue
            best = None
            for e in mine:
                a = env.vessel_go_emitter_action(e)
                if not mask[a]:
                    continue
                sc = _emitter_supply_score(env, e)
                if best is None or sc > best[0]:
                    best = (sc, a)
            if best is not None:
                acts.append(best[1]); continue
            acts.append(VESSEL_GO_TERMINAL if (mask[VESSEL_GO_TERMINAL] and cargo > _EPS) else VESSEL_WAIT)
        return _with_legacy_automatic_wells(env, acts)

    return policy


def _with_legacy_automatic_wells(
    env: CCSEnv,
    vessel_actions: list[int],
) -> dict[str, list]:
    action = {"vessels": list(vessel_actions)}
    if not env.automatic_well_control:
        action["wells"] = env.automatic_well_rate_indices()
    return action
