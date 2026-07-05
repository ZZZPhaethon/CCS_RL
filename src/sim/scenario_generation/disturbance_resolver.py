"""Resolve the current timestep's effective values after disturbances.

This module does not generate disturbance scenarios. It is the lookup layer used
after a :class:`Scenario` has written the current step's disturbance values into
:class:`PhysicalState`.

The physical entities (:class:`Emitter`, :class:`InjectionWell`, ...) store
nominal values. These helpers return the effective values for the current step:
use the value on :class:`PhysicalState` when a disturbance override exists,
otherwise fall back to the entity's nominal value.
"""

from __future__ import annotations

from ..entities.emitter import Emitter
from ..entities.state import PhysicalState
from ..entities.storage import InjectionWell
from ..entities.terminal import Terminal


def emitter_availability(state: PhysicalState, emitter: Emitter) -> float:
    """Capture multiplier for an emitter; values above 1 model over-profile output."""
    value = state.emitter_availability.get(emitter.entity_id, emitter.availability)
    return max(0.0, value)


def well_is_available(state: PhysicalState, well: InjectionWell) -> bool:
    """Whether an injection well can accept flow this step."""
    return bool(state.well_available.get(well.entity_id, well.available))


def well_injectivity_factor(state: PhysicalState, well: InjectionWell) -> float:
    """Non-negative multiplier applied to a well's nominal max injection rate.

    Defaults to ``1.0`` (nominal). Values below ``1.0`` model injectivity
    decline or partial deratings; ``0.0`` means no injection capacity.
    """
    return max(0.0, float(state.injectivity_factor.get(well.entity_id, 1.0)))


def well_max_injection_tph(state: PhysicalState, well: InjectionWell) -> float:
    """Effective per-hour injection ceiling after availability and injectivity."""
    if not well_is_available(state, well):
        return 0.0
    return well.max_injection_tph * well_injectivity_factor(state, well)


def vessel_speed_factor(state: PhysicalState, vessel_id: str) -> float:
    """Non-negative multiplier on a vessel's nominal sailing speed.

    Defaults to ``1.0``. Values below ``1.0`` model weather-induced slowdowns.
    """
    return max(0.0, float(state.vessel_speed_factor.get(vessel_id, 1.0)))


def leg_speed_factor(
    state: PhysicalState,
    origin_id: str,
    destination_id: str,
    *,
    fallback: float = 1.0,
) -> float:
    """Non-negative speed multiplier for a specific sailing leg."""
    leg_id = f"{origin_id}->{destination_id}"
    return max(0.0, float(state.leg_speed_factor.get(leg_id, fallback)))


def terminal_berth_count(state: PhysicalState, terminal: Terminal) -> int:
    """Number of usable berths after any explicit state override."""
    value = state.berth_count_override.get(terminal.entity_id, terminal.berth_count)
    return max(0, int(value))
