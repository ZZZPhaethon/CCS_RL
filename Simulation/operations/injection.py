"""Calculate well capacities and apply CO₂ injection into reservoirs.

计算注入井能力，并执行向储层注入二氧化碳的操作。
"""

from __future__ import annotations

from dataclasses import dataclass

from ..scenario_generation.disturbance_resolver import well_max_injection_tph
from ..entities.state import PhysicalState
from ..entities.storage import InjectionWell, Reservoir
from .pressure_limits import bottomhole_pressure_limited_rate_tph


@dataclass(frozen=True)
class WellCapacityBreakdown:
    well_capacity_t: float
    reservoir_capacity_t: float
    bottomhole_pressure_capacity_t: float
    remaining_capacity_t: float

    @property
    def bottomhole_pressure_limited(self) -> bool:
        physical_capacity_t = min(self.well_capacity_t, self.reservoir_capacity_t)
        return self.bottomhole_pressure_capacity_t < physical_capacity_t - 1e-9


def inject_to_well(
    network,
    state: PhysicalState,
    flows: dict[tuple[str, str], float],
    source_id: str,
    well_id: str,
    amount_t: float,
) -> None:
    network._move(state, flows, source_id, well_id, amount_t)
    state.last_injection_flow_tph[well_id] = (
        state.last_injection_flow_tph.get(well_id, 0.0)
        + amount_t / network.time_step_hours
    )
    reservoir_id = network._single_downstream_of_type(well_id, Reservoir)
    if reservoir_id is not None:
        network._move(state, flows, well_id, reservoir_id, amount_t)


def well_remaining_capacity(network, well_id: str, state: PhysicalState) -> float:
    return well_capacity_breakdown(network, well_id, state).remaining_capacity_t


def well_capacity_breakdown(network, well_id: str, state: PhysicalState) -> WellCapacityBreakdown:
    well = network.entities[well_id]
    assert isinstance(well, InjectionWell)
    effective_max_tph = well_max_injection_tph(state, well)
    if effective_max_tph <= 0.0:
        return WellCapacityBreakdown(0.0, 0.0, 0.0, 0.0)
    well_capacity_t = effective_max_tph * network.time_step_hours
    reservoir_id = network._single_downstream_of_type(well_id, Reservoir)
    if reservoir_id is None:
        return WellCapacityBreakdown(
            well_capacity_t,
            well_capacity_t,
            well_capacity_t,
            well_capacity_t,
        )
    reservoir_capacity_t = _reservoir_remaining_capacity(network, reservoir_id, state)
    physical_capacity_t = min(well_capacity_t, reservoir_capacity_t)
    pressure_limited_rate_tph = bottomhole_pressure_limited_rate_tph(
        network,
        state,
        well_id,
        physical_capacity_t / network.time_step_hours,
    )
    bottomhole_pressure_capacity_t = pressure_limited_rate_tph * network.time_step_hours
    remaining_capacity_t = min(physical_capacity_t, bottomhole_pressure_capacity_t)
    return WellCapacityBreakdown(
        well_capacity_t=well_capacity_t,
        reservoir_capacity_t=reservoir_capacity_t,
        bottomhole_pressure_capacity_t=bottomhole_pressure_capacity_t,
        remaining_capacity_t=remaining_capacity_t,
    )


def _reservoir_remaining_capacity(network, reservoir_id: str, state: PhysicalState) -> float:
    reservoir = network.entities[reservoir_id]
    assert isinstance(reservoir, Reservoir)
    current_t = state.entity_inventory_t.get(reservoir_id, 0.0)
    limit_t = min(reservoir.storage_capacity_t, reservoir.pressure_limited_capacity_t())
    return max(0.0, limit_t - current_t)
