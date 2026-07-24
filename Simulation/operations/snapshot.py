"""Build diagnostic snapshots of network state, flows, and pressure limits.

构建网络状态、流量和压力限制的诊断快照。
"""

from __future__ import annotations

from dataclasses import asdict

from ..scenario_generation.disturbance_resolver import (
    emitter_availability,
    well_injectivity_factor,
    well_is_available,
    well_max_injection_tph,
)
from ..entities.emitter import Emitter
from ..entities.pipeline import Pipeline
from ..entities.state import PhysicalState
from ..entities.storage import InjectionWell, Reservoir
from ..entities.line_source import (
    bottomhole_pressure_bar,
    multiwell_bottomhole_pressures_bar,
    multiwell_variable_rate_bottomhole_pressures_bar,
    pressure_at_radius_bar,
    variable_rate_bottomhole_pressure_bar,
    variable_rate_pressure_at_radius_bar,
)
from .pressure_limits import (
    WELL_RATE_LEVELS_MTPA,
    bottomhole_pressure_limited_rate_tph,
    line_source_parameters_for_state,
    pressure_limited_rate_level_mask,
    tph_to_mtpa,
)

HOURS_PER_YEAR = 365.25 * 24.0


def snapshot_network(network, state: PhysicalState) -> dict[str, object]:
    return {
        "time_h": state.time_h,
        "time_step_hours": network.time_step_hours,
        "entities": {
            entity_id: _snapshot_entity(network, entity, state)
            for entity_id, entity in network.entities.items()
        },
        "connections": [asdict(connection) for connection in network.connections],
    }


def _snapshot_entity(network, entity, state: PhysicalState) -> dict[str, object]:
    inventory_t = state.entity_inventory_t.get(entity.entity_id, 0.0)
    parameters = asdict(entity)
    if isinstance(entity, Emitter):
        parameters.pop("hourly_capture_profile_tph", None)
    snapshot: dict[str, object] = {
        "type": type(entity).__name__,
        "parameters": parameters,
        "inventory_t": inventory_t,
    }
    if isinstance(entity, Emitter):
        snapshot["capture_rate_tph"] = state.last_capture_tph.get(entity.entity_id, 0.0)
        snapshot["vent_rate_tph"] = state.last_vent_tph.get(entity.entity_id, 0.0)
        snapshot["cumulative_vent_t"] = state.cumulative_vent_t.get(entity.entity_id, 0.0)
        snapshot["effective_availability"] = emitter_availability(state, entity)
    if isinstance(entity, Pipeline):
        snapshot["pipeline_flow_rate_tph"] = state.last_pipeline_flow_tph.get(entity.entity_id, 0.0)
    if isinstance(entity, InjectionWell):
        snapshot["injection_rate_tph"] = state.last_injection_flow_tph.get(entity.entity_id, 0.0)
        snapshot["effective_available"] = well_is_available(state, entity)
        snapshot["injectivity_factor"] = well_injectivity_factor(state, entity)
        snapshot["effective_max_injection_tph"] = well_max_injection_tph(state, entity)
        _add_line_source_well_snapshot(network, snapshot, entity, state)
    if isinstance(entity, Reservoir):
        reservoir_pressure_bar = entity.pressure_bar(inventory_t)
        reservoir_pressure_margin_bar = entity.pressure_margin_bar(inventory_t)
        snapshot["pressure_bar"] = reservoir_pressure_bar
        snapshot["pressure_margin_bar"] = reservoir_pressure_margin_bar
        snapshot["reservoir_average_pressure_bar"] = reservoir_pressure_bar
        snapshot["reservoir_average_pressure_margin_bar"] = reservoir_pressure_margin_bar
        snapshot["reservoir_average_pressure_at_capacity_bar"] = entity.pressure_at_capacity_bar
        snapshot["reservoir_average_pressure_limit_bar"] = entity.max_pressure_bar
        snapshot["reservoir_pressure_model"] = entity.reservoir_pressure_model
        snapshot["seawater_depth_m"] = entity.seawater_depth_m
        snapshot["well_fracture_gradient_psi_per_ft"] = entity.well_fracture_gradient_psi_per_ft
        snapshot["well_fracture_gradient_reference_depth_m"] = (
            entity.well_fracture_gradient_reference_depth_m
        )
        snapshot["well_fracture_pressure_bar"] = entity.well_fracture_pressure_bar
        snapshot["well_bottomhole_pressure_safety_factor"] = (
            entity.well_bottomhole_pressure_safety_factor
        )
        snapshot["well_bottomhole_pressure_limit_bar"] = entity.well_bottomhole_pressure_limit_bar
        snapshot["fill_fraction"] = inventory_t / entity.storage_capacity_t
        _add_line_source_reservoir_snapshot(network, snapshot, entity, state)
    return snapshot


def _add_line_source_well_snapshot(
    network,
    snapshot: dict[str, object],
    well: InjectionWell,
    state: PhysicalState,
) -> None:
    reservoir_id = network._single_downstream_of_type(well.entity_id, Reservoir)
    if reservoir_id is None:
        return
    reservoir = network.entities[reservoir_id]
    assert isinstance(reservoir, Reservoir)
    parameters = line_source_parameters_for_state(reservoir, state)
    if parameters is None:
        return
    rate_tph = state.last_injection_flow_tph.get(well.entity_id, 0.0)
    elapsed_days = state.time_h / 24.0
    pressure_bar = parameters.initial_pressure_bar
    interference_delta_bar = 0.0
    if elapsed_days > 0.0:
        rate_history = _well_rate_history_mtpa(state, well.entity_id)
        if rate_history:
            self_pressure_bar = variable_rate_bottomhole_pressure_bar(
                parameters,
                rate_history,
                elapsed_days=elapsed_days,
            )
        else:
            self_pressure_bar = bottomhole_pressure_bar(
                parameters,
                _tph_to_mtpa(rate_tph),
                elapsed_days=elapsed_days,
            )
        pressure_bar = self_pressure_bar
        if reservoir.line_source_well_distances_m:
            well_histories_mtpa = {
                well_id: _well_rate_history_mtpa(state, well_id)
                for well_id in network._upstream_of_type(reservoir.entity_id, InjectionWell)
            }
            if any(well_histories_mtpa.values()):
                pressure_bar = multiwell_variable_rate_bottomhole_pressures_bar(
                    parameters,
                    well_histories_mtpa,
                    elapsed_days=elapsed_days,
                    well_distances_m=reservoir.line_source_well_distances_m,
                ).get(well.entity_id, self_pressure_bar)
            else:
                well_rates_mtpa = {
                    well_id: _tph_to_mtpa(state.last_injection_flow_tph.get(well_id, 0.0))
                    for well_id in network._upstream_of_type(reservoir.entity_id, InjectionWell)
                }
                pressure_bar = multiwell_bottomhole_pressures_bar(
                    parameters,
                    well_rates_mtpa,
                    elapsed_days=elapsed_days,
                    well_distances_m=reservoir.line_source_well_distances_m,
                ).get(well.entity_id, self_pressure_bar)
            interference_delta_bar = pressure_bar - self_pressure_bar
    snapshot["line_source_rate_tph"] = rate_tph
    snapshot["line_source_elapsed_days"] = elapsed_days
    snapshot["bottomhole_pressure_bar"] = pressure_bar
    snapshot["bottomhole_pressure_delta_bar"] = pressure_bar - parameters.initial_pressure_bar
    snapshot["well_bottomhole_pressure_limit_bar"] = reservoir.well_bottomhole_pressure_limit_bar
    if reservoir.well_bottomhole_pressure_limit_bar is not None:
        snapshot["well_bottomhole_pressure_margin_bar"] = (
            reservoir.well_bottomhole_pressure_limit_bar - pressure_bar
        )
    snapshot["well_fracture_pressure_bar"] = reservoir.well_fracture_pressure_bar
    snapshot["well_fracture_gradient_psi_per_ft"] = reservoir.well_fracture_gradient_psi_per_ft
    next_step_evaluation_h = state.time_h + network.time_step_hours
    next_step_start_h = state.time_h
    effective_max_tph = well_max_injection_tph(state, well)
    next_step_pressure_limited_rate_tph = bottomhole_pressure_limited_rate_tph(
        network,
        state,
        well.entity_id,
        effective_max_tph,
        evaluation_time_h=next_step_evaluation_h,
        interval_start_h=next_step_start_h,
    )
    snapshot["well_rate_levels_mtpa"] = list(WELL_RATE_LEVELS_MTPA)
    snapshot["next_step_pressure_limited_max_rate_tph"] = next_step_pressure_limited_rate_tph
    snapshot["next_step_pressure_limited_max_rate_mtpa"] = tph_to_mtpa(
        next_step_pressure_limited_rate_tph
    )
    snapshot["next_step_well_rate_action_mask"] = list(
        pressure_limited_rate_level_mask(
            network,
            state,
            well.entity_id,
            physical_max_rate_tph=effective_max_tph,
            evaluation_time_h=next_step_evaluation_h,
            interval_start_h=next_step_start_h,
        )
    )
    snapshot["line_source_interference_delta_bar"] = interference_delta_bar
    snapshot["line_source_rate_history_tph"] = state.injection_rate_history_tph.get(
        well.entity_id,
        [],
    )


def _add_line_source_reservoir_snapshot(
    network,
    snapshot: dict[str, object],
    reservoir: Reservoir,
    state: PhysicalState,
) -> None:
    parameters = line_source_parameters_for_state(reservoir, state)
    if parameters is None:
        return
    elapsed_days = state.time_h / 24.0
    upstream_wells = network._upstream_of_type(reservoir.entity_id, InjectionWell)
    well_rates_tph = {
        well_id: state.last_injection_flow_tph.get(well_id, 0.0)
        for well_id in upstream_wells
    }
    well_histories_mtpa = {
        well_id: _well_rate_history_mtpa(state, well_id)
        for well_id in upstream_wells
    }
    pressure_by_radius: dict[str, float] = {}
    delta_by_radius: dict[str, float] = {}
    for radius_m in reservoir.line_source_observation_radii_m:
        total_delta_bar = 0.0
        if elapsed_days > 0.0:
            if any(well_histories_mtpa.values()):
                for rate_history in well_histories_mtpa.values():
                    pressure_bar = variable_rate_pressure_at_radius_bar(
                        parameters,
                        rate_history,
                        elapsed_days=elapsed_days,
                        radius_m=radius_m,
                    )
                    total_delta_bar += pressure_bar - parameters.initial_pressure_bar
            else:
                for rate_tph in well_rates_tph.values():
                    pressure_bar = pressure_at_radius_bar(
                        parameters,
                        _tph_to_mtpa(rate_tph),
                        elapsed_days=elapsed_days,
                        radius_m=radius_m,
                    )
                    total_delta_bar += pressure_bar - parameters.initial_pressure_bar
        key = str(float(radius_m))
        delta_by_radius[key] = total_delta_bar
        pressure_by_radius[key] = parameters.initial_pressure_bar + total_delta_bar
    snapshot["line_source_elapsed_days"] = elapsed_days
    snapshot["line_source_well_rates_tph"] = well_rates_tph
    snapshot["line_source_well_rate_history_tph"] = {
        well_id: state.injection_rate_history_tph.get(well_id, [])
        for well_id in upstream_wells
    }
    snapshot["line_source_pressure_bar_by_radius_m"] = pressure_by_radius
    snapshot["line_source_delta_bar_by_radius_m"] = delta_by_radius


def _well_rate_history_mtpa(state: PhysicalState, well_id: str) -> list[tuple[float, float]]:
    return [
        (start_h / 24.0, _tph_to_mtpa(rate_tph))
        for start_h, rate_tph in state.injection_rate_history_tph.get(well_id, [])
    ]


def _tph_to_mtpa(rate_tph: float) -> float:
    return rate_tph * HOURS_PER_YEAR / 1_000_000.0
