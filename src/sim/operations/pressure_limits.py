from __future__ import annotations

from ..entities.state import PhysicalState
from ..entities.storage import InjectionWell, Reservoir
from ..line_source import (
    multiwell_variable_rate_bottomhole_pressures_bar,
    variable_rate_bottomhole_pressure_bar,
)

HOURS_PER_YEAR = 365.25 * 24.0
WELL_RATE_LEVELS_MTPA = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5)


def mtpa_to_tph(rate_mtpa: float) -> float:
    return rate_mtpa * 1_000_000.0 / HOURS_PER_YEAR


def tph_to_mtpa(rate_tph: float) -> float:
    return rate_tph * HOURS_PER_YEAR / 1_000_000.0


def projected_bottomhole_pressure_bar(
    network,
    state: PhysicalState,
    well_id: str,
    candidate_rate_tph: float,
    *,
    evaluation_time_h: float | None = None,
    interval_start_h: float | None = None,
) -> float | None:
    reservoir_id = network._single_downstream_of_type(well_id, Reservoir)
    if reservoir_id is None:
        return None
    reservoir = network.entities[reservoir_id]
    assert isinstance(reservoir, Reservoir)
    parameters = reservoir.line_source_parameters
    if parameters is None:
        return None

    evaluation_h = state.time_h if evaluation_time_h is None else evaluation_time_h
    if evaluation_h <= 0.0:
        return parameters.initial_pressure_bar
    start_h = (
        evaluation_h - network.time_step_hours
        if interval_start_h is None
        else interval_start_h
    )
    elapsed_days = evaluation_h / 24.0

    upstream_wells = network._upstream_of_type(reservoir.entity_id, InjectionWell)
    if reservoir.line_source_well_distances_m and len(upstream_wells) > 1:
        histories = {
            upstream_well_id: _history_with_candidate_rate(
                state,
                upstream_well_id,
                (
                    candidate_rate_tph
                    if upstream_well_id == well_id
                    else state.last_injection_flow_tph.get(upstream_well_id, 0.0)
                ),
                interval_start_h=start_h,
            )
            for upstream_well_id in upstream_wells
        }
        return multiwell_variable_rate_bottomhole_pressures_bar(
            parameters,
            histories,
            elapsed_days=elapsed_days,
            well_distances_m=reservoir.line_source_well_distances_m,
        )[well_id]

    return variable_rate_bottomhole_pressure_bar(
        parameters,
        _history_with_candidate_rate(
            state,
            well_id,
            candidate_rate_tph,
            interval_start_h=start_h,
        ),
        elapsed_days=elapsed_days,
    )


def bottomhole_pressure_limited_rate_tph(
    network,
    state: PhysicalState,
    well_id: str,
    physical_max_rate_tph: float,
    *,
    evaluation_time_h: float | None = None,
    interval_start_h: float | None = None,
) -> float:
    reservoir_id = network._single_downstream_of_type(well_id, Reservoir)
    if reservoir_id is None:
        return max(0.0, physical_max_rate_tph)
    reservoir = network.entities[reservoir_id]
    assert isinstance(reservoir, Reservoir)
    if (
        reservoir.line_source_parameters is None
        or reservoir.well_bottomhole_pressure_limit_bar is None
    ):
        return max(0.0, physical_max_rate_tph)

    max_rate_tph = max(0.0, physical_max_rate_tph)
    if max_rate_tph <= 0.0:
        return 0.0

    limit_bar = reservoir.well_bottomhole_pressure_limit_bar
    base_pressure_bar = projected_bottomhole_pressure_bar(
        network,
        state,
        well_id,
        0.0,
        evaluation_time_h=evaluation_time_h,
        interval_start_h=interval_start_h,
    )
    if base_pressure_bar is None:
        return max_rate_tph
    if base_pressure_bar >= limit_bar:
        return 0.0

    max_pressure_bar = projected_bottomhole_pressure_bar(
        network,
        state,
        well_id,
        max_rate_tph,
        evaluation_time_h=evaluation_time_h,
        interval_start_h=interval_start_h,
    )
    if max_pressure_bar is None or max_pressure_bar <= limit_bar:
        return max_rate_tph

    slope_bar_per_tph = (max_pressure_bar - base_pressure_bar) / max_rate_tph
    if slope_bar_per_tph <= 0.0:
        return max_rate_tph
    return max(0.0, min(max_rate_tph, (limit_bar - base_pressure_bar) / slope_bar_per_tph))


def pressure_limited_rate_level_mask(
    network,
    state: PhysicalState,
    well_id: str,
    *,
    rate_levels_mtpa: tuple[float, ...] = WELL_RATE_LEVELS_MTPA,
    physical_max_rate_tph: float | None = None,
    evaluation_time_h: float | None = None,
    interval_start_h: float | None = None,
) -> tuple[bool, ...]:
    if physical_max_rate_tph is None:
        well = network.entities[well_id]
        assert isinstance(well, InjectionWell)
        physical_max_rate_tph = well.max_injection_tph
    pressure_max_rate_tph = bottomhole_pressure_limited_rate_tph(
        network,
        state,
        well_id,
        physical_max_rate_tph,
        evaluation_time_h=evaluation_time_h,
        interval_start_h=interval_start_h,
    )
    feasible_rate_tph = min(max(0.0, physical_max_rate_tph), pressure_max_rate_tph)
    return tuple(
        mtpa_to_tph(rate_mtpa) <= feasible_rate_tph + 1e-9
        for rate_mtpa in rate_levels_mtpa
    )


def _history_with_candidate_rate(
    state: PhysicalState,
    well_id: str,
    candidate_rate_tph: float,
    *,
    interval_start_h: float,
) -> list[tuple[float, float]]:
    history_tph = list(state.injection_rate_history_tph.get(well_id, []))
    if history_tph and abs(history_tph[-1][0] - interval_start_h) <= 1e-12:
        history_tph[-1] = (interval_start_h, candidate_rate_tph)
    elif history_tph or abs(candidate_rate_tph) > 1e-12:
        history_tph.append((interval_start_h, candidate_rate_tph))
    return [
        (start_h / 24.0, tph_to_mtpa(rate_tph))
        for start_h, rate_tph in history_tph
    ]
