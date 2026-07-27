from __future__ import annotations

from ..entities.manifold import SubseaManifold
from ..entities.pipeline import Pipeline
from ..entities.state import PhysicalState, Violation
from ..entities.storage import InjectionWell
from .injection import inject_to_well, well_capacity_breakdown, well_remaining_capacity

HOURS_PER_YEAR = 365.25 * 24.0


def project_pipeline_outflow(
    network,
    terminal_id: str,
    pipeline: Pipeline,
    state: PhysicalState,
    actions: dict[str, dict[str, object]],
    violations: list[Violation],
    supply_limit_t: float,
) -> float:
    requested_t = actions.get(pipeline.entity_id, {}).get(
        "flow_tph",
        actions.get(terminal_id, {}).get("flow_tph", 0.0),
    ) * network.time_step_hours
    pipeline_capacity_t = pipeline.max_flow_tph * network.time_step_hours
    well_capacity_t = pipeline_injection_capacity(network, pipeline.entity_id, state)
    annual_capacity_t = pipeline_annual_remaining_capacity_t(network, pipeline, state)
    minimum_stable_t = pipeline_minimum_stable_injection_t(
        network,
        pipeline.entity_id,
        state,
    )
    actual_t = min(
        requested_t,
        pipeline_capacity_t,
        well_capacity_t,
        annual_capacity_t,
        max(0.0, supply_limit_t),
    )
    if 0.0 < actual_t < minimum_stable_t - 1e-9:
        actual_t = 0.0
        violations.append(
            Violation(
                "minimum_stable_injection_not_met",
                pipeline.entity_id,
                requested_t,
                0.0,
                requested_t,
                "Pipeline flow is below the minimum stable injection quantity.",
            )
        )
    state.last_pipeline_flow_tph[pipeline.entity_id] = actual_t / network.time_step_hours
    if actual_t < requested_t:
        violations.append(
            Violation(
                "flow_clipped",
                pipeline.entity_id,
                requested_t,
                actual_t,
                requested_t - actual_t,
                "Pipeline flow request clipped by pipeline limit, well capacity, or available supply.",
            )
        )
        physical_injection_capacity_t, pressure_limited_well_ids = (
            _pipeline_capacity_without_bottomhole_pressure(network, pipeline.entity_id, state)
        )
        achievable_without_bottomhole_pressure_t = min(
            requested_t,
            pipeline_capacity_t,
            physical_injection_capacity_t,
            max(0.0, supply_limit_t),
        )
        if actual_t < achievable_without_bottomhole_pressure_t - 1e-9:
            for well_id in pressure_limited_well_ids:
                violations.append(
                    Violation(
                        "bottomhole_pressure_clipped",
                        well_id,
                        requested_t,
                        actual_t,
                        requested_t - actual_t,
                        "Injection request clipped by bottomhole pressure limit.",
                    )
                )
    return actual_t


def pipeline_annual_remaining_capacity_t(
    network,
    pipeline: Pipeline,
    state: PhysicalState,
) -> float:
    if pipeline.annual_capacity_tpy is None:
        return float("inf")
    interval_start_h = state.time_h - network.time_step_hours
    window_start_h = interval_start_h - HOURS_PER_YEAR
    used_t = sum(
        amount_t
        for start_h, amount_t in state.pipeline_flow_history_t.get(
            pipeline.entity_id,
            [],
        )
        if start_h > window_start_h
    )
    return max(0.0, pipeline.annual_capacity_tpy - used_t)


def pipeline_minimum_stable_injection_t(
    network,
    pipeline_id: str,
    state: PhysicalState,
) -> float:
    well_ids = list(network._downstream_of_type(pipeline_id, InjectionWell))
    for manifold_id in network._downstream_of_type(pipeline_id, SubseaManifold):
        well_ids.extend(network._downstream_of_type(manifold_id, InjectionWell))
    minimums_t = [
        well.min_stable_injection_tph * network.time_step_hours
        for well_id in well_ids
        for well in [network.entities[well_id]]
        if isinstance(well, InjectionWell)
        and well_remaining_capacity(network, well_id, state)
        >= well.min_stable_injection_tph * network.time_step_hours - 1e-9
    ]
    return min(minimums_t, default=0.0)


def distribute_pipeline_outflow(
    network,
    state: PhysicalState,
    flows: dict[tuple[str, str], float],
    actions: dict[str, dict[str, object]],
    pipeline_id: str,
    outflow_t: float,
) -> None:
    remaining_t = outflow_t
    for manifold_id in network._downstream_of_type(pipeline_id, SubseaManifold):
        amount_t = min(remaining_t, manifold_remaining_capacity(network, manifold_id, state))
        if amount_t > 0.0:
            network._move(state, flows, pipeline_id, manifold_id, amount_t)
            _distribute_from_manifold(
                network,
                state,
                flows,
                actions,
                manifold_id,
                amount_t,
            )
            remaining_t -= amount_t
        if remaining_t <= 1e-12:
            return
    _distribute_to_wells(network, state, flows, pipeline_id, pipeline_id, remaining_t)


def pipeline_injection_capacity(network, pipeline_id: str, state: PhysicalState) -> float:
    direct_well_capacity_t = sum(
        well_remaining_capacity(network, well_id, state)
        for well_id in network._downstream_of_type(pipeline_id, InjectionWell)
    )
    manifold_capacity_t = sum(
        manifold_remaining_capacity(network, manifold_id, state)
        for manifold_id in network._downstream_of_type(pipeline_id, SubseaManifold)
    )
    return direct_well_capacity_t + manifold_capacity_t


def _pipeline_capacity_without_bottomhole_pressure(
    network,
    pipeline_id: str,
    state: PhysicalState,
) -> tuple[float, list[str]]:
    physical_capacity_t = 0.0
    pressure_limited_well_ids: list[str] = []
    for well_id in network._downstream_of_type(pipeline_id, InjectionWell):
        capacity = well_capacity_breakdown(network, well_id, state)
        physical_capacity_t += min(capacity.well_capacity_t, capacity.reservoir_capacity_t)
        if capacity.bottomhole_pressure_limited:
            pressure_limited_well_ids.append(well_id)

    for manifold_id in network._downstream_of_type(pipeline_id, SubseaManifold):
        manifold = network.entities[manifold_id]
        assert isinstance(manifold, SubseaManifold)
        if not manifold.available:
            continue
        capacities = [
            (well_id, well_capacity_breakdown(network, well_id, state))
            for well_id in network._downstream_of_type(manifold_id, InjectionWell)
        ]
        manifold_capacity_t = manifold.max_flow_tph * network.time_step_hours
        manifold_physical_capacity_t = min(
            manifold_capacity_t,
            sum(min(capacity.well_capacity_t, capacity.reservoir_capacity_t) for _, capacity in capacities),
        )
        manifold_pressure_capacity_t = min(
            manifold_capacity_t,
            sum(capacity.remaining_capacity_t for _, capacity in capacities),
        )
        physical_capacity_t += manifold_physical_capacity_t
        if manifold_pressure_capacity_t < manifold_physical_capacity_t - 1e-9:
            pressure_limited_well_ids.extend(
                well_id
                for well_id, capacity in capacities
                if capacity.bottomhole_pressure_limited
            )
    return physical_capacity_t, pressure_limited_well_ids


def manifold_remaining_capacity(network, manifold_id: str, state: PhysicalState) -> float:
    manifold = network.entities[manifold_id]
    assert isinstance(manifold, SubseaManifold)
    if not manifold.available:
        return 0.0
    manifold_capacity_t = manifold.max_flow_tph * network.time_step_hours
    well_capacity_t = sum(
        well_remaining_capacity(network, well_id, state)
        for well_id in network._downstream_of_type(manifold_id, InjectionWell)
    )
    return min(manifold_capacity_t, well_capacity_t)


def _distribute_from_manifold(
    network,
    state: PhysicalState,
    flows: dict[tuple[str, str], float],
    actions: dict[str, dict[str, object]],
    manifold_id: str,
    outflow_t: float,
) -> None:
    well_splits = actions.get(manifold_id, {}).get("well_splits")
    if isinstance(well_splits, dict) and well_splits:
        _distribute_to_wells_by_split(network, state, flows, manifold_id, outflow_t, well_splits)
        return
    _distribute_to_wells(network, state, flows, manifold_id, manifold_id, outflow_t)


def _distribute_to_wells_by_split(
    network,
    state: PhysicalState,
    flows: dict[tuple[str, str], float],
    source_id: str,
    outflow_t: float,
    well_splits: dict[str, float],
) -> None:
    remaining_t = outflow_t
    downstream_wells = network._downstream_of_type(source_id, InjectionWell)
    for well_id in downstream_wells:
        split = float(well_splits.get(well_id, 0.0))
        amount_t = min(outflow_t * split, well_remaining_capacity(network, well_id, state))
        if amount_t > 0.0:
            inject_to_well(network, state, flows, source_id, well_id, amount_t)
            remaining_t -= amount_t
    for well_id in downstream_wells:
        if remaining_t <= 1e-12:
            break
        amount_t = min(remaining_t, well_remaining_capacity(network, well_id, state))
        if amount_t > 0.0:
            inject_to_well(network, state, flows, source_id, well_id, amount_t)
            remaining_t -= amount_t


def _distribute_to_wells(
    network,
    state: PhysicalState,
    flows: dict[tuple[str, str], float],
    source_id: str,
    upstream_id: str,
    outflow_t: float,
) -> float:
    remaining_t = outflow_t
    for well_id in network._downstream_of_type(upstream_id, InjectionWell):
        amount_t = min(remaining_t, well_remaining_capacity(network, well_id, state))
        if amount_t > 0:
            inject_to_well(network, state, flows, source_id, well_id, amount_t)
            remaining_t -= amount_t
        if remaining_t <= 1e-12:
            break
    return remaining_t
