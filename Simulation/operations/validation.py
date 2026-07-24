"""Validate physical state, capacity, flow, pressure, and mass-balance limits.

校验物理状态、容量、流量、压力和质量守恒限制。
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from ..entities.emitter import Emitter
from ..entities.manifold import SubseaManifold
from ..entities.pipeline import Pipeline
from ..entities.state import PhysicalState, Violation
from ..entities.storage import InjectionWell, Reservoir
from ..entities.terminal import Terminal
from ..entities.vessel import Vessel
from ..scenario_generation.disturbance_resolver import well_max_injection_tph

if TYPE_CHECKING:
    from ..network import PhysicalNetwork


ABSOLUTE_TOLERANCE_T = 1e-6
RELATIVE_TOLERANCE = 1e-9


def validate_step(
    network: "PhysicalNetwork",
    state: PhysicalState,
    flows_t: dict[tuple[str, str], float],
    *,
    initial_mass_t: float,
    generated_t: float,
) -> list[Violation]:
    """Return invariant violations detected after one physical network step.

    The validator does not modify state or clip flows. Operations are
    responsible for enforcing requests; this function is the final audit layer
    that detects unexpected violations and prevents them from being silent.

    返回物理网络推进一个时间步后检测到的不变量违规。
    校验器不会修改状态或裁剪流量。各操作函数负责执行约束；本函数作为最终审计层，
    用于发现意外违规，避免其被静默忽略。
    """
    violations: list[Violation] = []
    _validate_inventories(network, state, violations)
    _validate_transport_limits(network, state, flows_t, violations)
    _validate_mass_balance(
        state,
        violations,
        initial_mass_t=initial_mass_t,
        generated_t=generated_t,
    )
    return violations


def _validate_inventories(
    network: "PhysicalNetwork",
    state: PhysicalState,
    violations: list[Violation],
) -> None:
    for entity_id, inventory_t in state.entity_inventory_t.items():
        entity = network.entities.get(entity_id)
        if entity is None:
            violations.append(
                Violation(
                    "unknown_inventory_entity",
                    entity_id,
                    inventory_t,
                    0.0,
                    abs(inventory_t),
                    "Inventory is recorded for an entity that is not in the network.",
                )
            )
            continue
        if not math.isfinite(inventory_t):
            violations.append(
                Violation(
                    "non_finite_inventory",
                    entity_id,
                    inventory_t,
                    0.0,
                    abs(inventory_t),
                    "Entity inventory must be a finite quantity.",
                )
            )
            continue
        if inventory_t < -ABSOLUTE_TOLERANCE_T:
            violations.append(
                Violation(
                    "negative_inventory",
                    entity_id,
                    inventory_t,
                    0.0,
                    -inventory_t,
                    "Entity inventory cannot be negative.",
                )
            )
        capacity_t = _inventory_capacity_t(entity)
        if capacity_t is not None and inventory_t > capacity_t + ABSOLUTE_TOLERANCE_T:
            violations.append(
                Violation(
                    "inventory_capacity_exceeded",
                    entity_id,
                    inventory_t,
                    capacity_t,
                    inventory_t - capacity_t,
                    "Entity inventory exceeds its physical storage capacity.",
                )
            )
        if isinstance(entity, Reservoir):
            pressure_bar = entity.pressure_bar(max(0.0, inventory_t))
            if pressure_bar > entity.max_pressure_bar + ABSOLUTE_TOLERANCE_T:
                violations.append(
                    Violation(
                        "reservoir_pressure_exceeded",
                        entity_id,
                        pressure_bar,
                        entity.max_pressure_bar,
                        pressure_bar - entity.max_pressure_bar,
                        "Reservoir pressure exceeds the configured maximum pressure.",
                    )
                )
        if isinstance(entity, (Pipeline, SubseaManifold, InjectionWell)):
            if abs(inventory_t) > ABSOLUTE_TOLERANCE_T:
                violations.append(
                    Violation(
                        "unsettled_transit_inventory",
                        entity_id,
                        inventory_t,
                        0.0,
                        abs(inventory_t),
                        "Non-buffer transport entities must settle inventory within a step.",
                    )
                )


def _inventory_capacity_t(
    entity: object,
) -> float | None:
    if isinstance(entity, Emitter):
        return entity.buffer_capacity_t
    if isinstance(entity, Vessel):
        return entity.capacity_t
    if isinstance(entity, Terminal):
        return entity.storage_capacity_t
    if isinstance(entity, Reservoir):
        return min(entity.storage_capacity_t, entity.pressure_limited_capacity_t())
    return None


def _validate_transport_limits(
    network: "PhysicalNetwork",
    state: PhysicalState,
    flows_t: dict[tuple[str, str], float],
    violations: list[Violation],
) -> None:
    for (source_id, target_id), amount_t in flows_t.items():
        if not math.isfinite(amount_t) or amount_t < -ABSOLUTE_TOLERANCE_T:
            violations.append(
                Violation(
                    "invalid_flow",
                    f"{source_id}->{target_id}",
                    amount_t,
                    0.0,
                    abs(amount_t),
                    "Transport flow must be a finite non-negative quantity.",
                )
            )
            continue
        connection_limit_t = _connection_limit_t(network, source_id, target_id)
        if connection_limit_t is not None and amount_t > connection_limit_t + ABSOLUTE_TOLERANCE_T:
            violations.append(
                Violation(
                    "connection_flow_exceeded",
                    f"{source_id}->{target_id}",
                    amount_t,
                    connection_limit_t,
                    amount_t - connection_limit_t,
                    "Transport flow exceeds the connection capacity.",
                )
            )

    for pipeline_id, pipeline in network._entities_of_type(Pipeline).items():
        actual_tph = state.last_pipeline_flow_tph.get(pipeline_id, 0.0)
        if actual_tph > pipeline.max_flow_tph + ABSOLUTE_TOLERANCE_T:
            violations.append(
                Violation(
                    "pipeline_flow_exceeded",
                    pipeline_id,
                    actual_tph,
                    pipeline.max_flow_tph,
                    actual_tph - pipeline.max_flow_tph,
                    "Pipeline flow exceeds its maximum throughput.",
                )
            )

    for well_id, well in network._entities_of_type(InjectionWell).items():
        actual_tph = state.last_injection_flow_tph.get(well_id, 0.0)
        effective_limit_tph = well_max_injection_tph(state, well)
        if actual_tph > effective_limit_tph + ABSOLUTE_TOLERANCE_T:
            violations.append(
                Violation(
                    "well_flow_exceeded",
                    well_id,
                    actual_tph,
                    effective_limit_tph,
                    actual_tph - effective_limit_tph,
                    "Injection flow exceeds the well's effective capacity.",
                )
            )

    for manifold_id, manifold in network._entities_of_type(SubseaManifold).items():
        incoming_t = sum(
            amount_t
            for (_source_id, target_id), amount_t in flows_t.items()
            if target_id == manifold_id
        )
        limit_t = (
            manifold.max_flow_tph * network.time_step_hours
            if manifold.available
            else 0.0
        )
        if incoming_t > limit_t + ABSOLUTE_TOLERANCE_T:
            violations.append(
                Violation(
                    "manifold_flow_exceeded",
                    manifold_id,
                    incoming_t,
                    limit_t,
                    incoming_t - limit_t,
                    "Manifold throughput exceeds its effective capacity.",
                )
            )


def _connection_limit_t(
    network: "PhysicalNetwork",
    source_id: str,
    target_id: str,
) -> float | None:
    for connection in network.connections:
        if connection.source == source_id and connection.target == target_id:
            if connection.max_flow_tph is None:
                return None
            return connection.max_flow_tph * network.time_step_hours
    return None


def _validate_mass_balance(
    state: PhysicalState,
    violations: list[Violation],
    *,
    initial_mass_t: float,
    generated_t: float,
) -> None:
    final_mass_t = sum(state.entity_inventory_t.values())
    residual_t = final_mass_t - initial_mass_t - generated_t
    scale_t = max(abs(initial_mass_t), abs(final_mass_t), abs(generated_t), 1.0)
    tolerance_t = max(ABSOLUTE_TOLERANCE_T, RELATIVE_TOLERANCE * scale_t)
    if not math.isfinite(residual_t) or abs(residual_t) > tolerance_t:
        violations.append(
            Violation(
                "mass_balance_error",
                "network",
                final_mass_t - initial_mass_t,
                generated_t,
                abs(residual_t),
                "Mass balance residual exceeds the numerical tolerance.",
            )
        )
