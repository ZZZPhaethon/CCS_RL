"""Compose entities and domain operations into a physical CCS network.

将实体和领域操作组合为完整的 CCS 物理网络。
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from .entities.emitter import Emitter
from .entities.manifold import SubseaManifold
from .entities.pipeline import Pipeline
from .entities.state import PhysicalState, StepResult, Violation
from .entities.storage import InjectionWell, Reservoir
from .entities.terminal import Terminal
from .entities.vessel import Vessel
from .operations.capture import apply_capture
from .operations.loading import apply_loading
from .operations.snapshot import snapshot_network
from .operations.transport import distribute_pipeline_outflow, project_pipeline_outflow
from .operations.unloading import project_terminal_unload, terminal_unload_request_capacity
from .operations.validation import validate_step

Entity = Emitter | Vessel | Terminal | Pipeline | SubseaManifold | InjectionWell | Reservoir


@dataclass(frozen=True)
class Connection:
    source: str
    target: str
    max_flow_tph: float | None = None


class PhysicalNetwork:
    """Graph of physical CCS entities with a one-step feasibility projection."""

    def __init__(self, time_step_hours: float = 1.0) -> None:
        if time_step_hours <= 0:
            raise ValueError("time_step_hours must be positive")
        self.time_step_hours = time_step_hours
        self.entities: dict[str, Entity] = {}
        self.connections: list[Connection] = []

    def add_entity(self, entity: Entity) -> None:
        """Add a validated physical entity with a unique identifier.

        添加经过校验且具有唯一标识符的物理实体。
        """
        if entity.entity_id in self.entities:
            raise ValueError(f"Duplicate entity id: {entity.entity_id}")
        self._validate_entity(entity)
        self.entities[entity.entity_id] = entity

    def connect(self, source: str, target: str, max_flow_tph: float | None = None) -> None:
        """Connect compatible physical entities with an optional flow limit.

        连接兼容的物理实体，并可选地指定流量上限。
        """
        self._require_entity(source)
        self._require_entity(target)
        self._validate_connection(source, target, max_flow_tph)
        if target not in self.downstream_of(source):
            self.connections.append(Connection(source, target, max_flow_tph))

    def disconnect(self, source: str, target: str) -> None:
        self.connections = [
            connection
            for connection in self.connections
            if not (connection.source == source and connection.target == target)
        ]

    def downstream_of(self, entity_id: str) -> list[str]:
        return [connection.target for connection in self.connections if connection.source == entity_id]

    def upstream_of(self, entity_id: str) -> list[str]:
        return [connection.source for connection in self.connections if connection.target == entity_id]

    def snapshot(self, state: PhysicalState) -> dict[str, object]:
        return snapshot_network(self, state)

    def step(self, state: PhysicalState, actions: dict[str, dict[str, object]] | None = None) -> StepResult:
        """Advance the network by one time step and audit physical invariants.

        推进网络一个时间步，并审计物理不变量。
        """
        self.validate_topology()
        actions = actions or {}
        next_state = state.copy()
        next_state.time_h += self.time_step_hours
        next_state.last_capture_tph = {
            emitter_id: 0.0 for emitter_id in self._entities_of_type(Emitter)
        }
        next_state.last_vent_tph = {
            emitter_id: 0.0 for emitter_id in self._entities_of_type(Emitter)
        }
        next_state.last_injection_flow_tph = {
            well_id: 0.0 for well_id in self._entities_of_type(InjectionWell)
        }
        flows: dict[tuple[str, str], float] = {}
        violations: list[Violation] = []
        initial_mass_t = sum(next_state.entity_inventory_t.values())
        generated_t = apply_capture(self, next_state, actions, violations)

        for terminal_id, terminal in self._entities_of_type(Terminal).items():
            pipeline_id = self._single_downstream_of_type(terminal_id, Pipeline)
            if pipeline_id is None:
                continue
            pipeline = self.entities[pipeline_id]
            assert isinstance(pipeline, Pipeline)
            potential_unload_t = terminal_unload_request_capacity(self, terminal, next_state, actions)
            outflow_t = project_pipeline_outflow(
                self,
                terminal_id,
                pipeline,
                next_state,
                actions,
                violations,
                supply_limit_t=next_state.entity_inventory_t.get(terminal_id, 0.0) + potential_unload_t,
            )

            unload_t = project_terminal_unload(
                self,
                terminal,
                outflow_t,
                next_state,
                actions,
                violations,
            )
            for vessel_id, amount_t in unload_t.items():
                self._move(next_state, flows, vessel_id, terminal_id, amount_t)

            if outflow_t > 0:
                self._move(next_state, flows, terminal_id, pipeline_id, outflow_t)
                distribute_pipeline_outflow(self, next_state, flows, actions, pipeline_id, outflow_t)

        apply_loading(self, next_state, actions, flows, violations)
        self._record_pipeline_flow_history(next_state)
        self._record_injection_rate_history(next_state)

        final_mass_t = sum(next_state.entity_inventory_t.values())
        mass_balance_error_t = final_mass_t - initial_mass_t - generated_t
        violations.extend(
            validate_step(
                self,
                next_state,
                flows,
                initial_mass_t=initial_mass_t,
                generated_t=generated_t,
            )
        )
        return StepResult(
            state=next_state,
            flows_t=flows,
            violations=violations,
            mass_balance_error_t=mass_balance_error_t,
        )

    def _move(
        self,
        state: PhysicalState,
        flows: dict[tuple[str, str], float],
        source: str,
        target: str,
        amount_t: float,
    ) -> None:
        if amount_t <= 0:
            return
        state.entity_inventory_t[source] = state.entity_inventory_t.get(source, 0.0) - amount_t
        state.entity_inventory_t[target] = state.entity_inventory_t.get(target, 0.0) + amount_t
        flows[(source, target)] = flows.get((source, target), 0.0) + amount_t

    def _entities_of_type(self, entity_type: type) -> dict[str, Entity]:
        return {
            entity_id: entity
            for entity_id, entity in self.entities.items()
            if isinstance(entity, entity_type)
        }

    def validate_topology(self) -> None:
        """Raise ``ValueError`` when entities or connections violate the model.

        当实体或连接不符合物理模型时引发 ``ValueError``。
        """
        for entity in self.entities.values():
            self._validate_entity(entity)
        seen_connections: set[tuple[str, str]] = set()
        for connection in self.connections:
            self._require_entity(connection.source)
            self._require_entity(connection.target)
            self._validate_connection(
                connection.source,
                connection.target,
                connection.max_flow_tph,
            )
            edge = (connection.source, connection.target)
            if edge in seen_connections:
                raise ValueError(
                    f"Duplicate connection: {connection.source}->{connection.target}"
                )
            seen_connections.add(edge)

    def _single_downstream_of_type(self, entity_id: str, entity_type: type) -> str | None:
        matches = self._downstream_of_type(entity_id, entity_type)
        return matches[0] if matches else None

    def _downstream_of_type(self, entity_id: str, entity_type: type) -> list[str]:
        return [
            downstream_id
            for downstream_id in self.downstream_of(entity_id)
            if isinstance(self.entities[downstream_id], entity_type)
        ]

    def _upstream_of_type(self, entity_id: str, entity_type: type) -> list[str]:
        return [
            upstream_id
            for upstream_id in self.upstream_of(entity_id)
            if isinstance(self.entities[upstream_id], entity_type)
        ]

    def _require_entity(self, entity_id: str) -> None:
        if entity_id not in self.entities:
            raise KeyError(f"Unknown entity: {entity_id}")

    def _validate_entity(self, entity: Entity) -> None:
        if not entity.entity_id:
            raise ValueError("Entity id must not be empty")
        if isinstance(entity, Emitter):
            _require_non_negative(entity.nominal_capture_tph, entity.entity_id, "capture rate")
            _require_non_negative(entity.buffer_capacity_t, entity.entity_id, "buffer capacity")
            _require_fraction(entity.min_utilization, entity.entity_id, "minimum utilization")
            _require_fraction(entity.default_utilization, entity.entity_id, "default utilization")
            _require_fraction(entity.availability, entity.entity_id, "availability")
            _require_non_negative(entity.loading_rate_tph, entity.entity_id, "loading rate")
        elif isinstance(entity, Vessel):
            _require_non_negative(entity.capacity_t, entity.entity_id, "cargo capacity")
            _require_non_negative(entity.loading_rate_tph, entity.entity_id, "loading rate")
            _require_non_negative(entity.unloading_rate_tph, entity.entity_id, "unloading rate")
        elif isinstance(entity, Terminal):
            _require_non_negative(entity.storage_capacity_t, entity.entity_id, "storage capacity")
            if entity.berth_count < 0:
                raise ValueError(f"Terminal {entity.entity_id} has a negative berth count")
        elif isinstance(entity, Pipeline):
            _require_non_negative(entity.max_flow_tph, entity.entity_id, "maximum flow")
        elif isinstance(entity, SubseaManifold):
            _require_non_negative(entity.max_flow_tph, entity.entity_id, "maximum flow")
        elif isinstance(entity, InjectionWell):
            _require_non_negative(entity.max_injection_tph, entity.entity_id, "maximum injection")
            _require_non_negative(
                entity.min_stable_injection_tph,
                entity.entity_id,
                "minimum stable injection",
            )
            if entity.min_stable_injection_tph > entity.max_injection_tph:
                raise ValueError(
                    f"Well {entity.entity_id} has a minimum rate above its maximum rate"
                )
        elif isinstance(entity, Reservoir):
            _require_non_negative(entity.storage_capacity_t, entity.entity_id, "storage capacity")
            if entity.max_pressure_bar < entity.initial_pressure_bar:
                raise ValueError(
                    f"Reservoir {entity.entity_id} maximum pressure is below initial pressure"
                )

    def _validate_connection(
        self,
        source_id: str,
        target_id: str,
        max_flow_tph: float | None,
    ) -> None:
        if source_id == target_id:
            raise ValueError(f"Self connection is not allowed: {source_id}")
        if max_flow_tph is not None:
            _require_non_negative(max_flow_tph, f"{source_id}->{target_id}", "flow limit")
        source = self.entities[source_id]
        target = self.entities[target_id]
        supported = (
            (Emitter, Vessel),
            (Vessel, Terminal),
            (Terminal, Pipeline),
            (Pipeline, SubseaManifold),
            (Pipeline, InjectionWell),
            (SubseaManifold, InjectionWell),
            (InjectionWell, Reservoir),
        )
        if not any(isinstance(source, src) and isinstance(target, dst) for src, dst in supported):
            raise ValueError(
                "Unsupported physical connection: "
                f"{type(source).__name__}({source_id}) -> "
                f"{type(target).__name__}({target_id})"
            )

    def _record_injection_rate_history(self, state: PhysicalState) -> None:
        interval_start_h = state.time_h - self.time_step_hours
        for well_id in self._entities_of_type(InjectionWell):
            rate_tph = state.last_injection_flow_tph.get(well_id, 0.0)
            history = state.injection_rate_history_tph.setdefault(well_id, [])
            if history and abs(history[-1][1] - rate_tph) <= 1e-12:
                continue
            if not history and abs(rate_tph) <= 1e-12:
                continue
            history.append((interval_start_h, rate_tph))

    def _record_pipeline_flow_history(self, state: PhysicalState) -> None:
        """Record throughput so annual pipeline capacities use a rolling window.

        记录管道输送量，使年度管道容量使用滚动时间窗口。
        """
        interval_start_h = state.time_h - self.time_step_hours
        window_start_h = interval_start_h - 365.25 * 24.0
        for pipeline_id in self._entities_of_type(Pipeline):
            history = state.pipeline_flow_history_t.setdefault(pipeline_id, [])
            history[:] = [
                entry for entry in history if entry[0] > window_start_h
            ]
            amount_t = (
                state.last_pipeline_flow_tph.get(pipeline_id, 0.0)
                * self.time_step_hours
            )
            history.append((interval_start_h, amount_t))


def _require_non_negative(value: float, entity_id: str, field_name: str) -> None:
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(
            f"{entity_id} has an invalid {field_name}: {value!r}; expected a finite non-negative value"
        )


def _require_fraction(value: float, entity_id: str, field_name: str) -> None:
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(
            f"{entity_id} has an invalid {field_name}: {value!r}; expected a value in [0, 1]"
        )
