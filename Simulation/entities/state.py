"""Define mutable simulation state, constraint violations, and step results.

定义可变仿真状态、约束违规记录和时间步计算结果。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PhysicalState:
    """Store mutable physical state for a simulation time step.

    存储仿真时间步中的可变物理状态。

    Attributes:
        time_h: Current simulation time in hours. / 当前仿真时间，单位为小时。
        entity_inventory_t: Inventory by entity ID, in tonnes.
            / 按实体 ID 记录的库存量，单位为吨。
        last_capture_tph: Most recent capture rate by emitter,
            in tonnes per hour.
            / 按排放源记录的最新捕集速率，单位为吨每小时。
        last_vent_tph: Most recent venting rate by entity, in tonnes per hour.
            / 按实体记录的最新放空速率，单位为吨每小时。
        cumulative_vent_t: Cumulative vented quantity by entity, in tonnes.
            / 按实体记录的累计放空量，单位为吨。
        last_pipeline_flow_tph: Most recent pipeline flow by pipeline ID.
            / 按管道 ID 记录的最新管道流量，单位为吨每小时。
        pipeline_flow_history_t: Per-interval pipeline throughput history.
            / 按时间段记录的管道输送量历史，单位为吨。
        last_injection_flow_tph: Most recent injection flow by well ID.
            / 按注入井 ID 记录的最新注入流量，单位为吨每小时。
        injection_rate_history_tph: Timestamped injection-rate history
            by well ID.
            / 按注入井 ID 记录的带时间戳注入速率历史。
        vessel_berths: Berth assigned to each vessel, indexed by vessel ID.
            / 按船舶 ID 记录的泊位分配情况。
        terminal_unload_queues: Vessel unloading queues by terminal ID.
            / 按终端 ID 记录的船舶卸载队列。
        emitter_availability: Runtime availability overrides for emitters.
            / 排放源的运行时可用率覆盖值。
        well_available: Runtime availability overrides for injection wells.
            / 注入井的运行时可用性覆盖值。
        injectivity_factor: Runtime injectivity multipliers for wells.
            / 注入井的运行时注入能力系数。
        vessel_speed_factor: Runtime speed multipliers for vessels.
            / 船舶的运行时航速系数。
        leg_speed_factor: Runtime speed multipliers for transport legs.
            / 运输航段的运行时航速系数。
        berth_count_override: Runtime berth-count overrides by terminal.
            / 按终端记录的运行时泊位数量覆盖值。
    """

    time_h: float = 0.0
    entity_inventory_t: dict[str, float] = field(default_factory=dict)
    last_capture_tph: dict[str, float] = field(default_factory=dict)
    last_vent_tph: dict[str, float] = field(default_factory=dict)
    cumulative_vent_t: dict[str, float] = field(default_factory=dict)
    last_pipeline_flow_tph: dict[str, float] = field(default_factory=dict)
    pipeline_flow_history_t: dict[str, list[tuple[float, float]]] = field(
        default_factory=dict
    )
    last_injection_flow_tph: dict[str, float] = field(default_factory=dict)
    injection_rate_history_tph: dict[str, list[tuple[float, float]]] = field(
        default_factory=dict
    )
    vessel_berths: dict[str, str] = field(default_factory=dict)
    terminal_unload_queues: dict[str, list[str]] = field(default_factory=dict)

    # Store time-varying disturbance overrides (the "ξ_t" channel).
    # 存储时变扰动覆盖值（“ξ_t”通道）。
    # Each mapping assigns an entity ID to a runtime value that overrides its
    # nominal immutable parameter.
    # An absent entry means to use its nominal value.
    # 每个映射将实体 ID 对应到运行时覆盖值；缺少条目时使用不可变的名义参数。
    # Scenario generators and RL/evaluation harnesses update these mappings at
    # each time step to model weather, outages, maintenance, and injectivity.
    # 场景生成器和强化学习/评估程序会在每个时间步更新这些映射，以模拟天气、
    # 停机、维护和注入能力变化。
    emitter_availability: dict[str, float] = field(default_factory=dict)
    well_available: dict[str, bool] = field(default_factory=dict)
    injectivity_factor: dict[str, float] = field(default_factory=dict)
    vessel_speed_factor: dict[str, float] = field(default_factory=dict)
    leg_speed_factor: dict[str, float] = field(default_factory=dict)
    berth_count_override: dict[str, int] = field(default_factory=dict)

    def copy(self) -> "PhysicalState":
        """Return an independent copy of this state and all mutable values.

        返回该状态及所有可变值的独立副本。
        """
        return PhysicalState(
            time_h=self.time_h,
            entity_inventory_t=dict(self.entity_inventory_t),
            last_capture_tph=dict(self.last_capture_tph),
            last_vent_tph=dict(self.last_vent_tph),
            cumulative_vent_t=dict(self.cumulative_vent_t),
            last_pipeline_flow_tph=dict(self.last_pipeline_flow_tph),
            pipeline_flow_history_t={
                pipeline_id: list(history)
                for pipeline_id, history in self.pipeline_flow_history_t.items()
            },
            last_injection_flow_tph=dict(self.last_injection_flow_tph),
            injection_rate_history_tph={
                well_id: list(history)
                for well_id, history in self.injection_rate_history_tph.items()
            },
            vessel_berths=dict(self.vessel_berths),
            terminal_unload_queues={
                terminal_id: list(queue)
                for terminal_id, queue in self.terminal_unload_queues.items()
            },
            emitter_availability=dict(self.emitter_availability),
            well_available=dict(self.well_available),
            injectivity_factor=dict(self.injectivity_factor),
            vessel_speed_factor=dict(self.vessel_speed_factor),
            leg_speed_factor=dict(self.leg_speed_factor),
            berth_count_override=dict(self.berth_count_override),
        )

    def as_dict(self) -> dict[str, object]:
        """Serialise the state into dictionaries and lists without aliasing.

        将状态序列化为不共享可变引用的字典和列表。
        """
        return {
            "time_h": self.time_h,
            "entity_inventory_t": dict(self.entity_inventory_t),
            "last_capture_tph": dict(self.last_capture_tph),
            "last_vent_tph": dict(self.last_vent_tph),
            "cumulative_vent_t": dict(self.cumulative_vent_t),
            "last_pipeline_flow_tph": dict(self.last_pipeline_flow_tph),
            "pipeline_flow_history_t": {
                pipeline_id: list(history)
                for pipeline_id, history in self.pipeline_flow_history_t.items()
            },
            "last_injection_flow_tph": dict(self.last_injection_flow_tph),
            "injection_rate_history_tph": {
                well_id: list(history)
                for well_id, history in self.injection_rate_history_tph.items()
            },
            "vessel_berths": dict(self.vessel_berths),
            "terminal_unload_queues": {
                terminal_id: list(queue)
                for terminal_id, queue in self.terminal_unload_queues.items()
            },
            "emitter_availability": dict(self.emitter_availability),
            "well_available": dict(self.well_available),
            "injectivity_factor": dict(self.injectivity_factor),
            "vessel_speed_factor": dict(self.vessel_speed_factor),
            "leg_speed_factor": dict(self.leg_speed_factor),
            "berth_count_override": dict(self.berth_count_override),
        }


@dataclass(frozen=True)
class Violation:
    """Describe a constraint violation detected during a simulation step.

    描述在一个仿真时间步内检测到的约束违规。

    Attributes:
        violation_type: Category of violated constraint. / 违规约束的类别。
        entity_id: Identifier of the affected entity. / 受影响实体的标识符。
        requested_t: Requested quantity in tonnes. / 请求量，单位为吨。
        actual_t: Quantity actually achieved in tonnes. / 实际完成量，单位为吨。
        magnitude_t: Size of the violation in tonnes. / 违规幅度，单位为吨。
        message: Human-readable explanation. / 供人阅读的违规说明。
    """

    violation_type: str
    entity_id: str
    requested_t: float
    actual_t: float
    magnitude_t: float
    message: str

    def as_dict(self) -> dict[str, object]:
        """Serialise this violation into a dictionary.

        将该违规记录序列化为字典。
        """
        return {
            "violation_type": self.violation_type,
            "entity_id": self.entity_id,
            "requested_t": self.requested_t,
            "actual_t": self.actual_t,
            "magnitude_t": self.magnitude_t,
            "message": self.message,
        }


@dataclass
class StepResult:
    """Collect the outputs produced when advancing one simulation step.

    汇集推进一个仿真时间步后产生的输出。

    Attributes:
        state: Physical state after the step. / 时间步结束后的物理状态。
        flows_t: Transported quantities keyed by source and target IDs.
            / 以源和目标 ID 为键的输送量，单位为吨。
        violations: Constraint violations detected during the step.
            / 在该时间步检测到的约束违规记录。
        mass_balance_error_t: Residual mass-balance error in tonnes.
            / 质量平衡残差，单位为吨。
    """

    state: PhysicalState
    flows_t: dict[tuple[str, str], float]
    violations: list[Violation]
    mass_balance_error_t: float

    def as_dict(self) -> dict[str, object]:
        """Serialise the step result into a dictionary suitable for output.

        将时间步结果序列化为适合输出的字典。
        """
        return {
            "state": self.state.as_dict(),
            "flows_t": {
                f"{source}->{target}": amount
                for (source, target), amount in self.flows_t.items()
            },
            "violations": [
                violation.as_dict() for violation in self.violations
            ],
            "mass_balance_error_t": self.mass_balance_error_t,
        }
