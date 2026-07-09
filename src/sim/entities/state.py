from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PhysicalState:
    """Mutable simulation state indexed by entity id."""

    time_h: float = 0.0
    entity_inventory_t: dict[str, float] = field(default_factory=dict)
    last_capture_tph: dict[str, float] = field(default_factory=dict)
    last_vent_tph: dict[str, float] = field(default_factory=dict)
    cumulative_vent_t: dict[str, float] = field(default_factory=dict)
    last_pipeline_flow_tph: dict[str, float] = field(default_factory=dict)
    last_injection_flow_tph: dict[str, float] = field(default_factory=dict)
    injection_rate_history_tph: dict[str, list[tuple[float, float]]] = field(default_factory=dict)
    vessel_berths: dict[str, str] = field(default_factory=dict)

    # Time-varying disturbance overrides (the "ξ_t" channel).
    # Each maps entity_id -> a runtime override that takes precedence over the
    # entity's static (frozen) nominal parameter. Absence means "use nominal".
    # A ScenarioGenerator or RL/eval harness writes these per timestep so the
    # physics can react to weather, outages, maintenance and injectivity factors
    # without mutating the frozen entity definitions.
    emitter_availability: dict[str, float] = field(default_factory=dict)
    well_available: dict[str, bool] = field(default_factory=dict)
    injectivity_factor: dict[str, float] = field(default_factory=dict)
    vessel_speed_factor: dict[str, float] = field(default_factory=dict)
    leg_speed_factor: dict[str, float] = field(default_factory=dict)
    berth_count_override: dict[str, int] = field(default_factory=dict)

    def copy(self) -> "PhysicalState":
        return PhysicalState(
            time_h=self.time_h,
            entity_inventory_t=dict(self.entity_inventory_t),
            last_capture_tph=dict(self.last_capture_tph),
            last_vent_tph=dict(self.last_vent_tph),
            cumulative_vent_t=dict(self.cumulative_vent_t),
            last_pipeline_flow_tph=dict(self.last_pipeline_flow_tph),
            last_injection_flow_tph=dict(self.last_injection_flow_tph),
            injection_rate_history_tph={
                well_id: list(history)
                for well_id, history in self.injection_rate_history_tph.items()
            },
            vessel_berths=dict(self.vessel_berths),
            emitter_availability=dict(self.emitter_availability),
            well_available=dict(self.well_available),
            injectivity_factor=dict(self.injectivity_factor),
            vessel_speed_factor=dict(self.vessel_speed_factor),
            leg_speed_factor=dict(self.leg_speed_factor),
            berth_count_override=dict(self.berth_count_override),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "time_h": self.time_h,
            "entity_inventory_t": dict(self.entity_inventory_t),
            "last_capture_tph": dict(self.last_capture_tph),
            "last_vent_tph": dict(self.last_vent_tph),
            "cumulative_vent_t": dict(self.cumulative_vent_t),
            "last_pipeline_flow_tph": dict(self.last_pipeline_flow_tph),
            "last_injection_flow_tph": dict(self.last_injection_flow_tph),
            "injection_rate_history_tph": {
                well_id: list(history)
                for well_id, history in self.injection_rate_history_tph.items()
            },
            "vessel_berths": dict(self.vessel_berths),
            "emitter_availability": dict(self.emitter_availability),
            "well_available": dict(self.well_available),
            "injectivity_factor": dict(self.injectivity_factor),
            "vessel_speed_factor": dict(self.vessel_speed_factor),
            "leg_speed_factor": dict(self.leg_speed_factor),
            "berth_count_override": dict(self.berth_count_override),
        }


@dataclass(frozen=True)
class Violation:
    violation_type: str
    entity_id: str
    requested_t: float
    actual_t: float
    magnitude_t: float
    message: str

    def as_dict(self) -> dict[str, object]:
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
    state: PhysicalState
    flows_t: dict[tuple[str, str], float]
    violations: list[Violation]
    mass_balance_error_t: float

    def as_dict(self) -> dict[str, object]:
        return {
            "state": self.state.as_dict(),
            "flows_t": {f"{source}->{target}": amount for (source, target), amount in self.flows_t.items()},
            "violations": [violation.as_dict() for violation in self.violations],
            "mass_balance_error_t": self.mass_balance_error_t,
        }
