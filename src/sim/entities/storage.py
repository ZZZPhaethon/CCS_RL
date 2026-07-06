from __future__ import annotations

from dataclasses import dataclass, field

from ..line_source import LineSourceParameters


@dataclass(frozen=True)
class InjectionWell:
    entity_id: str
    max_injection_tph: float
    min_stable_injection_tph: float = 0.0
    injectivity_index_tph_per_bar: float | None = None
    pressure_margin_bar: float | None = None
    available: bool = True


@dataclass(frozen=True)
class Reservoir:
    entity_id: str
    storage_capacity_t: float
    initial_pressure_bar: float
    pressure_at_capacity_bar: float
    max_pressure_bar: float
    depth_m: float | None = None
    reservoir_pressure_model: str | None = None
    seawater_depth_m: float | None = None
    well_fracture_gradient_psi_per_ft: float | None = None
    well_fracture_gradient_reference_depth_m: float | None = None
    well_fracture_pressure_bar: float | None = None
    well_bottomhole_pressure_safety_factor: float | None = None
    well_bottomhole_pressure_limit_bar: float | None = None
    line_source_parameters: LineSourceParameters | None = None
    line_source_observation_radii_m: tuple[float, ...] = ()
    line_source_well_distances_m: dict[str, dict[str, float]] = field(default_factory=dict)
    line_source_parameter_status: dict[str, str] = field(default_factory=dict)

    def pressure_bar(self, stored_t: float) -> float:
        fill_fraction = max(0.0, min(1.0, stored_t / self.storage_capacity_t))
        return self.initial_pressure_bar + fill_fraction * (
            self.pressure_at_capacity_bar - self.initial_pressure_bar
        )

    def pressure_margin_bar(self, stored_t: float) -> float:
        return self.max_pressure_bar - self.pressure_bar(stored_t)

    def pressure_limited_capacity_t(self) -> float:
        pressure_span = self.pressure_at_capacity_bar - self.initial_pressure_bar
        if pressure_span <= 0:
            return self.storage_capacity_t
        pressure_fraction = (self.max_pressure_bar - self.initial_pressure_bar) / pressure_span
        return self.storage_capacity_t * max(0.0, min(1.0, pressure_fraction))
