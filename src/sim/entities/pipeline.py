from __future__ import annotations

from dataclasses import dataclass, field

Coordinate = tuple[float, float]


@dataclass(frozen=True)
class Pipeline:
    entity_id: str
    max_flow_tph: float
    annual_capacity_tpy: float | None = None
    length_km: float | None = None
    route_color: str | None = None
    route_coordinates: list[Coordinate] = field(default_factory=list)
