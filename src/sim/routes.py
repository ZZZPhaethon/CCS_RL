from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

Coordinate = tuple[float, float]  # (lat, lon)


@dataclass(frozen=True)
class SeaRoute:
    coordinates: list[Coordinate]
    distance_km: float
    provider: str


def haversine_km(a: Coordinate, b: Coordinate) -> float:
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * 6371.0088 * math.asin(math.sqrt(h))


def route_distance_km(coordinates: Iterable[Coordinate]) -> float:
    points = list(coordinates)
    return sum(haversine_km(a, b) for a, b in zip(points, points[1:]))


def sea_route(origin: Coordinate, destination: Coordinate) -> SeaRoute:
    """Return a maritime route from the ``searoute`` package."""

    package_route = _route_with_searoute(origin, destination)
    if package_route is not None and len(package_route) > 2:
        return SeaRoute(package_route, route_distance_km(package_route), "searoute")

    raise RuntimeError("Maritime route generation requires searoute to return a multi-point route.")


def _route_with_searoute(origin: Coordinate, destination: Coordinate) -> list[Coordinate] | None:
    try:
        import searoute as searoute_package  # type: ignore
    except Exception:
        return None

    origin_lonlat = [origin[1], origin[0]]
    destination_lonlat = [destination[1], destination[0]]
    route = searoute_package.searoute(origin_lonlat, destination_lonlat)
    coordinates = route.get("geometry", {}).get("coordinates", [])
    if not coordinates:
        return None
    return [(lat, lon) for lon, lat in coordinates]
