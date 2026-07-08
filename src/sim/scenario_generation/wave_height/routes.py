from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from ...routes import haversine_km, route_distance_km
from .netcdf import ClassicNetCDF

Coordinate = tuple[float, float]


@dataclass(frozen=True)
class RouteWaveConfig:
    variable_name: str = "significant_wave_height"
    latitude_name: str = "latitude"
    longitude_name: str = "longitude"
    sample_spacing_km: float = 25.0
    aggregation: str = "p75"
    valid_probe_record: int = 0


def densify_route(coordinates: Iterable[Coordinate], spacing_km: float = 25.0) -> list[Coordinate]:
    """Sample a route at roughly fixed distance intervals."""
    points = list(coordinates)
    if len(points) < 2:
        return points
    if spacing_km <= 0.0:
        raise ValueError("spacing_km must be positive")

    densified = [points[0]]
    for start, end in zip(points, points[1:]):
        segment_km = haversine_km(start, end)
        steps = max(1, int(math.ceil(segment_km / spacing_km)))
        for step in range(1, steps + 1):
            fraction = step / steps
            densified.append(
                (
                    start[0] + (end[0] - start[0]) * fraction,
                    start[1] + (end[1] - start[1]) * fraction,
                )
            )
    return densified


def aggregate_wave_heights(values: Iterable[float], method: str = "p75") -> float:
    """Aggregate point wave heights into one route-level representative value."""
    clean = [value for value in values if math.isfinite(value)]
    if not clean:
        raise ValueError("Cannot aggregate an empty wave-height series.")
    method = method.lower()
    if method == "mean":
        return sum(clean) / len(clean)
    if method == "max":
        return max(clean)
    if method.startswith("p"):
        percentile = float(method[1:])
        if not 0.0 <= percentile <= 100.0:
            raise ValueError(f"Unsupported percentile aggregation: {method!r}.")
        return _percentile(clean, percentile)
    raise ValueError(f"Unsupported wave-height aggregation method: {method!r}.")


def route_wave_height_series(
    nc_paths: str | Path | Iterable[str | Path],
    route_coordinates: Iterable[Coordinate],
    *,
    start_record: int = 0,
    hours: int | None = None,
    config: RouteWaveConfig | None = None,
) -> list[float]:
    """Read a route-level significant-wave-height trajectory from NetCDF files."""
    return WaveHeightReader(nc_paths, config=config).route_wave_height_series(
        route_coordinates,
        start_record=start_record,
        hours=hours,
    )


class WaveHeightReader:
    """Read route-level wave-height time series from one or more NetCDF files."""

    def __init__(
        self,
        nc_paths: str | Path | Iterable[str | Path],
        config: RouteWaveConfig | None = None,
    ) -> None:
        self.config = config or RouteWaveConfig()
        if isinstance(nc_paths, (str, Path)):
            paths = [Path(nc_paths)]
        else:
            paths = [Path(path) for path in nc_paths]
        if not paths:
            raise ValueError("At least one wave-height NetCDF path is required.")
        self.files = [ClassicNetCDF(path) for path in paths]
        self._offsets = _record_offsets(self.files)
        self.total_records = sum(nc.num_records for nc in self.files)
        self._latitudes: list[float] | None = None
        self._longitudes: list[float] | None = None
        self._valid_indices: list[int] | None = None
        self._route_index_cache: dict[tuple[Coordinate, ...], list[int]] = {}

    def route_wave_height_series(
        self,
        route_coordinates: Iterable[Coordinate],
        *,
        start_record: int = 0,
        hours: int | None = None,
    ) -> list[float]:
        if start_record < 0:
            raise ValueError("start_record must be non-negative")
        if hours is None:
            hours = self.total_records - start_record
        if hours < 0:
            raise ValueError("hours must be non-negative")
        if start_record + hours > self.total_records:
            raise ValueError(
                f"Requested records {start_record}:{start_record + hours}, "
                f"but only {self.total_records} records are available."
            )

        indices = self.route_grid_indices(route_coordinates)
        series: list[float] = []
        for global_record in range(start_record, start_record + hours):
            nc, local_record = self._file_for_record(global_record)
            values = nc.read_record_values(self.config.variable_name, local_record, indices)
            fill_value = nc.fill_value(self.config.variable_name)
            valid = [
                value
                for value in values
                if math.isfinite(value) and (fill_value is None or not math.isclose(value, fill_value))
            ]
            series.append(aggregate_wave_heights(valid, self.config.aggregation))
        return series

    def route_grid_indices(self, route_coordinates: Iterable[Coordinate]) -> list[int]:
        sampled = tuple(densify_route(route_coordinates, self.config.sample_spacing_km))
        if not sampled:
            raise ValueError("Route must contain at least one coordinate.")
        if sampled not in self._route_index_cache:
            self._route_index_cache[sampled] = [
                self._nearest_valid_grid_index(point)
                for point in sampled
            ]
        return self._route_index_cache[sampled]

    def _file_for_record(self, global_record: int) -> tuple[ClassicNetCDF, int]:
        if not 0 <= global_record < self.total_records:
            raise IndexError(f"Record index {global_record} outside 0..{self.total_records - 1}.")
        for offset, nc in zip(self._offsets, self.files):
            if global_record < offset + nc.num_records:
                return nc, global_record - offset
        raise IndexError(global_record)

    def _nearest_valid_grid_index(self, point: Coordinate) -> int:
        latitudes = self._latitudes_grid()
        longitudes = self._longitudes_grid()
        valid_indices = self._valid_grid_indices()
        target_lat, target_lon = point
        best_index = valid_indices[0]
        best_distance = float("inf")
        for index in valid_indices:
            lat = latitudes[index]
            lon = longitudes[index]
            lon_scale = math.cos(math.radians(target_lat))
            distance = (lat - target_lat) ** 2 + ((lon - target_lon) * lon_scale) ** 2
            if distance < best_distance:
                best_distance = distance
                best_index = index
        return best_index

    def _latitudes_grid(self) -> list[float]:
        if self._latitudes is None:
            self._latitudes = self.files[0].read_grid(self.config.latitude_name)
        return self._latitudes

    def _longitudes_grid(self) -> list[float]:
        if self._longitudes is None:
            self._longitudes = self.files[0].read_grid(self.config.longitude_name)
        return self._longitudes

    def _valid_grid_indices(self) -> list[int]:
        if self._valid_indices is None:
            nc = self.files[0]
            probe_record = min(self.config.valid_probe_record, nc.num_records - 1)
            grid = nc.read_record_grid(self.config.variable_name, probe_record)
            fill_value = nc.fill_value(self.config.variable_name)
            self._valid_indices = [
                index
                for index, value in enumerate(grid)
                if math.isfinite(value) and (fill_value is None or not math.isclose(value, fill_value))
            ]
            if not self._valid_indices:
                raise ValueError(f"No valid cells found for {self.config.variable_name!r}.")
        return self._valid_indices


def _record_offsets(files: list[ClassicNetCDF]) -> list[int]:
    offsets: list[int] = []
    total = 0
    for nc in files:
        offsets.append(total)
        total += nc.num_records
    return offsets


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (percentile / 100.0) * (len(ordered) - 1)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return ordered[lo]
    fraction = rank - lo
    return ordered[lo] + (ordered[hi] - ordered[lo]) * fraction
