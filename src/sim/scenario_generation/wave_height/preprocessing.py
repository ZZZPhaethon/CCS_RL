from __future__ import annotations

import csv
import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ...environment import build_phase1_env
from ...ship_speed import NORTHERN_LIGHTS_SHIP, ShipSpeedParameters, speed_factor
from .routes import RouteWaveConfig, WaveHeightReader, aggregate_wave_heights

ProgressCallback = Callable[[str], None]


@dataclass(frozen=True)
class RouteWaveDatasetConfig:
    """Settings for exporting route-level wave-height datasets."""

    wave_config: RouteWaveConfig = field(
        default_factory=lambda: RouteWaveConfig(sample_spacing_km=75.0, aggregation="p75")
    )
    aggregations: tuple[str, ...] = ("mean", "p75", "p90", "max")
    default_ship_parameters: ShipSpeedParameters = NORTHERN_LIGHTS_SHIP
    ship_parameters_by_vessel: Mapping[str, ShipSpeedParameters] = field(default_factory=dict)
    start_record: int = 0
    hours: int | None = None


def discover_wave_height_files(wave_dir: str | Path, pattern: str = "wam10ei_*.nc") -> list[Path]:
    """Return sorted wave-height NetCDF files from a directory."""
    files = sorted(Path(wave_dir).glob(pattern))
    if not files:
        raise FileNotFoundError(f"No NetCDF files matching {pattern!r} found in {wave_dir}.")
    return files


def write_phase1_route_wave_dataset(
    wave_dir_or_paths: str | Path | Iterable[str | Path],
    output_path: str | Path,
    *,
    config: RouteWaveDatasetConfig | None = None,
    progress: ProgressCallback | None = None,
) -> Path:
    """Export Phase 1 route-level wave-height and speed-factor rows to CSV."""
    env = build_phase1_env()
    nc_paths = (
        discover_wave_height_files(wave_dir_or_paths)
        if isinstance(wave_dir_or_paths, (str, Path)) and Path(wave_dir_or_paths).is_dir()
        else list(wave_dir_or_paths) if not isinstance(wave_dir_or_paths, (str, Path)) else [wave_dir_or_paths]
    )
    return write_route_wave_dataset(
        nc_paths,
        env._routes,
        output_path,
        config=config,
        progress=progress,
    )


def write_route_wave_dataset(
    nc_paths: Iterable[str | Path],
    routes: Mapping[str, Mapping[str, Any]],
    output_path: str | Path,
    *,
    config: RouteWaveDatasetConfig | None = None,
    progress: ProgressCallback | None = None,
) -> Path:
    """Write route-level hourly wave-height features for a set of vessel routes.

    Each row represents one vessel route at one hourly NetCDF record. Columns
    include wave-height aggregations and the corresponding STAwave-1 speed
    factors, ready for training a time-series model or driving scenarios.
    """
    config = config or RouteWaveDatasetConfig()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    reader = WaveHeightReader(nc_paths, config=config.wave_config)
    hours = reader.total_records - config.start_record if config.hours is None else config.hours
    if config.start_record < 0:
        raise ValueError("start_record must be non-negative")
    if hours < 0:
        raise ValueError("hours must be non-negative")
    if config.start_record + hours > reader.total_records:
        raise ValueError(
            f"Requested records {config.start_record}:{config.start_record + hours}, "
            f"but only {reader.total_records} records are available."
        )

    route_indices = {
        vessel_id: reader.route_grid_indices(route["coordinates"])
        for vessel_id, route in routes.items()
        if route.get("coordinates")
    }
    fieldnames = _fieldnames(config.aggregations)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for offset, global_record in enumerate(range(config.start_record, config.start_record + hours)):
            nc, local_record = reader._file_for_record(global_record)
            grid = nc.read_record_grid(config.wave_config.variable_name, local_record)
            fill_value = nc.fill_value(config.wave_config.variable_name)
            for vessel_id, indices in route_indices.items():
                route = routes[vessel_id]
                wave_values = _valid_values((grid[index] for index in indices), fill_value)
                row = _base_row(
                    global_record=global_record,
                    local_record=local_record,
                    source_file=nc.path,
                    vessel_id=vessel_id,
                    route=route,
                )
                _add_wave_and_speed_columns(
                    row,
                    wave_values,
                    aggregations=config.aggregations,
                    parameters=config.ship_parameters_by_vessel.get(
                        vessel_id,
                        config.default_ship_parameters,
                    ),
                    nominal_speed_knots=float(route.get("speed_knots") or config.default_ship_parameters.design_speed_knots),
                )
                writer.writerow(row)
            if progress is not None and (offset == 0 or (offset + 1) % 500 == 0 or offset + 1 == hours):
                progress(f"wrote {offset + 1}/{hours} hourly records to {output}")
    return output


def _fieldnames(aggregations: tuple[str, ...]) -> list[str]:
    fields = [
        "global_record",
        "source_file",
        "source_record",
        "vessel_id",
        "origin",
        "destination",
        "route_provider",
        "distance_km",
        "speed_knots",
    ]
    fields.extend(f"hs_{aggregation}_m" for aggregation in aggregations)
    for aggregation in aggregations:
        fields.append(f"speed_factor_{aggregation}")
    return fields


def _base_row(
    *,
    global_record: int,
    local_record: int,
    source_file: Path,
    vessel_id: str,
    route: Mapping[str, Any],
) -> dict[str, object]:
    return {
        "global_record": global_record,
        "source_file": source_file.name,
        "source_record": local_record,
        "vessel_id": vessel_id,
        "origin": route.get("origin", ""),
        "destination": route.get("destination", ""),
        "route_provider": route.get("provider", ""),
        "distance_km": route.get("distance_km", ""),
        "speed_knots": route.get("speed_knots", ""),
    }


def _add_wave_and_speed_columns(
    row: dict[str, object],
    wave_values: list[float],
    *,
    aggregations: tuple[str, ...],
    parameters: ShipSpeedParameters,
    nominal_speed_knots: float,
) -> None:
    for aggregation in aggregations:
        wave_height = aggregate_wave_heights(wave_values, aggregation)
        row[f"hs_{aggregation}_m"] = wave_height
        row[f"speed_factor_{aggregation}"] = speed_factor(
            wave_height,
            parameters,
            nominal_speed_knots=nominal_speed_knots,
        )


def _valid_values(values: Iterable[float], fill_value: float | None) -> list[float]:
    valid = [
        float(value)
        for value in values
        if math.isfinite(float(value)) and (fill_value is None or not math.isclose(float(value), fill_value))
    ]
    if not valid:
        raise ValueError("No valid wave-height values were found for a route.")
    return valid
