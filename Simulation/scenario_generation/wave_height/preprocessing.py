"""Prepare route- and leg-level wave-height datasets for CCS scenarios.

为 CCS 场景准备航线级和航段级波高数据集。
"""

from __future__ import annotations

import csv
import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ...routes import route_distance_km, sea_route
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
    from ...environment import build_phase1_env

    env = build_phase1_env(weather_mode="window")
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


def build_candidate_leg_routes(
    env,
    *,
    include_emitter_to_emitter: bool = True,
    include_terminal_to_emitter: bool = True,
    default_speed_knots: float | None = None,
) -> dict[str, dict[str, object]]:
    """Build maritime routes for all controllable sailing legs in an environment.

    Route-level weather is tied to the fixed vessel home route. Leg-level weather
    instead keys weather by ``origin->destination``, which matches the rolling
    MILP action arcs when vessels can visit multiple emitters before sailing to
    the terminal.
    """
    speed_knots = (
        float(default_speed_knots)
        if default_speed_knots is not None
        else _default_env_route_speed_knots(env)
    )
    legs: dict[str, dict[str, object]] = {}
    for origin_id in env.emitter_ids:
        for terminal_id in env.terminal_ids:
            _add_leg_route(legs, env.locations, origin_id, terminal_id, speed_knots=speed_knots)
    if include_terminal_to_emitter:
        for terminal_id in env.terminal_ids:
            for emitter_id in env.emitter_ids:
                _add_leg_route(legs, env.locations, terminal_id, emitter_id, speed_knots=speed_knots)
    if include_emitter_to_emitter:
        for origin_id in env.emitter_ids:
            for destination_id in env.emitter_ids:
                if origin_id != destination_id:
                    _add_leg_route(legs, env.locations, origin_id, destination_id, speed_knots=speed_knots)
    return legs


def write_phase1_leg_wave_dataset(
    wave_dir_or_paths: str | Path | Iterable[str | Path],
    output_path: str | Path,
    *,
    config: RouteWaveDatasetConfig | None = None,
    progress: ProgressCallback | None = None,
    include_emitter_to_emitter: bool = True,
    include_terminal_to_emitter: bool = True,
) -> Path:
    """Export Phase 1 leg-level wave-height rows for all controllable legs."""
    from ...environment import build_phase1_env

    env = build_phase1_env(weather_mode="window")
    nc_paths = (
        discover_wave_height_files(wave_dir_or_paths)
        if isinstance(wave_dir_or_paths, (str, Path)) and Path(wave_dir_or_paths).is_dir()
        else list(wave_dir_or_paths) if not isinstance(wave_dir_or_paths, (str, Path)) else [wave_dir_or_paths]
    )
    legs = build_candidate_leg_routes(
        env,
        include_emitter_to_emitter=include_emitter_to_emitter,
        include_terminal_to_emitter=include_terminal_to_emitter,
    )
    return write_leg_wave_dataset(
        nc_paths,
        legs,
        output_path,
        config=config,
        progress=progress,
    )


def write_leg_wave_dataset(
    nc_paths: Iterable[str | Path],
    legs: Mapping[str, Mapping[str, Any]],
    output_path: str | Path,
    *,
    config: RouteWaveDatasetConfig | None = None,
    progress: ProgressCallback | None = None,
) -> Path:
    """Write hourly wave-height features keyed by sailing leg instead of vessel."""
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

    leg_indices = {
        leg_id: reader.route_grid_indices(leg["coordinates"])
        for leg_id, leg in legs.items()
        if leg.get("coordinates")
    }
    last_valid_wave_values: dict[str, list[float]] = {}
    fieldnames = _leg_fieldnames(config.aggregations)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for offset, global_record in enumerate(range(config.start_record, config.start_record + hours)):
            nc, local_record = reader._file_for_record(global_record)
            grid = nc.read_record_grid(config.wave_config.variable_name, local_record)
            fill_value = nc.fill_value(config.wave_config.variable_name)
            for leg_id, indices in leg_indices.items():
                leg = legs[leg_id]
                wave_values = _valid_values_or_none((grid[index] for index in indices), fill_value)
                if wave_values is None:
                    wave_values = last_valid_wave_values.get(leg_id, [0.0])
                else:
                    last_valid_wave_values[leg_id] = wave_values
                row = _base_leg_row(
                    global_record=global_record,
                    local_record=local_record,
                    source_file=nc.path,
                    leg_id=leg_id,
                    leg=leg,
                )
                _add_wave_and_speed_columns(
                    row,
                    wave_values,
                    aggregations=config.aggregations,
                    parameters=config.default_ship_parameters,
                    nominal_speed_knots=float(leg.get("speed_knots") or config.default_ship_parameters.design_speed_knots),
                )
                writer.writerow(row)
            if progress is not None and (offset == 0 or (offset + 1) % 500 == 0 or offset + 1 == hours):
                progress(f"wrote {offset + 1}/{hours} hourly leg records to {output}")
    return output


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


def _leg_fieldnames(aggregations: tuple[str, ...]) -> list[str]:
    fields = [
        "global_record",
        "source_file",
        "source_record",
        "leg_id",
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


def _base_leg_row(
    *,
    global_record: int,
    local_record: int,
    source_file: Path,
    leg_id: str,
    leg: Mapping[str, Any],
) -> dict[str, object]:
    return {
        "global_record": global_record,
        "source_file": source_file.name,
        "source_record": local_record,
        "leg_id": leg_id,
        "origin": leg.get("origin", ""),
        "destination": leg.get("destination", ""),
        "route_provider": leg.get("provider", ""),
        "distance_km": leg.get("distance_km", ""),
        "speed_knots": leg.get("speed_knots", ""),
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
    valid = _valid_values_or_none(values, fill_value)
    if valid is None:
        raise ValueError("No valid wave-height values were found for a route.")
    return valid


def _valid_values_or_none(values: Iterable[float], fill_value: float | None) -> list[float] | None:
    valid = [
        float(value)
        for value in values
        if math.isfinite(float(value)) and (fill_value is None or not math.isclose(float(value), fill_value))
    ]
    return valid or None


def _default_env_route_speed_knots(env) -> float:
    speeds = [
        float(route["speed_knots"])
        for route in env._routes.values()
        if route.get("speed_knots")
    ]
    if speeds:
        return speeds[0]
    return NORTHERN_LIGHTS_SHIP.design_speed_knots


def _add_leg_route(
    legs: dict[str, dict[str, object]],
    locations: Mapping[str, tuple[float, float]],
    origin_id: str,
    destination_id: str,
    *,
    speed_knots: float,
) -> None:
    leg_id = f"{origin_id}->{destination_id}"
    if leg_id in legs:
        return
    origin = locations[origin_id]
    destination = locations[destination_id]
    try:
        maritime_route = sea_route(origin, destination)
        coordinates = _connect_route_to_endpoints(
            maritime_route.coordinates,
            origin,
            destination,
        )
        provider = maritime_route.provider
    except Exception:
        coordinates = [origin, destination]
        provider = "direct"
    legs[leg_id] = {
        "id": leg_id,
        "origin": origin_id,
        "destination": destination_id,
        "provider": provider,
        "distance_km": route_distance_km(coordinates),
        "speed_knots": speed_knots,
        "coordinates": coordinates,
    }


def _connect_route_to_endpoints(
    coordinates: Iterable[tuple[float, float]],
    origin: tuple[float, float],
    destination: tuple[float, float],
) -> list[tuple[float, float]]:
    connected = list(coordinates)
    if not connected:
        return [origin, destination]
    if connected[0] != origin:
        connected.insert(0, origin)
    if connected[-1] != destination:
        connected.append(destination)
    return connected
