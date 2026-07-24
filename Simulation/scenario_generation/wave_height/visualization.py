"""Plot wave-height fields with CCS routes and locations for data validation.

绘制带有 CCS 航线和地点的波高场，以验证数据质量。
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Iterable, Mapping

from .netcdf import ClassicNetCDF

Coordinate = tuple[float, float]


def plot_wave_height_snapshot(
    nc_path: str | Path,
    *,
    record_index: int = 0,
    output_path: str | Path = "output/wave_height/wave_height_snapshot.png",
    routes: Mapping[str, Mapping[str, object]] | None = None,
    locations: Mapping[str, Coordinate] | None = None,
    variable_name: str = "significant_wave_height",
    latitude_name: str = "latitude",
    longitude_name: str = "longitude",
    stride: int = 1,
    title: str | None = None,
    zoom_to_routes: bool = True,
    padding_degrees: float = 1.5,
    figsize: tuple[float, float] = (12.0, 9.0),
    dpi: int = 260,
    label_locations: bool = True,
    location_label_ids: Iterable[str] | None = None,
) -> Path:
    """Plot one NetCDF wave-height record and optionally overlay vessel routes.

    The function reads only one hourly record, plus the latitude/longitude grids.
    ``stride`` down-samples the grid for plotting so a 400x248 field remains fast
    and visually clear.
    """
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / "output" / "matplotlib-cache"))
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise ImportError("plot_wave_height_snapshot requires matplotlib.") from exc

    if stride <= 0:
        raise ValueError("stride must be positive")

    nc = ClassicNetCDF(nc_path)
    latitudes = nc.read_grid(latitude_name)
    longitudes = nc.read_grid(longitude_name)
    wave_heights = nc.read_record_grid(variable_name, record_index)
    fill_value = nc.fill_value(variable_name)

    shape = _variable_shape(nc, variable_name, skip_time=True)
    lon_grid = np.asarray(longitudes, dtype=float).reshape(shape)[::stride, ::stride]
    lat_grid = np.asarray(latitudes, dtype=float).reshape(shape)[::stride, ::stride]
    wave_grid = np.asarray(wave_heights, dtype=float).reshape(shape)[::stride, ::stride]
    if fill_value is not None:
        wave_grid = np.where(np.isclose(wave_grid, fill_value), np.nan, wave_grid)

    if not np.isfinite(wave_grid).any():
        raise ValueError(f"No valid {variable_name!r} values found for record {record_index}.")

    fig, ax = plt.subplots(figsize=figsize)
    mesh = ax.pcolormesh(lon_grid, lat_grid, wave_grid, cmap="viridis", shading="auto", rasterized=True)
    colorbar = fig.colorbar(mesh, ax=ax, pad=0.02)
    colorbar.set_label("Significant wave height (m)")

    if routes:
        _plot_routes(ax, routes)
    if locations:
        _plot_locations(ax, locations, label_locations=label_locations, label_ids=location_label_ids)
    if zoom_to_routes:
        _zoom_to_overlays(ax, routes=routes, locations=locations, padding_degrees=padding_degrees)

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title or f"{variable_name}, record {record_index}")
    ax.grid(True, alpha=0.2)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi)
    plt.close(fig)
    return output


def plot_phase1_wave_height_snapshot(
    nc_path: str | Path,
    *,
    record_index: int = 0,
    output_path: str | Path = "output/wave_height/wave_height_phase1_snapshot.png",
    stride: int = 1,
) -> Path:
    """Plot one wave-height snapshot with the Phase 1 vessel routes overlaid."""
    from ...environment import build_phase1_env

    env = build_phase1_env(weather_mode="window")
    return plot_wave_height_snapshot(
        nc_path,
        record_index=record_index,
        output_path=output_path,
        routes=env._routes,
        locations=env.locations,
        stride=stride,
        title=f"Phase 1 routes over significant wave height, record {record_index}",
        location_label_ids={"brevik", "celsio", "yara_sluiskil", "oygarden_terminal"},
    )


def _plot_routes(ax, routes: Mapping[str, Mapping[str, object]]) -> None:
    for vessel_id, route in routes.items():
        coordinates = route.get("coordinates")
        if not coordinates:
            continue
        lats, lons = _split_coordinates(coordinates)
        ax.plot(lons, lats, linewidth=1.6, alpha=0.9, label=str(vessel_id))
    if routes:
        ax.legend(loc="upper right", fontsize=7, frameon=True)


def _plot_locations(
    ax,
    locations: Mapping[str, Coordinate],
    *,
    label_locations: bool = True,
    label_ids: Iterable[str] | None = None,
) -> None:
    label_set = set(label_ids) if label_ids is not None else None
    for location_id, coordinate in locations.items():
        lat, lon = coordinate
        ax.scatter([lon], [lat], marker="x", c="black", s=28, linewidths=1.2)
        if label_locations and (label_set is None or location_id in label_set):
            ax.text(
                lon,
                lat,
                f" {location_id}",
                fontsize=8,
                color="black",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.65, "pad": 1.0},
            )


def _split_coordinates(coordinates: Iterable[Coordinate]) -> tuple[list[float], list[float]]:
    lats: list[float] = []
    lons: list[float] = []
    for lat, lon in coordinates:
        lats.append(float(lat))
        lons.append(float(lon))
    return lats, lons


def _variable_shape(nc: ClassicNetCDF, variable_name: str, *, skip_time: bool = False) -> tuple[int, ...]:
    variable = nc.variable(variable_name)
    dimensions = variable.dimensions[1:] if skip_time and variable.is_record_variable else variable.dimensions
    return tuple(nc.dimensions[dimension] for dimension in dimensions)


def _zoom_to_overlays(
    ax,
    *,
    routes: Mapping[str, Mapping[str, object]] | None,
    locations: Mapping[str, Coordinate] | None,
    padding_degrees: float,
) -> None:
    coordinates: list[Coordinate] = []
    if routes:
        for route in routes.values():
            route_coordinates = route.get("coordinates")
            if route_coordinates:
                coordinates.extend(route_coordinates)
    if locations:
        coordinates.extend(locations.values())
    if not coordinates:
        return
    lats, lons = _split_coordinates(coordinates)
    ax.set_xlim(min(lons) - padding_degrees, max(lons) + padding_degrees)
    ax.set_ylim(min(lats) - padding_degrees, max(lats) + padding_degrees)
