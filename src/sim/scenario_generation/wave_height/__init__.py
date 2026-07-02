"""Wave-height driven weather scenarios.

This package turns historical or forecast significant-wave-height fields into
the existing ``Scenario.vessel_speed_factor`` disturbance interface.
"""

from .netcdf import ClassicNetCDF, NetCDFVariable
from .preprocessing import (
    RouteWaveDatasetConfig,
    discover_wave_height_files,
    write_phase1_route_wave_dataset,
    write_route_wave_dataset,
)
from .routes import (
    RouteWaveConfig,
    aggregate_wave_heights,
    densify_route,
    route_wave_height_series,
)
from .scenario import WaveHeightScenarioGenerator
from .visualization import plot_phase1_wave_height_snapshot, plot_wave_height_snapshot

__all__ = [
    "ClassicNetCDF",
    "NetCDFVariable",
    "RouteWaveDatasetConfig",
    "RouteWaveConfig",
    "WaveHeightScenarioGenerator",
    "aggregate_wave_heights",
    "densify_route",
    "discover_wave_height_files",
    "plot_phase1_wave_height_snapshot",
    "plot_wave_height_snapshot",
    "route_wave_height_series",
    "write_phase1_route_wave_dataset",
    "write_route_wave_dataset",
]
