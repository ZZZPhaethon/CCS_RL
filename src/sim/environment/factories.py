"""Factory helpers that wire the real public scenarios into :class:`CCSEnv`.

These build a ready-to-train RL environment on the calibrated Northern Lights
network (real emitters, the four 7,500 t Phase 1 ships, the single-berth Oygarden
terminal, the Aurora reservoir and - when available - the real hourly capture
profiles) instead of a toy network, so metrics are research-meaningful.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Literal

from ..economics import CostModel
from ..network_scenarios import (
    _load_fixed_scenario_data,
    _load_phase1_data,
    build_fixed_scenario_demo,
    build_northern_lights_phase1_demo,
)
from ..scenario_generation import ScenarioConfig, ScenarioGenerator
from ..simulator import SimulatorStepCounter
from .env import CCSEnv, CCSEnvConfig

Coordinate = tuple[float, float]
WeatherMode = Literal["window", "block", "leg_wave_climatology", "wave_height_netcdf", "lstm_forecast"]
DEFAULT_PHASE1_LEG_WAVE_CSV = Path("output/wave_height/phase1_leg_wave_2010_2014.csv")


def _scenario_locations(data: dict) -> dict[str, Coordinate]:
    return {
        location_id: (float(values[0]), float(values[1]))
        for location_id, values in data["locations"].items()
    }


def build_phase1_env(
    scenario_generator: ScenarioGenerator | None = None,
    cost_model: CostModel | None = None,
    config: CCSEnvConfig | None = None,
    *,
    scenario: str = "northern_lights_phase1",
    scenario_config: ScenarioConfig | None = None,
    weather_mode: WeatherMode = "window",
    leg_wave_csv: str | Path = DEFAULT_PHASE1_LEG_WAVE_CSV,
    wave_height_nc_paths: str | Path | list[str | Path] | None = None,
    wave_height_reader=None,
    lstm_prediction_csv: str | Path | None = None,
    simulator_step_counter: SimulatorStepCounter | None = None,
) -> CCSEnv:
    """A ``CCSEnv`` on a registered fixed scenario (default Northern Lights Phase 1).

    ``scenario`` selects any id in the fixed-scenario registry (e.g. the milk-run
    variants). ``weather_mode`` chooses one of the supported weather sources:
    probability windows, fixed-interval global blocks, leg-wave climatology,
    wave-height NetCDF, or LSTM forecast CSV.
    """
    if scenario == "northern_lights_phase1":
        network, _state = build_northern_lights_phase1_demo()
        locations = _scenario_locations(_load_phase1_data())
    else:
        network, _state = build_fixed_scenario_demo(scenario)
        locations = _scenario_locations(_load_fixed_scenario_data(scenario))
    weather_observation_layout = "global" if weather_mode in {"window", "block"} else "leg"
    env_config = replace(
        config or CCSEnvConfig(),
        weather_observation_layout=weather_observation_layout,
    )
    routes = None
    if scenario_generator is None and weather_mode in {"wave_height_netcdf", "lstm_forecast"}:
        routes = _phase1_routes(network, locations, env_config, scenario_config)
    if scenario_generator is None:
        scenario_generator = _default_phase1_scenario_generator(
            env_config,
            scenario_config=scenario_config,
            weather_mode=weather_mode,
            leg_wave_csv=leg_wave_csv,
            routes=routes,
            wave_height_nc_paths=wave_height_nc_paths,
            wave_height_reader=wave_height_reader,
            lstm_prediction_csv=lstm_prediction_csv,
        )
    return CCSEnv(
        network,
        locations,
        scenario_generator=scenario_generator,
        cost_model=cost_model,
        config=env_config,
        routes=routes,
        simulator_step_counter=simulator_step_counter,
    )


def _default_phase1_scenario_generator(
    env_config: CCSEnvConfig,
    *,
    scenario_config: ScenarioConfig | None,
    weather_mode: WeatherMode,
    leg_wave_csv: str | Path,
    routes: dict[str, dict] | None,
    wave_height_nc_paths: str | Path | list[str | Path] | None,
    wave_height_reader,
    lstm_prediction_csv: str | Path | None,
) -> ScenarioGenerator:
    path = Path(leg_wave_csv)
    effective_scenario_config = scenario_config or ScenarioConfig(
        episode_hours=env_config.episode_hours,
        time_step_hours=1.0,
    )
    if weather_mode == "window":
        return ScenarioGenerator(config=effective_scenario_config)
    if weather_mode == "block":
        return ScenarioGenerator(config=replace(effective_scenario_config, weather_process="block"))
    if weather_mode == "leg_wave_climatology":
        if not path.exists():
            raise FileNotFoundError(f"Leg-wave climatology CSV not found: {path}")
        from ..scenario_generation.wave_height.climatology_scenario import (
            LegWaveClimatologyScenarioGenerator,
        )

        return LegWaveClimatologyScenarioGenerator(
            path,
            config=effective_scenario_config,
        )
    if weather_mode == "wave_height_netcdf":
        if wave_height_reader is None and wave_height_nc_paths is None:
            raise ValueError("weather_mode='wave_height_netcdf' requires wave_height_nc_paths or wave_height_reader.")
        if routes is None:
            raise ValueError("weather_mode='wave_height_netcdf' requires routes.")
        from ..scenario_generation.wave_height import WaveHeightScenarioGenerator

        return WaveHeightScenarioGenerator(
            wave_height_nc_paths,
            routes=routes,
            config=effective_scenario_config,
            reader=wave_height_reader,
        )
    if weather_mode == "lstm_forecast":
        if lstm_prediction_csv is None:
            raise ValueError("weather_mode='lstm_forecast' requires lstm_prediction_csv.")
        if routes is None:
            raise ValueError("weather_mode='lstm_forecast' requires routes.")
        from ..scenario_generation.wave_height import LSTMWaveHeightScenarioGenerator

        return LSTMWaveHeightScenarioGenerator(
            lstm_prediction_csv,
            routes=routes,
            config=effective_scenario_config,
        )
    raise ValueError(f"Unknown weather_mode: {weather_mode!r}")


def _phase1_routes(
    network,
    locations: dict[str, Coordinate],
    env_config: CCSEnvConfig,
    scenario_config: ScenarioConfig | None,
) -> dict[str, dict]:
    route_env = CCSEnv(
        network,
        locations,
        scenario_generator=ScenarioGenerator(
            config=scenario_config or ScenarioConfig(episode_hours=env_config.episode_hours)
        ),
        config=env_config,
    )
    return route_env._routes
