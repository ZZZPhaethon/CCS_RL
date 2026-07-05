"""Factory helpers that wire the real public scenarios into :class:`CCSEnv`.

These build a ready-to-train RL environment on the calibrated Northern Lights
network (real emitters, the four 7,500 t Phase 1 ships, the single-berth Oygarden
terminal, the Aurora reservoir and - when available - the real hourly capture
profiles) instead of a toy network, so metrics are research-meaningful.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from ..economics import CostModel
from ..network_scenarios import (
    _load_phase1_data,
    build_northern_lights_phase1_demo,
)
from ..scenario_generation import ScenarioConfig, ScenarioGenerator
from .env import CCSEnv, CCSEnvConfig

Coordinate = tuple[float, float]
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
    scenario_config: ScenarioConfig | None = None,
    use_leg_wave_weather: bool = True,
    leg_wave_csv: str | Path = DEFAULT_PHASE1_LEG_WAVE_CSV,
) -> CCSEnv:
    """A ``CCSEnv`` on the real Northern Lights Phase 1 network."""
    network, _state = build_northern_lights_phase1_demo()
    locations = _scenario_locations(_load_phase1_data())
    env_config = config or CCSEnvConfig()
    if scenario_generator is None:
        scenario_generator = _default_phase1_scenario_generator(
            env_config,
            scenario_config=scenario_config,
            use_leg_wave_weather=use_leg_wave_weather,
            leg_wave_csv=leg_wave_csv,
        )
    return CCSEnv(
        network,
        locations,
        scenario_generator=scenario_generator,
        cost_model=cost_model,
        config=env_config,
    )


def _default_phase1_scenario_generator(
    env_config: CCSEnvConfig,
    *,
    scenario_config: ScenarioConfig | None,
    use_leg_wave_weather: bool,
    leg_wave_csv: str | Path,
) -> ScenarioGenerator:
    path = Path(leg_wave_csv)
    effective_scenario_config = replace(
        scenario_config,
        enable_weather=False,
    ) if scenario_config is not None else ScenarioConfig(
        episode_hours=env_config.episode_hours,
        time_step_hours=1.0,
        enable_weather=False,
    )
    if use_leg_wave_weather and path.exists():
        from ..scenario_generation.wave_height.climatology_scenario import (
            LegWaveClimatologyScenarioGenerator,
        )

        return LegWaveClimatologyScenarioGenerator(
            path,
            config=effective_scenario_config,
        )
    return ScenarioGenerator(config=effective_scenario_config)
