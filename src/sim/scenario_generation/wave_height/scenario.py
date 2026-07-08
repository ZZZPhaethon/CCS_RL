from __future__ import annotations

import random
from collections.abc import Mapping
from pathlib import Path
from typing import Protocol

from ...entities.vessel import Vessel
from ...ship_speed import NORTHERN_LIGHTS_SHIP, ShipSpeedParameters, speed_factor_series
from ..generator import Scenario, ScenarioConfig, ScenarioGenerator
from .routes import RouteWaveConfig, WaveHeightReader


class RouteWaveReader(Protocol):
    total_records: int

    def route_wave_height_series(
        self,
        route_coordinates,
        *,
        start_record: int = 0,
        hours: int | None = None,
    ) -> list[float]:
        ...


class WaveHeightScenarioGenerator(ScenarioGenerator):
    """Scenario generator whose vessel slowdowns come from wave-height fields.

    It preserves the base ``ScenarioGenerator`` behavior for capture, well
    maintenance, injectivity, and initial inventory. Only
    ``Scenario.vessel_speed_factor`` is replaced with STAwave-1 factors derived
    from route-level significant wave height.
    """

    def __init__(
        self,
        nc_paths: str | Path | list[str | Path] | None = None,
        *,
        routes: Mapping[str, Mapping[str, object]],
        ship_parameters_by_vessel: Mapping[str, ShipSpeedParameters] | None = None,
        default_ship_parameters: ShipSpeedParameters = NORTHERN_LIGHTS_SHIP,
        wave_config: RouteWaveConfig | None = None,
        config: ScenarioConfig | None = None,
        seed: int | None = None,
        reader: RouteWaveReader | None = None,
    ) -> None:
        super().__init__(config=config, seed=seed)
        if reader is None and nc_paths is None:
            raise ValueError("Either nc_paths or reader must be provided.")
        self.routes = routes
        self.ship_parameters_by_vessel = dict(ship_parameters_by_vessel or {})
        self.default_ship_parameters = default_ship_parameters
        self.reader: RouteWaveReader = reader or WaveHeightReader(nc_paths, config=wave_config)
        self.last_start_record: int | None = None

    @classmethod
    def from_env(
        cls,
        env,
        nc_paths: str | Path | list[str | Path],
        *,
        ship_parameters_by_vessel: Mapping[str, ShipSpeedParameters] | None = None,
        default_ship_parameters: ShipSpeedParameters = NORTHERN_LIGHTS_SHIP,
        wave_config: RouteWaveConfig | None = None,
        config: ScenarioConfig | None = None,
        seed: int | None = None,
    ) -> "WaveHeightScenarioGenerator":
        return cls(
            nc_paths,
            routes=env._routes,
            ship_parameters_by_vessel=ship_parameters_by_vessel,
            default_ship_parameters=default_ship_parameters,
            wave_config=wave_config,
            config=config,
            seed=seed,
        )

    def sample(self, network, seed: int | None = None) -> Scenario:
        scenario = super().sample(network, seed=seed)
        if scenario.n_steps > self.reader.total_records:
            raise ValueError(
                f"Wave-height data has {self.reader.total_records} records, "
                f"but the scenario needs {scenario.n_steps} steps."
            )
        start_record = self._sample_start_record(seed, scenario.n_steps)
        self.last_start_record = start_record

        vessel_speed_factor: dict[str, list[float]] = {}
        for vessel_id in network._entities_of_type(Vessel):
            route = self.routes.get(vessel_id)
            if route is None:
                continue
            coordinates = route.get("coordinates")
            if not coordinates:
                continue
            parameters = self.ship_parameters_by_vessel.get(vessel_id, self.default_ship_parameters)
            nominal_speed_knots = float(route.get("speed_knots") or parameters.design_speed_knots)
            wave_heights = self.reader.route_wave_height_series(
                coordinates,
                start_record=start_record,
                hours=scenario.n_steps,
            )
            factors = speed_factor_series(
                wave_heights,
                parameters,
                nominal_speed_knots=nominal_speed_knots,
            )
            vessel_speed_factor[vessel_id] = [min(1.0, max(0.0, factor)) for factor in factors]

        if vessel_speed_factor:
            scenario.vessel_speed_factor = vessel_speed_factor
        return scenario

    def _sample_start_record(self, seed: int | None, n_steps: int) -> int:
        max_start = self.reader.total_records - n_steps
        if max_start <= 0:
            return 0
        episode_seed = seed if seed is not None else self.seed
        rng = random.Random(f"wave-height:{episode_seed}") if episode_seed is not None else random.Random()
        return rng.randint(0, max_start)
